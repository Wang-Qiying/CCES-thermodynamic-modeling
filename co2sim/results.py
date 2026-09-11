import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .caverns import STATE_NAMES
from .model import ENERGY_NAMES
from .constraints import evaluate_constraints, margin_summary
from .errors import ModelError


def write_json(path, obj):
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple,np.ndarray)): return [clean(x) for x in v]
        if isinstance(v, (float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.bool_): return bool(v)
        return v
    Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def export_results(model, stages, trial, logs, folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"parameters_snapshot.yaml").write_text(yaml.safe_dump(model.cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    write_json(folder/"events.json",logs)
    model.save_snapshot(folder/"initial_state.json",model.x0)
    samples, margins, profiles, warnings = [], [], [], []
    numerical = {"max_pressure_match_Pa":0.0,"max_HX_energy_residual_W":0.0,"max_HX_UA_residual_W_per_K":0.0,
                 "max_well_energy_residual_W":0.0,"max_water_energy_residual_W":0.0,"max_system_rate_residual_W":0.0}
    def sample(stage,t,y):
        mode = stage.mode
        try:
            st,wp,dx,de,diag = model.evaluate(mode,y[:8],stage.flow(y[:8]))
            rows = evaluate_constraints(model,mode,y[:8],st,wp,diag)
        except ModelError as exc:
            warnings.append(dict(time=t,mode=mode,kind=exc.kind,reason=str(exc)))
            st,wp,dx,de,diag = model.evaluate("idle",y[:8],0.0)
            rows = evaluate_constraints(model,"idle",y[:8],st,wp,diag)
            # State constraints remain assessable; failed active equipment is not certified.
            for r in rows:
                if r.group in ("equipment","water_hx","phase") and not r.active:
                    r.reason = "working point unavailable: "+str(exc)
        row = dict(time_s=float(t),time_h=float(t/3600),mode=mode,
                   record_type="actual" if stages else "preflight_candidate_not_operated",
                   **dict(zip(STATE_NAMES,y[:8])),
                   **dict(zip(ENERGY_NAMES,y[8:])),**diag)
        row["m_L"] = model.total_mass-y[0]
        row["M_w_cold"] = model.cfg["water"]["mass_total"]-y[5]
        for i in ("H","L"):
            row.update({f"{k}_{i}":getattr(st[i],k) for k in ("p","T","h","rho","u")})
        for k in ("q","mw","P_comp","P_gross","P_pump","P_other","P_grid","P_net","ratio","pressure_residual"):
            row[k] = wp[k]
        for j in range(1,7):
            sj = wp["points"].get(j)
            for k in ("p","T","h","rho"):
                row[f"point{j}_{k}"] = getattr(sj,k) if sj else np.nan
        hx = wp["hx"]
        row.update(Q_hx=hx.Q if hx else np.nan, water_return_T=hx.water_out_T if hx else np.nan,
                   hx_approach=hx.min_approach if hx else np.nan, UA_residual=hx.UA_residual if hx else np.nan)
        row["h_H_inj"] = wp["inject"].outlet.h if wp["mode"]=="charge" else np.nan
        row["h_L_inj"] = wp["inject"].outlet.h if wp["mode"]=="discharge" else np.nan
        for name in ("take","inject"):
            well = wp[name]
            row[f"{name}_well_heat"] = well.heat if well else np.nan
            row[f"{name}_well_out_p"] = well.outlet.p if well else np.nan
            row[f"{name}_well_out_T"] = well.outlet.T if well else np.nan
        row["stored_energy"] = model.stored_energy(y[:8])
        active = wp["mode"] != "idle"
        row["well_inventory"] = sum(wp[k].inventory for k in ("take","inject")) if active else np.nan
        row["well_inventory_fraction"] = row["well_inventory"]/model.total_mass
        row["max_well_mach"] = max(wp[k].max_mach for k in ("take","inject")) if active else np.nan
        row["max_well_velocity"] = max(wp[k].max_velocity for k in ("take","inject")) if active else np.nan
        row["max_well_kinetic_change_J_kg"] = max(abs(wp[k].kinetic_change) for k in ("take","inject")) if active else np.nan
        for group,(value,name) in margin_summary(rows).items():
            row[f"margin_{group}"] = value
            row[f"limiting_{group}"] = name
        for r in rows:
            margins.append(dict(time_s=t,time_h=t/3600,mode=mode,**r.row()))
        if active:
            numerical["max_pressure_match_Pa"] = max(numerical["max_pressure_match_Pa"],abs(wp["pressure_residual"]))
            numerical["max_HX_energy_residual_W"] = max(numerical["max_HX_energy_residual_W"],abs(hx.energy_residual))
            numerical["max_HX_UA_residual_W_per_K"] = max(numerical["max_HX_UA_residual_W_per_K"],abs(hx.UA_residual))
            numerical["max_well_energy_residual_W"] = max(numerical["max_well_energy_residual_W"],*(abs(wp[k].energy_residual) for k in ("take","inject")))
            profiles.append(dict(time_s=t,mode=mode,take=wp["take"].profile,inject=wp["inject"].profile,hx=hx.profile,
                                 pressure_bracket=wp["pressure_bracket"],rejected_pressure_trials=wp["trial_failures"]))
            if row["max_well_mach"] > model.cfg["well"]["mach_warning"]:
                warnings.append(dict(time=t,mode=mode,kind="model_applicability",reason="well Mach exceeds configured warning threshold"))
            frac = max(abs(wp[k].kinetic_change)/(model.cfg["environment"]["g"]*model.cfg["caverns"]["H"]["depth"]) for k in ("take","inject"))
            if frac > model.cfg["well"]["kinetic_head_warning_fraction"]:
                warnings.append(dict(time=t,mode=mode,kind="model_applicability",reason="neglected kinetic-energy change exceeds fraction of gravitational head"))
        cp, totalw = model.cfg["water"]["cp"], model.cfg["water"]["mass_total"]
        dwater = cp*(dx[5]*(y[6]-y[7])+y[5]*dx[6]+(totalw-y[5])*dx[7])
        water_expected = ((hx.Q*(1 if mode=="charge" else -1) if hx else 0)
                          -diag["tank_loss"]+diag["tank_conditioning"])
        numerical["max_water_energy_residual_W"] = max(numerical["max_water_energy_residual_W"],abs(dwater-water_expected))
        dsystem = dx[1]+dx[2]+model.geometry["H"]["C_rock"]*dx[3]+model.geometry["L"]["C_rock"]*dx[4]+dwater
        expected = de[3]+de[4]+de[5]-de[6]-de[9]+de[10]
        numerical["max_system_rate_residual_W"] = max(numerical["max_system_rate_residual_W"],abs(dsystem-expected))
        samples.append(row)

    for stage in stages:
        times = np.unique(np.r_[np.arange(stage.start,stage.end,model.cfg["numerics"]["output_interval"]),stage.end])
        for t in times:
            sample(stage,float(t),stage.at(t))
        model.save_snapshot(folder/(stage.mode+"_end.json"),stage.yend[:8],stage.end,stage.yend[8:])
    if not stages:
        sample(trial,trial.start,trial.y0)
    df, mf = pd.DataFrame(samples),pd.DataFrame(margins)
    if not stages:
        # Keep attempted algebraic values, including negative margins, for diagnosis.
        # The actual trajectory has only the unoperated initial state.
        df.to_csv(folder/"preflight_workpoint.csv",index=False,encoding="utf-8-sig")
        actual = df.copy()
        actual["mode"] = "not_started"
        actual["record_type"] = "initial_state_only"
        for k in ["q","mw","P_comp","P_gross","P_pump","P_other","P_grid","P_net"]:
            actual[k] = 0.0
        for k in actual.columns:
            if k.startswith("point") or k in ("ratio","pressure_residual","Q_hx","water_return_T","hx_approach","UA_residual","well_inventory","well_inventory_fraction","max_well_mach","max_well_velocity","max_well_kinetic_change_J_kg","h_H_inj","h_L_inj") or k.startswith("take_well_") or k.startswith("inject_well_") or k.startswith("margin_") or k.startswith("limiting_"):
                actual[k] = np.nan
        # Quasi-steady active-mode derivatives are candidate diagnostics, not actual evolution.
        for k in ("dp_H","dp_L","dT_H","dT_L","Q_H","Q_L","tank_loss","geo_heat","well_heat"):
            actual[k] = np.nan
        actual.to_csv(folder/"trajectory.csv",index=False,encoding="utf-8-sig")
    else:
        df.to_csv(folder/"trajectory.csv",index=False,encoding="utf-8-sig")
    mf.to_csv(folder/"constraints.csv",index=False,encoding="utf-8-sig")
    with open(folder/"component_profiles.jsonl","w",encoding="utf-8") as f:
        for p in profiles: f.write(json.dumps(p,ensure_ascii=False)+"\n")
    write_json(folder/"warnings.json",warnings)
    eligible = mf[mf.evaluated & mf.active]
    minima = eligible.loc[eligible.groupby("name")["normalized_margin"].idxmin()]
    minima.to_csv(folder/"constraint_minima.csv",index=False,encoding="utf-8-sig")
    unevaluated = mf[mf.active & ~mf.evaluated][["name","group","location","reason"]].drop_duplicates()
    unevaluated.to_csv(folder/"unevaluated_constraints.csv",index=False,encoding="utf-8-sig")
    yfinal = stages[-1].yend if stages else trial.y0
    energy = dict(zip(ENERGY_NAMES,yfinal[8:]))
    deltaU = model.stored_energy(yfinal[:8])-model.stored_energy(model.x0)
    thermal_input = energy["E_fluid_work"]+energy["E_geo"]+energy["E_well_heat"]-energy["E_tank_loss"]-energy["E_potential_transfer"]+energy["E_tank_conditioning"]
    electric_input = energy["E_grid"]-energy["E_net"]-energy["E_machine_loss"]-energy["E_aux"]+energy["E_geo"]+energy["E_well_heat"]-energy["E_tank_loss"]-energy["E_potential_transfer"]+energy["E_tank_conditioning"]
    E_in, E_out = energy["E_grid"],energy["E_net"]
    charged_mass = sum(abs(s.yend[0]-s.y0[0]) for s in stages if s.mode=="charge")
    discharged_mass = sum(abs(s.yend[0]-s.y0[0]) for s in stages if s.mode=="discharge")
    charge_stage = next((s for s in stages if s.mode=="charge"),None)
    initial_caverns = model.states(model.x0)
    charge_end_caverns = model.states(charge_stage.yend[:8]) if charge_stage is not None else initial_caverns
    H_cfg,L_cfg = model.cfg["caverns"]["H"],model.cfg["caverns"]["L"]
    H_isothermal_headroom = max(0.0,
        model.fluid.pt(H_cfg["p_max"],initial_caverns["H"].T).rho*H_cfg["volume"]-model.x0[0])
    L_isothermal_headroom = max(0.0,
        model.total_mass-model.x0[0]-model.fluid.pt(L_cfg["p_min"],initial_caverns["L"].T).rho*L_cfg["volume"])
    duration_mass_limit = model.cfg["operation"]["q_charge"]*model.cfg["operation"]["active_duration_max"]
    reference_mass_limit = min(H_isothermal_headroom,L_isothermal_headroom,duration_mass_limit)
    capacity_utilization = {
        "definition":"Screening metrics; cavern volume remains full of CO2, so utilization means cyclic working mass and pressure-window use.",
        "working_CO2_mass_kg":charged_mass,
        "total_CO2_inventory_kg":model.total_mass,
        "working_mass_fraction_of_inventory":charged_mass/model.total_mass if model.total_mass>0 else None,
        "H_isothermal_receiving_headroom_kg":H_isothermal_headroom,
        "L_isothermal_withdrawal_headroom_kg":L_isothermal_headroom,
        "duration_mass_limit_kg":duration_mass_limit,
        "reference_working_mass_limit_kg":reference_mass_limit,
        "reference_working_mass_utilization":charged_mass/reference_mass_limit if reference_mass_limit>0 else None,
        "H_charge_pressure_headroom_utilization":((charge_end_caverns["H"].p-initial_caverns["H"].p)/(H_cfg["p_max"]-initial_caverns["H"].p)
            if H_cfg["p_max"]>initial_caverns["H"].p else None),
        "L_charge_pressure_headroom_utilization":((initial_caverns["L"].p-charge_end_caverns["L"].p)/(initial_caverns["L"].p-L_cfg["p_min"])
            if initial_caverns["L"].p>L_cfg["p_min"] else None),
        "note":"Isothermal mass headrooms use initial cavern temperatures; the transient event solver determines the adopted cycle."}
    numerical.update(CO2_mass_error_kg=float(abs(df.m_H+df.m_L-model.total_mass).max()),
                     water_mass_error_kg=float(abs(df.M_w_hot+df.M_w_cold-model.cfg["water"]["mass_total"]).max()),
                     integrated_energy_residual_J=deltaU-thermal_input,
                     grid_boundary_energy_residual_J=deltaU-electric_input,
                     integrated_energy_residual_relative_to_purchase=abs(deltaU-thermal_input)/abs(E_in) if E_in!=0 else np.nan,
                     nonzero_duration_process_energy_balance_evaluated=bool(stages and stages[-1].end>0))
    state_df = actual if not stages else df
    has_discharge = any(s.mode=="discharge" and s.end>s.start for s in stages)
    summary = dict(status="completed_with_discharge" if has_discharge else "initial_charge_infeasible" if not stages else "no_effective_discharge",
        duration_h={mode:sum((s.end-s.start)/3600 for s in stages if s.mode==mode) for mode in ("charge","idle","discharge")},
        transferred_CO2_kg={mode:sum(abs(s.yend[0]-s.y0[0]) for s in stages if s.mode==mode) for mode in ("charge","idle","discharge")},
        energy_MWh={k:v/3.6e9 for k,v in energy.items()},
        cycle_closure={
            "charged_CO2_kg":charged_mass,
            "discharged_CO2_kg":discharged_mass,
            "CO2_transfer_mismatch_kg":charged_mass-discharged_mass,
            "discharge_to_charge_mass_ratio":discharged_mass/charged_mass if charged_mass>0 else None,
            "final_high_cavern_mass_minus_initial_kg":float(yfinal[0]-model.x0[0]),
            "final_hot_water_mass_minus_initial_kg":float(yfinal[5]-model.x0[5]),
            "mass_balanced":bool(charged_mass>0 and abs(charged_mass-discharged_mass)<=max(1e-6,charged_mass*1e-9)),
            "note":"Mass-balanced first cycle; a strict periodic cycle also requires thermal states to converge."},
        capacity_utilization=capacity_utilization,
        electric_loss_assessment={
            "compressor_fluid_work_MWh":energy.get("E_comp_fluid",0)/3.6e9,
            "compressor_electricity_MWh":energy.get("E_comp_electric",0)/3.6e9,
            "compressor_motor_mechanical_loss_MWh":(energy.get("E_comp_electric",0)-energy.get("E_comp_fluid",0))/3.6e9,
            "charge_water_pump_MWh":energy.get("E_pump_charge",0)/3.6e9,
            "charge_other_auxiliary_MWh":energy.get("E_other_charge",0)/3.6e9,
            "turbine_fluid_work_MWh":energy.get("E_turbine_fluid",0)/3.6e9,
            "generator_gross_MWh":energy.get("E_turbine_gross",0)/3.6e9,
            "turbine_generator_mechanical_loss_MWh":(energy.get("E_turbine_fluid",0)-energy.get("E_turbine_gross",0))/3.6e9,
            "discharge_water_pump_MWh":energy.get("E_pump_discharge",0)/3.6e9,
            "discharge_other_auxiliary_MWh":energy.get("E_other_discharge",0)/3.6e9,
            "net_discharge_MWh":E_out/3.6e9,
            "generic_auxiliary_assumption":"Zero in the thermodynamic baseline; only explicit water pumps and motor/generator/mechanical losses are counted."},
        heat_recovery_assessment={
            "charge_CO2_to_water_MWh_th":energy.get("E_HX_charge",0)/3.6e9,
            "charge_water_pump_MWh_e":energy.get("E_pump_charge",0)/3.6e9,
            "requested_heat_minus_pump_MWh_equivalent":(energy.get("E_HX_charge",0)-energy.get("E_pump_charge",0))/3.6e9,
            "requested_metric_positive":bool(energy.get("E_HX_charge",0)>energy.get("E_pump_charge",0)),
            "pump_fraction_of_recovered_heat":(
                energy.get("E_pump_charge", 0) / energy.get("E_HX_charge", 0)
                if energy.get("E_HX_charge", 0) > 0
                else None
            ),
            "discharge_reheat_MWh_th":energy.get("E_HX_discharge",0)/3.6e9,
            "discharge_water_pump_MWh_e":energy.get("E_pump_discharge",0)/3.6e9,
            "total_water_pump_MWh_e":(energy.get("E_pump_charge",0)+energy.get("E_pump_discharge",0))/3.6e9,
            "charge_heat_minus_all_cycle_pumps_MWh_equivalent":(energy.get("E_HX_charge",0)-energy.get("E_pump_charge",0)-energy.get("E_pump_discharge",0))/3.6e9,
            "all_cycle_pump_fraction_of_recovered_heat":(
                (energy.get("E_pump_charge", 0) + energy.get("E_pump_discharge", 0))
                / energy.get("E_HX_charge", 0)
                if energy.get("E_HX_charge", 0) > 0
                else None
            ),
            "conditioning_charge_MWh_th":energy.get("E_conditioning_charge",0)/3.6e9,
            "conditioning_idle_MWh_th":energy.get("E_conditioning_idle",0)/3.6e9,
            "conditioning_discharge_MWh_th":energy.get("E_conditioning_discharge",0)/3.6e9,
            "interpretation":"Thermal heat and electricity are shown as energy-equivalent MWh but have different thermodynamic quality. Fixed-temperature tank conditioning is external and excluded from recovered heat."},
        charge_purchase_MWh=(E_in-energy["E_idle"])/3.6e9,
        electricity_recovery_ratio=E_out/E_in if E_in>0 else None,
        net_energy_density_kWh_m3=E_out/3.6e6/sum(c["volume"] for c in model.cfg["caverns"].values()),
        initial_state=state_df.iloc[0].to_dict(),final_state=state_df.iloc[-1].to_dict(),
        critical={"p":model.fluid.pcrit,"T":model.fluid.Tcrit,
                  "pressure_operating_min":model.fluid.pcrit+model.cfg["phase"]["pressure_above_critical"],
                  "temperature_operating_min":model.fluid.Tcrit+model.cfg["phase"]["temperature_above_critical"]},
        final_minus_initial_state={k:float(yfinal[i]-model.x0[i]) for i,k in enumerate(STATE_NAMES)},
        stored_energy_change_J=deltaU, numerical_diagnostics=numerical,
        events=logs, unevaluated_constraints=unevaluated.to_dict("records"),
        rho_reference_kg_m3=model.rho_ref, geometry=model.geometry,
        neglected_well_inventory_fraction_range=[df.well_inventory_fraction.min(),df.well_inventory_fraction.max()],
        estimated_surface_inventory_per_m3_kg=[df[[f"point{i}_rho" for i in range(1,7)]].min().min(),df[[f"point{i}_rho" for i in range(1,7)]].max().max()],
        surface_inventory_volume_m3=model.cfg["engineering"]["surface_CO2_inventory_volume"],
        max_well_mach=df.max_well_mach.max(),
        margin_scope="Only evaluated and stage-applicable constraints; normalization affects ranking; not geological certification",
        data_scope="Initial working-point feasibility check only; no physical stage executed" if not stages else "Actual adopted stages only",
        energy_boundary="CO2 internal + equivalent rock sensible + both isothermal water tanks; PE transfer accounted; well/geo ambient heat and explicit tank conditioning included; pump, auxiliaries and motor/generator/mechanical losses rejected outside modeled thermal stores")
    write_json(folder/"summary.json",summary)
    write_json(folder/"numerical_diagnostics.json",numerical)
    table = []
    for mode in ("charge","idle","discharge"):
        table.append(dict(metric=mode+"_duration_h",value=summary["duration_h"].get(mode,0)))
    table += [dict(metric=k,value=summary[k]) for k in ("charge_purchase_MWh","electricity_recovery_ratio","net_energy_density_kWh_m3")]
    table += [dict(metric=k,value=v) for k,v in capacity_utilization.items() if isinstance(v,(int,float,np.integer,np.floating))]
    table += [dict(metric=k+"_MWh",value=v) for k,v in summary["energy_MWh"].items()]
    pd.DataFrame(table).to_csv(folder/"summary_table.csv",index=False,encoding="utf-8-sig")
    return df,mf,summary
