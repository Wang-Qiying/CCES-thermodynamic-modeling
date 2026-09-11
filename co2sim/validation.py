"""Numerical refinements and isolated module tests, never relaxed operating runs."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp,quad
from scipy.optimize import brentq
from .model import Model, ENERGY_NAMES
from .wells import map_well
from .water import tank_derivatives
from .constraints import evaluate_constraints
from .integrate import run_stage,run_process
from .results import write_json
from .machines import compress,expand
from .exchangers import exchange
from .errors import ModeInapplicable,NoPhysicalSolution


def isolated_component_checks(model):
    """Synthetic component fixtures, not alternate system initializations/cycles."""
    f,c=model.fluid,model.cfg
    inlet=f.pt(12000000.0,350.0)
    out,W,P=expand(f,inlet,9000000.0,100.0,c)
    ideal=f.ps(9000000.0,inlet.s)
    turbine=dict(eta_error=(inlet.h-out.h)/(inlet.h-ideal.h)-c["turbine"]["eta_is"],
                 isentropic_entropy_error_J_kgK=ideal.s-inlet.s,
                 fluid_energy_residual_W=W-100*(inlet.h-out.h),
                 electrical_conversion_residual_W=P-W*c["turbine"]["eta_generator"]*c["turbine"]["eta_mech"])
    cold=f.pt(10000000.0,330.0)
    hot_water=360.0
    hx=exchange(f,cold,100.0,hot_water,"discharge",c,model.rho_ref)
    heating=dict(scope="Isolated reheater fixture: inlet CO2 10 MPa/330 K, water 360 K; not a system run or adopted inventory",
                 Q_W=hx.Q,energy_residual_W=hx.energy_residual,UA_residual_W_K=hx.UA_residual,
                 water_inlet_pairing_error_K=hx.profile[-1]["T_water"]-hot_water,
                 CO2_inlet_pairing_error_K=hx.profile[0]["T_CO2"]-cold.T,
                 water_outlet_pairing_error_K=hx.profile[0]["T_water"]-hx.water_out_T)
    idle=model.workpoint("idle",model.x0,0)
    wrong_direction=False
    try: exchange(f,cold,100.0,303.15,"discharge",c,model.rho_ref)
    except ModeInapplicable: wrong_direction=True
    zero_well=False
    try: map_well(f,cold,0.0,1200.0,True,c)
    except NoPhysicalSolution: zero_well=True
    return dict(turbine=turbine,reheater=heating,wrong_heat_direction_classified=wrong_direction,
                zero_flow_well_not_evaluated=zero_well,idle_components_not_applicable=idle["take"] is None and idle["hx"] is None,
                scope="Isolated numerical module verification only; no reset or replacement of actual tank/CO2 inventories")


def independent_hx_reference(model, wp):
    """Independent adaptive quadrature of continuous dQ/d(Delta T), fixed inlet.

    Unlike the production piecewise-LMTD method this integrates 1/DeltaT directly.
    Its result is a check of spatial discretization, not a substitute operating HX.
    """
    f,c = model.fluid,model.cfg
    inlet = wp["points"][2]
    pout = wp["hx"].outlet.p
    q = wp["q"]
    Tw = model.x0[7]
    Cw = wp["mw"]*c["water"]["cp"]
    def delta(x,Q):
        Tco = f.temperature_ph(inlet.p+(pout-inlet.p)*x,inlet.h-Q*x/q)
        return Tco-(Tw+Q*(1-x)/Cw)
    cap = min(Cw*(inlet.T-Tw),q*(inlet.h-f.pt(pout,Tw).h))
    # Avoid evaluating arbitrarily close to a zero-temperature-difference singularity.
    upper = wp["hx"].Q+.5*(cap-wp["hx"].Q)
    def ua(Q):
        value,error = quad(lambda x:1/delta(x,Q),0,1,epsabs=1e-8,epsrel=1e-8,limit=200)
        return Q*value
    Qref = brentq(lambda Q:ua(Q)-c["exchangers"]["charge"]["UA"],0,upper,xtol=1e-4)
    return dict(Q_adaptive_quadrature_W=Qref,Q_segmented_W=wp["hx"].Q,
                relative_difference=(wp["hx"].Q-Qref)/Qref,
                reference_min_approach_K=min(delta(x,Qref) for x in np.linspace(0,1,1001)),
                UA_at_segmented_Q_W_per_K=ua(wp["hx"].Q),
                UA_continuous_residual_W_per_K=ua(wp["hx"].Q)-c["exchangers"]["charge"]["UA"])


def validate(model,stages,trial,summary,folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    c = model.cfg
    checks = {}
    checks["isolated_active_components"] = isolated_component_checks(model)
    print("Validation: EOS round trip, well, sealed cavern, tanks, snapshot and boundary handling",flush=True)
    st0 = model.states(model.x0)
    checks["EOS_roundtrip"] = {i:{"pressure_error_Pa":st0[i].p-c["caverns"][i]["p_initial"],
                                     "temperature_error_K":st0[i].T-c["caverns"][i]["T_initial"]} for i in ("H","L")}
    cw = deepcopy(c)
    cw["well"].update(U=0.0,friction=False)
    adwell = map_well(model.fluid,st0["L"],100.0,c["caverns"]["L"]["depth"],True,cw)
    conserved = np.array([p["h"]+c["environment"]["g"]*p["z"] for p in adwell.profile])
    checks["adiabatic_frictionless_well"] = dict(max_h_plus_gz_deviation_J_kg=float(abs(conserved-conserved[0]).max()),
                                                energy_residual_W=adwell.energy_residual)
    sealed = deepcopy(c)
    sealed["rock"].update(alpha=0.0,conductivity=0.0)
    sealed["water"]["loss_conductance"] = 0.0
    sm = Model(sealed)
    sol = solve_ivp(lambda t,x:sm.evaluate("idle",x,0)[2],(0,3600),sm.x0,rtol=1e-9,atol=c["numerics"]["atol"],max_step=30)
    checks["sealed_adiabatic_constant_volume"] = dict(mass_change_kg=float(sol.y[0,-1]-sm.x0[0]),
        H_energy_change_J=float(sol.y[1,-1]-sm.x0[1]),L_energy_change_J=float(sol.y[2,-1]-sm.x0[2]),integration_success=sol.success)
    # Independent closed-form balances for a synthetic adiabatic receiving tank.
    tc = deepcopy(c); tc["water"]["loss_conductance"] = 0.0
    tc["water"]["fixed_temperatures"] = False
    class Return:
        water_out_T = 333.15
    wp = {"mw":25.0,"hx":Return()}
    xinitial = model.x0.copy()
    def twrhs(t,y):
        x = xinitial.copy(); x[5:8] = y
        return tank_derivatives(x,"charge",wp,tc)[0]
    solw = solve_ivp(twrhs,(0,1800),xinitial[5:8],rtol=1e-10,atol=[1e-5,1e-9,1e-9],max_step=30)
    Mfinal = xinitial[5]+25*1800
    Texact = (xinitial[5]*xinitial[6]+25*1800*333.15)/Mfinal
    checks["mixed_receiving_tank_analytic"] = dict(mass_error_kg=float(solw.y[0,-1]-Mfinal),
         temperature_error_K=float(solw.y[1,-1]-Texact),cold_withdrawal_temperature_change_K=float(solw.y[2,-1]-xinitial[7]))
    # Cavern module balance with prescribed inlet enthalpy, not a system operating run.
    mass0,U0 = model.x0[:2]
    qtest,hin = 100.0,st0["H"].h
    isolated = solve_ivp(lambda t,y:[qtest,qtest*hin],(0,600),[mass0,U0],rtol=1e-10,atol=[1e-5,1],max_step=30)
    checks["cavern_open_balance_analytic"] = dict(mass_error_kg=float(isolated.y[0,-1]-(mass0+qtest*600)),
           energy_error_J=float(isolated.y[1,-1]-(U0+qtest*hin*600)))
    snapshot = folder/"validation_snapshot.json"
    model.save_snapshot(snapshot,model.x0,123.0,np.zeros(len(ENERGY_NAMES)))
    restored,xrest,t,erest = Model.load_snapshot(snapshot)
    checks["snapshot_roundtrip"] = dict(max_state_absolute_error=float(abs(xrest-model.x0).max()),time_error_s=t-123.0,
                                       mass_error_kg=restored.total_mass-model.total_mass)
    idle30 = run_stage(model,"idle",model.x0,0,7200,progress=False)
    cc = deepcopy(c); cc["numerics"]["max_step"] = c["numerics"]["max_step"]/2
    fine = Model(cc)
    idle15 = run_stage(fine,"idle",fine.x0,0,7200,progress=False)
    checks["idle_boundary_and_time_step"] = dict(scope="Isolated idle module regression; not part of actual process",
        coarse_termination=idle30.event,fine_termination=idle15.event,
        max_state_difference_by_name=dict(zip(["m_H","U_H","U_L","T_r_H","T_r_L","M_w_hot","T_w_hot","T_w_cold"],abs(idle30.yend[:8]-idle15.yend[:8]))),
        note="Initial hot inventory is exactly its lower bound and stationary; integration must reach 2 h")
    resumed = run_stage(restored,"idle",xrest,0,3600,start=t,energy0=erest,progress=False)
    continuation = run_stage(restored,"idle",resumed.yend[:8],0,3600,start=resumed.end,energy0=resumed.yend[8:],progress=False)
    checks["snapshot_continuation"] = dict(max_state_difference_vs_uninterrupted=float(abs(continuation.yend[:8]-idle30.yend[:8]).max()),
        energy_difference_J=float(abs(continuation.yend[8:]-idle30.yend[8:]).max()))
    basewp = model.workpoint("charge",model.x0,c["operation"]["q_charge"])
    checks["HX_independent_continuous_quadrature"] = independent_hx_reference(model,basewp)
    spatial=[]
    print("Validation: same initial state, spatial refinements (no extra operating case)",flush=True)
    for factor in (1,2,4):
        sc = deepcopy(c)
        sc["well"]["segments"] = c["well"]["segments"]*factor
        sc["exchangers"]["segments"] = c["exchangers"]["segments"]*factor
        mm = Model(sc)
        st,ww,dx,de,dd = mm.evaluate("charge",mm.x0,c["operation"]["q_charge"])
        spatial.append(dict(well_segments=sc["well"]["segments"],hx_segments=sc["exchangers"]["segments"],
            Q_W=ww["hx"].Q,approach_K=ww["hx"].min_approach,compressor_power_W=ww["P_comp"],
            p2_Pa=ww["points"][2].p,pressure_residual_Pa=ww["pressure_residual"],
            min_normalized_margin=min(r.normalized_margin for r in evaluate_constraints(mm,"charge",mm.x0,st,ww,dd) if r.active and r.evaluated)))
        print(spatial[-1],flush=True)
    pd.DataFrame(spatial).to_csv(folder/"spatial_convergence.csv",index=False,encoding="utf-8-sig")
    checks["spatial_convergence_initial_workpoint"] = spatial
    checks["spatial_change_coarse_to_finest"] = {k:spatial[0][k]-spatial[-1][k] for k in ("Q_W","approach_K","compressor_power_W","p2_Pa")}
    # The earlier lower-flow pinch debug applies only when the adopted charge
    # trajectory is infeasible.  For a completed process it would be an extra
    # operating case and can also leave the pressure-matching problem's physical
    # domain, so record why it is intentionally not run.
    checks["finite_flow_debug"] = dict(
        evaluated=False,
        reason="The adopted charge trajectory is feasible; no lower-flow diagnostic operating case is needed.",
        adopted=False,
        attempt_count=0)
    if stages:
        print("Validation: full adopted process temporal/spatial convergence",flush=True)
        refined=[]
        for label,factor_t,factor_s in [("time",.5,1),("space",1,2)]:
            cc=deepcopy(c)
            cc["numerics"]["max_step"]*=factor_t
            cc["well"]["segments"]*=factor_s
            cc["exchangers"]["segments"]*=factor_s
            mm=Model(cc)
            ss,tt,ll=run_process(mm)
            rr=dict(refinement=label,events=ll)
            if ss:
                rr["final_state_difference"]=(ss[-1].yend[:8]-stages[-1].yend[:8]).tolist()
                rr["energy_difference_J"]=(ss[-1].yend[8:]-stages[-1].yend[8:]).tolist()
                rr["duration_difference_s"]={s.mode:s.end-s.start-next(b.end-b.start for b in stages if b.mode==s.mode) for s in ss}
                # Compare common physical times separately from event-time shifts.
                rr["common_time_max_state_difference"]={}
                for ss1,ss2 in zip(stages,ss):
                    lo,hi=max(ss1.start,ss2.start),min(ss1.end,ss2.end)
                    if hi>=lo:
                        diffs=[abs(ss1.at(t)[:8]-ss2.at(t)[:8]) for t in np.linspace(lo,hi,25)]
                        rr["common_time_max_state_difference"][ss1.mode]=np.max(diffs,axis=0).tolist()
            refined.append(rr)
        checks["trajectory_convergence"] = refined
    else:
        checks["trajectory_convergence"] = dict(evaluated=False,
            reason="No feasible nonzero-duration baseline trajectory exists. A time-step comparison of a three-stage trajectory is not applicable; initial-workpoint spatial refinement and independent isolated idle time refinement are reported separately.")
    checks["sampled_balance_checks"] = summary["numerical_diagnostics"]
    checks["criteria"] = dict(EOS_pressure_roundtrip_Pa=0.1,EOS_temperature_roundtrip_K=1e-6,
         well_h_plus_gz_J_kg=1e-6,water_analytic_temperature_K=1e-6,pressure_match_Pa=0.1,
         HX_relative_continuous_Q_difference=0.001,system_rate_residual_W=0.001,
         purpose="Numerical acceptance tolerances, not physical safety thresholds")
    checks["module_checks_passed"] = bool(
        max(abs(v["pressure_error_Pa"]) for v in checks["EOS_roundtrip"].values())<.1 and
        checks["adiabatic_frictionless_well"]["max_h_plus_gz_deviation_J_kg"]<1e-6 and
        checks["sealed_adiabatic_constant_volume"]["H_energy_change_J"]==0 and
        abs(checks["mixed_receiving_tank_analytic"]["temperature_error_K"])<1e-6 and
        idle30.event["kind"]=="duration_limit" and idle15.event["kind"]=="duration_limit" and
        abs(checks["HX_independent_continuous_quadrature"]["relative_difference"])<.001 and
        summary["numerical_diagnostics"]["max_system_rate_residual_W"]<.001 and
        abs(checks["isolated_active_components"]["reheater"]["energy_residual_W"])<.001 and
        abs(checks["isolated_active_components"]["turbine"]["eta_error"])<1e-8 and
        checks["isolated_active_components"]["wrong_heat_direction_classified"] and
        checks["isolated_active_components"]["idle_components_not_applicable"])
    write_json(folder/"validation.json",checks)
    print("Validation saved; module_checks_passed =",checks["module_checks_passed"],flush=True)
    return checks
