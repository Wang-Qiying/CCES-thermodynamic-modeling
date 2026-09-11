"""Reproducible PESIM application studies for the two-cavern sCO2 model."""
from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from co2sim.constraints import evaluate_constraints, margin_summary
from co2sim.caverns import STATE_NAMES
from co2sim.errors import ModelError
from co2sim.flowpath import solve_flow_for_power
from co2sim.integrate import Stage, run_process, run_stage
from co2sim.model import ENERGY_NAMES, Model, load_config
from co2sim.results import export_results, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "case_config.yaml"
DEFAULT_RESULTS = HERE / "results"


def _write_turbine_rating_decision(cfg, state=None, state_label="representative charged state",
                                   folder=DEFAULT_RESULTS):
    """Persist the pre-simulation rating decision for paper traceability."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = Model(copy.deepcopy(cfg))
    rating_state = model.x0 if state is None else np.asarray(state, dtype=float)
    rows = []
    for q in (50.0, 100.0, 250.0, 275.0, 500.0):
        power = model.net_power(rating_state, q)/1e6
        rows.append({
            "flow_kg_s": q, "representative_net_power_MW": power,
            "exceeds_5MW_rating": power > 5.0,
            "within_selected_10MW_rating": power <= 10.0,
        })
    table = pd.DataFrame(rows)
    table.to_csv(folder / "turbine_rating_decision.csv", index=False, encoding="utf-8-sig")
    write_json(folder / "turbine_rating_decision.json", {
        "original_rating_MW": 5.0, "selected_rating_MW": 10.0,
        "representative_state": state_label,
        "reason": (
            "The 5 MW rating truncates the configured 50-500 kg/s flow range near 250 kg/s; "
            "10 MW retains the intended range while remaining above the calculated q=500 kg/s point."
        ),
        "basis": rows,
    })
    return table


def _plot_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans") if n in available]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fonts,
        "axes.unicode_minus": False, "font.size": 8.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def _save_figure(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_case_config(config_path=DEFAULT_CONFIG):
    cfg = load_config(config_path)
    if not math.isclose(cfg["turbine"]["power_max"], 10e6):
        raise ValueError("PESIM case configuration must use the synchronized 10 MW turbine rating")
    if cfg["water"].get("enabled", True):
        raise ValueError("The PESIM base case is defined without water thermal storage")
    return cfg


def _shift_stage(stage: Stage, time_offset: float, energy_offset: np.ndarray):
    eoff = np.asarray(energy_offset, dtype=float).copy()

    def shift_y(y):
        z = np.asarray(y, dtype=float).copy()
        z[8:] += eoff
        return z

    shifted_segments = []
    for a, b, dense in stage.segments:
        def shifted_dense(t, dense=dense, time_offset=time_offset):
            return shift_y(dense(t-time_offset))
        shifted_segments.append((a+time_offset, b+time_offset, shifted_dense))
    event = dict(stage.event)
    event["time"] = float(event.get("time", stage.end)+time_offset)
    return Stage(stage.mode, stage.q, stage.start+time_offset, stage.end+time_offset,
                 shift_y(stage.y0), shift_y(stage.yend), shifted_segments, event)


def _run_active_stage_with_idle_check(cfg, mode, x_start, energy0, start, progress=False):
    """Run to the margin floor, then verify a full 2-h idle at the 2% floor."""
    op = cfg["operation"]
    requested_floor = float(op.get("endpoint_margin_floor", 0.02))
    increment = float(op.get("endpoint_margin_increment", 0.01))
    maximum_floor = float(op.get("endpoint_margin_max", 0.20))
    q = op["q_charge"] if mode == "charge" else op["q_discharge"]
    attempts = []
    for floor in np.arange(requested_floor, maximum_floor+0.5*increment, increment):
        floor = float(round(floor, 12))
        model = Model(copy.deepcopy(cfg))
        model.x0 = np.asarray(x_start, dtype=float).copy()
        active = run_stage(
            model, mode, np.asarray(x_start, dtype=float), q,
            op["active_duration_max"], start=start, energy0=energy0,
            progress=progress, constraint_margin_floor=floor,
        )
        if active.event["kind"] != "margin_floor":
            attempts.append({"floor": floor, "active_event": active.event})
            continue
        idle = run_stage(
            model, "idle", active.yend[:8], 0.0, op["idle_duration"],
            start=active.end, energy0=active.yend[8:], progress=progress,
            constraint_margin_floor=requested_floor,
        )
        attempts.append({"floor": floor, "active_event": active.event, "idle_event": idle.event})
        if idle.event["kind"] == "duration_limit":
            active.event["requested_margin_floor"] = requested_floor
            active.event["adopted_margin_floor"] = floor
            idle.event["verified_margin_floor"] = requested_floor
            return model, active, idle, floor, attempts
    raise RuntimeError(
        f"No {mode}-idle pair retained the requested normalized margin after 2 h: "
        + json.dumps(attempts, ensure_ascii=False)
    )


def _build_repeated_cycles(cfg, count=3, save_root: Path | None = None, progress=True):
    op = cfg["operation"]
    requested_floor = float(op.get("endpoint_margin_floor", 0.02))
    original_model = Model(copy.deepcopy(cfg))
    precondition = run_stage(
        original_model, "discharge", original_model.x0, op["q_discharge"],
        op["active_duration_max"], progress=progress,
        constraint_margin_floor=requested_floor,
    )
    if precondition.event["kind"] != "margin_floor":
        raise RuntimeError(
            "Initial-state discharge did not reach the requested margin floor: "
            + json.dumps(precondition.event, ensure_ascii=False)
        )
    precondition.event["role"] = "initial_state_correction"
    corrected_initial_x = precondition.yend[:8].copy()
    if save_root is not None:
        export_results(
            original_model, [precondition], precondition, [precondition.event],
            save_root / "initial_discharge",
        )
        write_json(save_root / "corrected_initial_state.json", {
            "source": "configured initial state",
            "procedure": "fixed-flow discharge to normalized margin floor",
            "normalized_margin_floor": requested_floor,
            "event": precondition.event,
            "state_names": STATE_NAMES,
            "x": corrected_initial_x.tolist(),
            "duration_h": (precondition.end-precondition.start)/3600,
            "transferred_mass_t": abs(precondition.yend[0]-precondition.y0[0])/1e3,
        })

    base = Model(copy.deepcopy(cfg))
    base.x0 = corrected_initial_x.copy()
    x_current = corrected_initial_x.copy()
    energy_offset = np.zeros(len(ENERGY_NAMES))
    time_offset = 0.0
    combined_stages, combined_logs, records, cycles = [], [], [], []

    for number in range(1, count+1):
        if progress:
            print(f"Cycle {number}/{count}")
        zero_energy = np.zeros(len(ENERGY_NAMES))
        _, charge, pre_idle, charge_floor, charge_attempts = _run_active_stage_with_idle_check(
            cfg, "charge", x_current, zero_energy, 0.0, progress=progress,
        )
        _, discharge, post_idle, discharge_floor, discharge_attempts = _run_active_stage_with_idle_check(
            cfg, "discharge", pre_idle.yend[:8], pre_idle.yend[8:], pre_idle.end,
            progress=progress,
        )
        model = Model(copy.deepcopy(cfg))
        model.x0 = np.asarray(x_current, dtype=float).copy()
        stages = [charge, pre_idle, discharge, post_idle]
        logs = [
            dict(charge.event, role="charge_margin_endpoint"),
            dict(pre_idle.event, role="pre_discharge_idle_check"),
            dict(discharge.event, role="discharge_margin_endpoint"),
            dict(post_idle.event, role="post_discharge_idle_check"),
        ]
        if save_root is not None:
            cycle_folder = save_root / "per_cycle" / f"cycle_{number:02d}"
            _, _, summary = export_results(model, stages, charge, logs, cycle_folder)
        else:
            summary = None

        for stage in stages:
            combined_stages.append(_shift_stage(stage, time_offset, energy_offset))
        for log in logs:
            shifted = dict(log)
            shifted["time"] = float(log.get("time", 0.0)+time_offset)
            shifted["cycle"] = number
            combined_logs.append(shifted)

        start_states = model.states(model.x0)
        discharge_end_states = model.states(stages[2].yend[:8])
        post_idle_states = model.states(stages[-1].yend[:8])
        energy = dict(zip(ENERGY_NAMES, stages[-1].yend[8:]))
        E_in, E_out = energy["E_grid"], energy["E_net"]
        charge_mass = abs(charge.yend[0]-charge.y0[0])
        discharge_mass = abs(discharge.yend[0]-discharge.y0[0])
        mean_working_mass = 0.5*(charge_mass+discharge_mass)
        record = {
            "cycle": number,
            "requested_endpoint_margin": requested_floor,
            "adopted_charge_margin": charge_floor,
            "adopted_discharge_margin": discharge_floor,
            "charge_h": (stages[0].end-stages[0].start)/3600,
            "pre_discharge_idle_h": (stages[1].end-stages[1].start)/3600,
            "discharge_h": (stages[2].end-stages[2].start)/3600,
            "post_discharge_idle_h": (stages[3].end-stages[3].start)/3600,
            "charge_mass_t": charge_mass/1e3,
            "discharge_mass_t": discharge_mass/1e3,
            "working_mass_t": mean_working_mass/1e3,
            "charge_discharge_mass_difference_pct": (
                100.0*(charge_mass-discharge_mass)/mean_working_mass
                if mean_working_mass > 0 else np.nan
            ),
            "charge_input_MWh": E_in/3.6e9,
            "net_discharge_MWh": E_out/3.6e9,
            "single_cycle_efficiency": E_out/E_in if E_in > 0 else np.nan,
            "charge_limiting_constraint": ";".join(charge.event.get("constraints", [])),
            "discharge_limiting_constraint": ";".join(discharge.event.get("constraints", [])),
            "p_H_start_MPa": start_states["H"].p/1e6,
            "p_L_start_MPa": start_states["L"].p/1e6,
            "p_H_discharge_end_MPa": discharge_end_states["H"].p/1e6,
            "p_L_discharge_end_MPa": discharge_end_states["L"].p/1e6,
            "p_H_after_idle_MPa": post_idle_states["H"].p/1e6,
            "p_L_after_idle_MPa": post_idle_states["L"].p/1e6,
            "T_H_after_idle_C": post_idle_states["H"].T-273.15,
            "T_L_after_idle_C": post_idle_states["L"].T-273.15,
            "mass_return_error_kg": float(stages[2].yend[0]-stages[0].y0[0]),
        }
        records.append(record)
        cycles.append({
            "model": model, "stages": stages, "logs": logs,
            "summary": summary, "record": record,
            "charge_attempts": charge_attempts,
            "discharge_attempts": discharge_attempts,
        })

        energy_offset += stages[-1].yend[8:]
        time_offset += stages[-1].end
        x_current = stages[-1].yend[:8].copy()

    return {
        "base_model": base, "cycles": cycles, "stages": combined_stages,
        "logs": combined_logs, "records": records,
        "precondition_stage": precondition,
    }


def _stage_spans(axes, stages):
    colors = {"charge": "#EAF2FB", "idle": "#F1F1F1", "discharge": "#FCEFE5"}
    for ax in axes:
        for stage in stages:
            ax.axvspan(stage.start/3600, stage.end/3600, color=colors[stage.mode], zorder=-10)
        ax.grid(alpha=0.20)


def _plot_three_cycle_results(df, stages, cycle_table, cfg, folder):
    _plot_style()
    t = df.time_h.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 8.8), sharex=True, layout="constrained")
    colors = {"H": "#1767A4", "L": "#DC7434"}
    for name in ("H", "L"):
        axes[0].plot(t, df[f"p_{name}"]/1e6, color=colors[name], lw=1.7, label=f"{name}穴")
        axes[0].axhline(cfg["caverns"][name]["p_min"]/1e6, color=colors[name], ls=":", lw=0.9)
        axes[0].axhline(cfg["caverns"][name]["p_max"]/1e6, color=colors[name], ls="-.", lw=0.9)
        axes[1].plot(t, df[f"T_{name}"]-273.15, color=colors[name], lw=1.7, label=f"CO2 {name}")
        axes[1].plot(t, df[f"T_r_{name}"]-273.15, color=colors[name], ls="--", lw=1.0, label=f"围岩 {name}")
    axes[1].axhline(304.1282000029807+cfg["phase"]["temperature_above_critical"]-273.15,
                    color="#222222", ls=":", lw=1.0, label="超临界温度下限")
    axes[2].plot(t, df.P_grid/1e6, color="#386CB0", lw=1.5, label="充电输入")
    axes[2].plot(t, df.P_net/1e6, color="#1B9E77", lw=1.5, label="净放电输出")
    for key, label, color in [
        ("margin_cavern_pressure", "穴压", "#1767A4"),
        ("margin_phase", "相态", "#D95F02"),
        ("margin_equipment", "设备", "#7570B3"),
        ("margin_all_evaluated_applicable", "综合最小", "#9B273B"),
    ]:
        axes[3].plot(t, df[key], lw=1.2, label=label, color=color)
    axes[3].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("穴压 / MPa")
    axes[1].set_ylabel("温度 / °C")
    axes[2].set_ylabel("功率 / MW")
    axes[3].set_ylabel("热力学可行裕度 μ")
    axes[3].set_xlabel("累计时间 / h")
    for ax in axes:
        ax.legend(ncol=4, fontsize=7.2, loc="best")
    _stage_spans(axes, stages)
    fig.suptitle("三个连续充放电周期的状态、功率与热力学可行裕度", fontsize=12)
    _save_figure(fig, folder, "fig_three_cycle_dynamics")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), layout="constrained")
    x = np.arange(len(cycle_table))
    width = 0.34
    axes[0].bar(x-width/2, cycle_table.charge_input_MWh, width, label="充电输入")
    axes[0].bar(x+width/2, cycle_table.net_discharge_MWh, width, label="净放电输出")
    axes[0].set_xticks(x, [f"周期 {i}" for i in cycle_table.cycle])
    axes[0].set_ylabel("电量 / MWh")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=.2)
    axes[1].plot(cycle_table.cycle, 100*cycle_table.single_cycle_efficiency, marker="o", label="单次循环效率")
    axes[1].set_xticks(cycle_table.cycle)
    axes[1].set_xlabel("周期")
    axes[1].set_ylabel("单次循环效率 / %")
    axes[1].grid(alpha=.2)
    axes[1].set_title("初末热状态不完全相同")
    _save_figure(fig, folder, "fig_cycle_repeatability")


def run_three_cycle_case(config_path=DEFAULT_CONFIG, output_dir=None, cycles=3, progress=True):
    """Run repeated charge-idle-discharge-idle cycles and save all paper data."""
    cfg = _load_case_config(config_path)
    folder = Path(output_dir) if output_dir else DEFAULT_RESULTS / "three_cycles"
    folder.mkdir(parents=True, exist_ok=True)
    built = _build_repeated_cycles(cfg, count=cycles, save_root=folder, progress=progress)
    _write_turbine_rating_decision(
        cfg, built["cycles"][-1]["stages"][0].yend[:8],
        f"cycle {cycles} charge endpoint", DEFAULT_RESULTS,
    )
    df, mf, summary = export_results(
        built["base_model"], built["stages"], built["stages"][0], built["logs"], folder / "combined"
    )
    table = pd.DataFrame(built["records"])
    table.to_csv(folder / "cycle_summary.csv", index=False, encoding="utf-8-sig")
    _plot_three_cycle_results(df, built["stages"], table, cfg, folder)
    write_json(folder / "application_summary.json", {
        "case": "margin_limited_three_continuous_cycles", "cycles": cycles,
        "turbine_rating_MW": cfg["turbine"]["power_max"]/1e6,
        "water_thermal_storage_active": False,
        "initial_state_correction": {
            "method": "fixed-flow discharge from configured state to normalized margin floor",
            "duration_h": (
                built["precondition_stage"].end-built["precondition_stage"].start
            )/3600,
            "transferred_mass_t": abs(
                built["precondition_stage"].yend[0]-built["precondition_stage"].y0[0]
            )/1e3,
            "event": built["precondition_stage"].event,
        },
        "endpoint_rule": {
            "requested_normalized_margin": cfg["operation"].get("endpoint_margin_floor", 0.02),
            "idle_check_duration_h": cfg["operation"]["idle_duration"]/3600,
            "fallback": "increase active-stage margin floor and rerun if the following idle fails",
        },
        "terminology": {
            "efficiency": "single-cycle electricity efficiency; thermal state is not strictly periodic",
            "margin": "fractional headroom for modeled and evaluated constraints; not certified geomechanical safety",
        },
        "combined_summary": summary, "per_cycle": built["records"],
    })
    if progress:
        print(f"Saved three-cycle application to {folder}")
    return table, summary


def _truncate_stage_at_mass(stage: Stage, target_m_H: float):
    if stage.yend[0] > target_m_H:
        return stage
    for a, b, dense in stage.segments:
        ya, yb = dense(a), dense(b)
        if (ya[0]-target_m_H)*(yb[0]-target_m_H) <= 0:
            root = brentq(lambda t: dense(t)[0]-target_m_H, a, b, xtol=1e-6)
            yend = dense(root)
            segments = [(u, min(v, root), d) for u, v, d in stage.segments if u < root]
            event = {"mode": "discharge", "time": float(root), "kind": "matched_working_mass_target",
                     "reason": "Returned the working CO2 mass prepared by the representative charge", "constraints": []}
            return Stage(stage.mode, stage.q, stage.start, float(root), stage.y0, yend, segments, event)
    return stage


def _sample_stage(model, stage: Stage, interval=60.0):
    times = np.unique(np.r_[np.arange(stage.start, stage.end, interval), stage.end])
    rows = []
    for t in times:
        y = stage.at(float(t))
        q = stage.flow(y[:8])
        states, wp, dx, de, diag = model.evaluate(stage.mode, y[:8], q)
        constraints = evaluate_constraints(model, stage.mode, y[:8], states, wp, diag)
        margins = margin_summary(constraints)
        rows.append({
            "time_s": float(t-stage.start), "time_h": float((t-stage.start)/3600),
            "q_kg_s": q, "P_net_MW": wp["P_net"]/1e6, "P_gross_MW": wp["P_gross"]/1e6,
            "ratio": wp["ratio"], "p_H_MPa": states["H"].p/1e6, "p_L_MPa": states["L"].p/1e6,
            "T_H_C": states["H"].T-273.15, "T_L_C": states["L"].T-273.15,
            "point6_p_MPa": wp["points"][6].p/1e6, "point6_T_C": wp["points"][6].T-273.15,
            "margin_pressure": margins["cavern_pressure"][0],
            "margin_phase": margins["phase"][0], "margin_equipment": margins["equipment"][0],
            "limiting_pressure": margins["cavern_pressure"][1],
            "limiting_phase": margins["phase"][1], "limiting_equipment": margins["equipment"][1],
        })
    return pd.DataFrame(rows)


def _stop_label(stage):
    if stage.event["kind"] == "matched_working_mass_target":
        return "工作气质量用尽"
    if stage.event.get("constraints"):
        return "+".join(stage.event["constraints"])
    if stage.event["kind"] == "no_physical_solution":
        return "恒功率不可维持"
    return stage.event["kind"]


def _plot_envelope(summary, traces, folder, annotated):
    _plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    ordered = summary.sort_values("target_power_MW")
    power = ordered.target_power_MW.to_numpy()
    duration = ordered.duration_h.to_numpy()
    axes[0].fill_between(power, 0, duration, color="#DCEAF7", alpha=.75)
    axes[0].plot(power, duration, color="#1767A4", lw=1.8, marker="o", ms=3.5, label="恒净功率包络")
    ymax = max(duration.max()*1.08, .1)
    pp = np.linspace(power.min(), power.max(), 400)
    max_energy = np.nanmax(ordered.net_energy_MWh)
    energy_levels = np.linspace(max_energy/4, max_energy, 4)
    for energy in energy_levels:
        tt = energy/pp
        mask = tt <= ymax
        axes[0].plot(pp[mask], tt[mask], color="#888888", ls=":", lw=.75)
        if mask.any():
            k = np.flatnonzero(mask)[-1]
            axes[0].text(pp[k], tt[k], f" {energy:.1f} MWh", fontsize=6.5,
                         color="#666666", ha="right", va="bottom")
    if annotated:
        palette = {
            label: color for label, color in zip(
                ordered.stop_reason.unique(), ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"]
            )
        }
        for label, group in ordered.groupby("stop_reason"):
            axes[0].scatter(group.target_power_MW, group.duration_h, s=26,
                            color=palette[label], label=label, zorder=5)
        axes[0].legend(fontsize=6.8, loc="best")
    else:
        axes[0].legend(fontsize=7.2, loc="best")
    axes[0].set_xlim(power.min()-.15, power.max()+.15)
    axes[0].set_ylim(0, ymax)
    axes[0].set_xlabel("目标净功率 / MW")
    axes[0].set_ylabel("恒功率持续时间 / h")
    axes[0].grid(alpha=.2)

    preferred = (2.0, 4.0, 6.0, 9.0)
    selected = []
    for target in preferred:
        idx = int(np.argmin(abs(power-target)))
        if idx not in selected:
            selected.append(idx)
    if len(selected) < min(4, len(ordered)):
        for idx in np.linspace(0, len(ordered)-1, min(4, len(ordered))).round().astype(int):
            if int(idx) not in selected:
                selected.append(int(idx))
    for idx in selected[:4]:
        row = ordered.iloc[int(idx)]
        trace = traces[float(row.target_power_MW)]
        x = 100*trace.time_h/max(trace.time_h.iloc[-1], 1e-12)
        axes[1].plot(x, trace.q_kg_s, lw=1.3, label=f"{row.target_power_MW:.2f} MW")
    axes[1].axhline(50, color="#777777", ls=":", lw=.8)
    axes[1].axhline(500, color="#777777", ls=":", lw=.8)
    axes[1].set_xlabel("各恒功率过程的归一化时间 / %")
    axes[1].set_ylabel("CO2流量 / kg s$^{-1}$")
    axes[1].grid(alpha=.2)
    axes[1].legend(ncol=4, fontsize=6.8, loc="best")
    title = "恒净功率—持续时间包络（标注终止原因）" if annotated else "恒净功率—持续时间包络"
    fig.suptitle(title, fontsize=11)
    _save_figure(fig, folder, "fig_power_duration_annotated" if annotated else "fig_power_duration_clean")


def run_constant_power_envelope(config_path=DEFAULT_CONFIG, output_dir=None, target_powers_MW=None,
                                source_snapshot=None, discharge_reserve_fraction=0.98,
                                integration_max_step_s=300.0, output_interval_s=300.0,
                                progress=True):
    """Scan one complete constant-power discharge from a saved charge endpoint.

    The default source is the charge endpoint already produced by cycle 3 of
    :func:`run_three_cycle_case`; the three preconditioning cycles are therefore
    never repeated here.  Only ``discharge_reserve_fraction`` of that cycle's
    charged working mass is released, leaving the requested 2% discharge reserve.
    """
    cfg = _load_case_config(config_path)
    folder = Path(output_dir) if output_dir else DEFAULT_RESULTS / "power_duration"
    folder.mkdir(parents=True, exist_ok=True)
    if not 0 < discharge_reserve_fraction <= 1:
        raise ValueError("discharge_reserve_fraction must lie in (0, 1]")
    snapshot = Path(source_snapshot) if source_snapshot else (
        DEFAULT_RESULTS / "three_cycles" / "per_cycle" / "cycle_03" / "charge_end.json"
    )
    initial_snapshot = snapshot.with_name("initial_state.json")
    if not snapshot.exists() or not initial_snapshot.exists():
        raise FileNotFoundError(
            "The saved charge endpoint is unavailable. Run run_three_cycle_case() first, "
            f"or pass source_snapshot explicitly. Expected: {snapshot}"
        )
    model, x0, _, _ = Model.load_snapshot(snapshot)
    _, x_before_charge, _, _ = Model.load_snapshot(initial_snapshot)
    if not math.isclose(model.cfg["turbine"]["power_max"], cfg["turbine"]["power_max"]):
        raise ValueError("Snapshot turbine rating differs from the synchronized case configuration")
    if model.cfg["water"].get("enabled", True):
        raise ValueError("Envelope snapshot must use the no-water-thermal-storage base case")
    model.cfg["numerics"]["max_step"] = float(integration_max_step_s)
    charged_working_mass = float(x0[0]-x_before_charge[0])
    if charged_working_mass <= 0:
        raise ValueError("Source snapshot is not a charge endpoint: high-pressure inventory did not increase")
    discharge_mass = discharge_reserve_fraction*charged_working_mass
    model.save_snapshot(folder / "representative_charge_end.json", x0)

    qmin, qmax = cfg["operation"]["q_min"], cfg["operation"]["q_max"]
    pmin = model.net_power(x0, qmin)/1e6
    pmax = min(model.net_power(x0, qmax), cfg["turbine"]["power_max"])/1e6
    if target_powers_MW is None:
        target_powers_MW = np.arange(2.0, 10.0, 1.0)
    targets = sorted({float(p) for p in target_powers_MW if p > 0})
    outside = [p for p in targets if not pmin < p < pmax]
    if outside:
        raise ValueError(
            f"Requested target powers {outside} MW lie outside the instantaneous feasible "
            f"range ({pmin:.6g}, {pmax:.6g}) MW at the saved charge endpoint"
        )
    rows, traces = [], {}
    runs_folder = folder / "runs"
    if runs_folder.exists():
        resolved_runs = runs_folder.resolve()
        if resolved_runs.parent != folder.resolve():
            raise RuntimeError(f"Refusing to clean unexpected runs directory: {resolved_runs}")
        shutil.rmtree(resolved_runs)
    runs_folder.mkdir(parents=True, exist_ok=True)

    for index, target_MW in enumerate(targets, 1):
        if progress:
            print(f"Envelope {index}/{len(targets)}: {target_MW:.4f} MW")
        target = target_MW*1e6
        controller = lambda x, target=target: solve_flow_for_power(model, x, target)
        stage = run_stage(
            model, "discharge", x0, controller, cfg["operation"]["active_duration_max"],
            start=0.0, energy0=np.zeros(len(ENERGY_NAMES)), progress=progress,
            terminal_state_event=lambda y, target_m=x0[0]-discharge_mass: y[0]-target_m,
            terminal_kind="matched_working_mass_target",
            terminal_reason=(
                f"Released {100*discharge_reserve_fraction:g}% of the charged working CO2 mass; "
                f"{100*(1-discharge_reserve_fraction):g}% reserve retained"
            ),
        )
        trace = _sample_stage(model, stage, float(output_interval_s))
        traces[target_MW] = trace
        run_folder = runs_folder / f"P_{target_MW:.4f}MW".replace(".", "p")
        run_folder.mkdir(parents=True, exist_ok=True)
        trace.to_csv(run_folder / "trajectory.csv", index=False, encoding="utf-8-sig")
        write_json(run_folder / "event.json", stage.event)
        net_energy = (stage.yend[8+ENERGY_NAMES.index("E_net")]-stage.y0[8+ENERGY_NAMES.index("E_net")])/3.6e9
        rows.append({
            "target_power_MW": target_MW, "duration_h": (stage.end-stage.start)/3600,
            "net_energy_MWh": net_energy, "transferred_mass_t": (x0[0]-stage.yend[0])/1e3,
            "q_start_kg_s": trace.q_kg_s.iloc[0], "q_end_kg_s": trace.q_kg_s.iloc[-1],
            "ratio_min": trace.ratio.min(), "ratio_max": trace.ratio.max(),
            "minimum_phase_margin": trace.margin_phase.min(),
            "power_tracking_max_error_kW": abs(trace.P_net_MW-target_MW).max()*1e3,
            "stop_reason": _stop_label(stage), "event_kind": stage.event["kind"],
        })
    summary = pd.DataFrame(rows).sort_values("target_power_MW")
    summary.to_csv(folder / "power_duration_envelope.csv", index=False, encoding="utf-8-sig")
    write_json(folder / "scan_metadata.json", {
        "representative_state": "saved cycle-3 charge endpoint; no preconditioning rerun",
        "source_snapshot": str(snapshot.resolve()),
        "charged_working_mass_kg": charged_working_mass,
        "discharged_mass_limit_kg": discharge_mass,
        "discharge_reserve_fraction": discharge_reserve_fraction,
        "remaining_fraction_of_charged_working_mass": 1-discharge_reserve_fraction,
        "integration_max_step_s": integration_max_step_s,
        "output_interval_s": output_interval_s,
        "relative_tolerance": model.cfg["numerics"]["rtol"],
        "q_bounds_kg_s": [qmin, qmax],
        "instantaneous_power_range_MW": [pmin, pmax],
        "turbine_rating_MW": cfg["turbine"]["power_max"]/1e6,
        "target_powers_MW": targets,
    })
    _plot_envelope(summary, traces, folder, annotated=False)
    _plot_envelope(summary, traces, folder, annotated=True)
    if progress:
        print(f"Saved constant-power envelope to {folder}")
    return summary


def _plot_ablation(variant_data, table, cfg, folder):
    _plot_style()
    colors = {"完整井筒": "#1767A4", "无摩擦绝热井筒": "#D95F02", "无井筒": "#1B9E77"}
    styles = {"完整井筒": "-", "无摩擦绝热井筒": "--", "无井筒": "-."}
    widths = {"完整井筒": 2.2, "无摩擦绝热井筒": 1.7, "无井筒": 1.8}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True, layout="constrained")
    for label, data in variant_data.items():
        df = data["df"]
        x = 100*df.time_h/df.time_h.max()
        axes[0].plot(x, df.p_H/1e6, lw=widths[label], ls=styles[label],
                     color=colors[label], label=label)
        axes[1].plot(x, df.p_L/1e6, lw=widths[label], ls=styles[label],
                     color=colors[label], label=label)
    axes[0].axhline(cfg["caverns"]["H"]["p_min"]/1e6, color="#777777", ls=":", lw=.8)
    axes[0].axhline(cfg["caverns"]["H"]["p_max"]/1e6, color="#777777", ls=":", lw=.8)
    axes[1].axhline(cfg["caverns"]["L"]["p_min"]/1e6, color="#777777", ls=":", lw=.8)
    axes[1].axhline(cfg["caverns"]["L"]["p_max"]/1e6, color="#777777", ls=":", lw=.8)
    axes[0].set_ylabel("高压穴压力 / MPa")
    axes[1].set_ylabel("低压穴压力 / MPa")
    axes[1].set_xlabel("循环过程归一化进度 / %")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7.2)
    fig.suptitle("井筒模型消融下的盐穴压力变化", fontsize=11)
    _save_figure(fig, folder, "fig_well_ablation_pressure")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), layout="constrained")
    x = np.arange(len(table))
    width = .34
    axes[0].bar(x-width/2, table.total_charge_MWh, width, label="充电输入")
    axes[0].bar(x+width/2, table.total_discharge_MWh, width, label="净放电输出")
    axes[0].set_xticks(x, table.variant, rotation=12)
    axes[0].set_ylabel("累计电量 / MWh")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(x, 100*table.aggregate_efficiency, color=[colors[v] for v in table.variant])
    axes[1].set_xticks(x, table.variant, rotation=12)
    axes[1].set_ylabel("累计电量比 / %")
    axes[1].grid(axis="y", alpha=.2)
    _save_figure(fig, folder, "fig_well_ablation_metrics")


def run_well_ablation(config_path=DEFAULT_CONFIG, output_dir=None, cycles=1, progress=True):
    """Compare full, frictionless-adiabatic, and absent-well variants."""
    base_cfg = _load_case_config(config_path)
    folder = Path(output_dir) if output_dir else DEFAULT_RESULTS / "well_ablation"
    folder.mkdir(parents=True, exist_ok=True)
    variants = {
        "完整井筒": {},
        "无摩擦绝热井筒": {"friction": False, "U": 0.0},
        "无井筒": {"enabled": False},
    }
    rows, data = [], {}
    for index, (label, changes) in enumerate(variants.items(), 1):
        if progress:
            print(f"Well ablation {index}/{len(variants)}: {label}")
        cfg = copy.deepcopy(base_cfg)
        cfg["well"].update(changes)
        cfg["numerics"]["output_interval"] = 300.0
        variant_folder = folder / f"variant_{index:02d}"
        built = _build_repeated_cycles(cfg, count=cycles, save_root=None, progress=False)
        df, mf, summary = export_results(
            built["base_model"], built["stages"], built["stages"][0], built["logs"], variant_folder / "combined"
        )
        records = pd.DataFrame(built["records"])
        records.to_csv(variant_folder / "cycle_summary.csv", index=False, encoding="utf-8-sig")
        E_in = records.charge_input_MWh.sum()
        E_out = records.net_discharge_MWh.sum()
        well_enabled = cfg["well"].get("enabled", True)
        rows.append({
            "variant": label, "well_enabled": well_enabled,
            "friction": bool(well_enabled and cfg["well"].get("friction", False)),
            "well_U_W_m2K": cfg["well"].get("U", 0.0) if well_enabled else 0.0,
            "total_charge_MWh": E_in, "total_discharge_MWh": E_out,
            "aggregate_efficiency": E_out/E_in if E_in > 0 else np.nan,
            "mean_working_mass_t": records.working_mass_t.mean(),
            "mean_charge_h": records.charge_h.mean(), "mean_discharge_h": records.discharge_h.mean(),
            "p_H_min_MPa": df.p_H.min()/1e6, "p_H_max_MPa": df.p_H.max()/1e6,
            "p_L_min_MPa": df.p_L.min()/1e6, "p_L_max_MPa": df.p_L.max()/1e6,
            "minimum_phase_margin": df.margin_phase.min(),
        })
        data[label] = {"df": df, "summary": summary, "records": records}
    table = pd.DataFrame(rows)
    table.to_csv(folder / "well_ablation_summary.csv", index=False, encoding="utf-8-sig")
    _plot_ablation(data, table, base_cfg, folder)
    write_json(folder / "ablation_metadata.json", {
        "cycles": cycles, "common_turbine_rating_MW": base_cfg["turbine"]["power_max"]/1e6,
        "common_water_thermal_storage_active": False,
        "variants": list(variants),
    })
    if progress:
        print(f"Saved well ablation to {folder}")
    return table
