"""Build publication-ready case-study figures and their source tables.

The script only post-processes the existing application-analysis results.  It
does not rerun the thermodynamic simulations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE.parent
RESULTS = STUDY_DIR / "results"
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

BLUE = "#2F6F9F"
ORANGE = "#D97732"
GREEN = "#27896F"
PURPLE = "#7562A8"
RED = "#B44E55"
DARK = "#333333"
GRID = "#D9D9D9"
CHARGE_SHADE = "#DCEAF5"
DISCHARGE_SHADE = "#F7E5D7"
IDLE_SHADE = "#EEEEEE"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#BBBBBB",
            "legend.fancybox": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = 0.015,
    y: float = 0.975,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.7},
        zorder=20,
    )


def mode_spans(traj: pd.DataFrame) -> list[tuple[float, float, str]]:
    mode = traj["mode"].astype(str).to_numpy()
    time_h = traj["time_h"].to_numpy(float)
    starts = np.r_[0, np.flatnonzero(mode[1:] != mode[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(traj) - 1]
    spans: list[tuple[float, float, str]] = []
    for s, e in zip(starts, ends):
        spans.append((float(time_h[s]), float(time_h[e]), mode[s]))
    return spans


def shade_modes(ax: plt.Axes, spans: list[tuple[float, float, str]]) -> None:
    colors = {"charge": CHARGE_SHADE, "discharge": DISCHARGE_SHADE, "idle": IDLE_SHADE}
    alphas = {"charge": 0.36, "discharge": 0.34, "idle": 0.46}
    for t0, t1, mode in spans:
        ax.axvspan(t0, t1, color=colors[mode], alpha=alphas[mode], lw=0, zorder=-20)


def mark_mode_boundaries(ax: plt.Axes, spans: list[tuple[float, float, str]]) -> None:
    """Draw unobtrusive separators at charge/idle/discharge transitions."""
    for t0, _, _ in spans[1:]:
        ax.axvline(t0, color="#888888", lw=0.42, alpha=0.55, zorder=1)


def constraint_array(constraints: pd.DataFrame, name: str) -> np.ndarray:
    values = constraints.loc[constraints["name"].eq(name), "normalized_margin"].to_numpy(float)
    if len(values) == 0:
        raise KeyError(f"Constraint not found: {name}")
    return values


def active_array(constraints: pd.DataFrame, name: str) -> np.ndarray:
    values = constraints.loc[constraints["name"].eq(name), "active"]
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.lower().eq("true").to_numpy()


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600)
    fig.savefig(FIG_DIR / f"{stem}.svg")
    plt.close(fig)


def build_three_cycle_figure() -> dict[str, float]:
    traj = pd.read_csv(RESULTS / "three_cycles" / "combined" / "trajectory.csv")
    constraints = pd.read_csv(RESULTS / "three_cycles" / "combined" / "constraints.csv")
    if len(traj) != len(constraints.loc[constraints["name"].eq("H.p_min")]):
        raise ValueError("Trajectory and constraint histories are not row-aligned.")

    time_h = traj["time_h"].to_numpy(float)
    spans = mode_spans(traj)

    pressure = pd.DataFrame(
        {
            "time_h": time_h,
            "p_H_MPa": traj["p_H"].to_numpy(float) / 1e6,
            "p_L_MPa": traj["p_L"].to_numpy(float) / 1e6,
            "p_H_min_MPa": 16.50,
            "p_H_max_MPa": 44.00,
            "p_L_min_MPa": 7.92,
            "p_L_max_MPa": 21.12,
            "mode": traj["mode"].astype(str),
        }
    )
    temperature = pd.DataFrame(
        {
            "time_h": time_h,
            "T_CO2_H_degC": traj["T_H"].to_numpy(float) - 273.15,
            "T_rock_H_degC": traj["T_r_H"].to_numpy(float) - 273.15,
            "T_CO2_L_degC": traj["T_L"].to_numpy(float) - 273.15,
            "T_rock_L_degC": traj["T_r_L"].to_numpy(float) - 273.15,
            "mode": traj["mode"].astype(str),
        }
    )

    mu_p_H = np.minimum(
        constraint_array(constraints, "H.p_min"),
        constraint_array(constraints, "H.p_max"),
    )
    mu_p_L = np.minimum(
        constraint_array(constraints, "L.p_min"),
        constraint_array(constraints, "L.p_max"),
    )
    mu_p = np.minimum(mu_p_H, mu_p_L)
    mu_ratio_charge = np.minimum(
        constraint_array(constraints, "compressor.ratio_min"),
        constraint_array(constraints, "compressor.ratio_max"),
    )
    mu_ratio_discharge = np.minimum(
        constraint_array(constraints, "turbine.ratio_min"),
        constraint_array(constraints, "turbine.ratio_max"),
    )
    mu_all = traj["margin_all_evaluated_applicable"].to_numpy(float)

    phase_margin: dict[str, np.ndarray] = {"time_h": time_h}
    for point in range(1, 7):
        p_name = f"phase.point{point}.p"
        mu = np.minimum(
            constraint_array(constraints, p_name),
            constraint_array(constraints, f"phase.point{point}.T"),
        )
        mu[~active_array(constraints, p_name)] = np.nan
        phase_margin[f"mu_sc_point_{point}"] = mu
    phase_margin["mode"] = traj["mode"].astype(str).to_numpy()
    phase_margin_df = pd.DataFrame(phase_margin)

    def normalized_stage_table(mode: str, points: tuple[int, int]) -> pd.DataFrame:
        mode_values = traj["mode"].astype(str).to_numpy()
        transitions = np.flatnonzero(mode_values[1:] != mode_values[:-1]) + 1
        starts = np.r_[0, transitions]
        ends = np.r_[transitions, len(traj)]
        blocks = [(s, e) for s, e in zip(starts, ends) if mode_values[s] == mode]
        if len(blocks) != 3:
            raise ValueError(f"Expected three {mode} stages, found {len(blocks)}")

        progress_grid = np.linspace(0.0, 100.0, 201)
        table: dict[str, np.ndarray] = {f"normalized_{mode}_progress_pct": progress_grid}
        start, end = blocks[-1]
        idx = np.arange(start, end)
        local_time = time_h[idx]
        progress = 100.0 * (local_time - local_time[0]) / (local_time[-1] - local_time[0])
        progress, unique_idx = np.unique(progress, return_index=True)
        idx = idx[unique_idx]
        ratio_margin = mu_ratio_charge if mode == "charge" else mu_ratio_discharge
        table["mu_all"] = np.interp(progress_grid, progress, mu_all[idx])
        table["mu_cavern_pressure"] = np.interp(progress_grid, progress, mu_p[idx])
        table["mu_pressure_ratio"] = np.interp(progress_grid, progress, ratio_margin[idx])
        for point in points:
            table[f"mu_sc_point_{point}"] = np.interp(
                progress_grid,
                progress,
                phase_margin_df[f"mu_sc_point_{point}"].to_numpy(float)[idx],
            )
        return pd.DataFrame(table)

    charge_margin = normalized_stage_table("charge", (1, 2))
    discharge_margin = normalized_stage_table("discharge", (4, 6))

    pressure.to_csv(DATA_DIR / "fig1a_cavern_pressure.csv", index=False)
    temperature.to_csv(DATA_DIR / "fig1b_cavern_temperature.csv", index=False)
    charge_margin.to_csv(DATA_DIR / "fig1c_charge_margin.csv", index=False)
    discharge_margin.to_csv(DATA_DIR / "fig1d_discharge_margin.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(3.50, 3.12), sharex=False)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    for ax in (ax_a, ax_b):
        shade_modes(ax, spans)
        mark_mode_boundaries(ax, spans)
        ax.grid(False)
        ax.set_xlim(time_h.min(), time_h.max())
    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=7.8)

    ax_a.plot(time_h, pressure["p_H_MPa"], color=BLUE, label=r"$H$")
    ax_a.plot(time_h, pressure["p_L_MPa"], color=ORANGE, label=r"$L$")
    ax_a.axhline(16.50, color=BLUE, ls=":", lw=0.7, alpha=0.8)
    ax_a.axhline(44.00, color=BLUE, ls="-.", lw=0.7, alpha=0.8)
    ax_a.axhline(7.92, color=ORANGE, ls=":", lw=0.7, alpha=0.8)
    ax_a.axhline(21.12, color=ORANGE, ls="-.", lw=0.7, alpha=0.8)
    ax_a.set_ylabel("Pressure (MPa)")
    ax_a.set_xlabel("Time (h)", labelpad=0.5)
    ax_a.set_ylim(6.8, 45.2)
    ax_a.legend(loc="center left", ncol=1, handlelength=1.7, borderpad=0.25, labelspacing=0.2)
    add_panel_label(ax_a, "(a)")

    ax_b.plot(time_h, temperature["T_CO2_H_degC"], color=BLUE, label=r"CO$_2$, $H$")
    ax_b.plot(time_h, temperature["T_rock_H_degC"], color=BLUE, ls="--", alpha=0.72, label=r"Rock, $H$")
    ax_b.plot(time_h, temperature["T_CO2_L_degC"], color=ORANGE, label=r"CO$_2$, $L$")
    ax_b.plot(time_h, temperature["T_rock_L_degC"], color=ORANGE, ls="--", alpha=0.72, label=r"Rock, $L$")
    ax_b.set_ylabel(r"Temperature ($^\circ$C)")
    ax_b.set_xlabel("Time (h)", labelpad=0.5)
    ax_b.set_ylim(47.5, 104.5)
    ax_b.legend(loc="center right", ncol=2, columnspacing=0.55, handlelength=1.6, borderpad=0.25, labelspacing=0.18)
    add_panel_label(ax_b, "(b)")

    x_charge = charge_margin["normalized_charge_progress_pct"]
    ax_c.plot(x_charge, charge_margin["mu_all"], color=DARK, lw=1.4, label=r"$\mu_{\min}$")
    ax_c.plot(x_charge, charge_margin["mu_cavern_pressure"], color="#777777", ls="--", label=r"$\mu_p$")
    ax_c.plot(x_charge, charge_margin["mu_pressure_ratio"], color=PURPLE,
              ls=(0, (4.0, 1.35, 1.0, 1.35)), label=r"$\mu_r$")
    ax_c.plot(x_charge, charge_margin["mu_sc_point_1"], color=BLUE, label=r"$\mu_{\mathrm{sc},1}$")
    ax_c.plot(x_charge, charge_margin["mu_sc_point_2"], color=ORANGE, label=r"$\mu_{\mathrm{sc},2}$")
    ax_c.axhline(0.02, color=RED, lw=0.8, ls=":", zorder=2, label="2% floor")
    ax_c.set_ylabel("Normalized margin")
    ax_c.set_xlabel("Normalized charge progress (%)")
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0.0, 1.05*charge_margin.drop(columns=x_charge.name).max().max())
    ax_c.legend(loc="upper right", bbox_to_anchor=(0.97, 0.975), ncol=2,
                columnspacing=0.32, handlelength=2.15, handletextpad=0.28,
                borderpad=0.16, labelspacing=0.08, fontsize=5.9)
    add_panel_label(ax_c, "(c)")

    x_discharge = discharge_margin["normalized_discharge_progress_pct"]
    ax_d.plot(x_discharge, discharge_margin["mu_all"], color=DARK, lw=1.4, label=r"$\mu_{\min}$")
    ax_d.plot(x_discharge, discharge_margin["mu_cavern_pressure"], color="#777777", ls="--", label=r"$\mu_p$")
    ax_d.plot(x_discharge, discharge_margin["mu_pressure_ratio"], color=PURPLE,
              ls=(0, (4.0, 1.35, 1.0, 1.35)), label=r"$\mu_r$")
    ax_d.plot(x_discharge, discharge_margin["mu_sc_point_4"], color=BLUE, label=r"$\mu_{\mathrm{sc},4}$")
    ax_d.plot(x_discharge, discharge_margin["mu_sc_point_6"], color=ORANGE, label=r"$\mu_{\mathrm{sc},6}$")
    ax_d.axhline(0.02, color=RED, lw=0.8, ls=":", zorder=2, label="2% floor")
    ax_d.set_ylabel("Normalized margin")
    ax_d.set_xlabel("Normalized discharge progress (%)")
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0.0, 1.15*discharge_margin.drop(columns=x_discharge.name).max().max())
    ax_d.legend(loc="upper right", bbox_to_anchor=(0.99, 0.975), ncol=2,
                columnspacing=0.32, handlelength=2.15, handletextpad=0.28,
                borderpad=0.16, labelspacing=0.08, fontsize=5.9)
    add_panel_label(ax_d, "(d)")

    for ax in axes.ravel():
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)

    fig.subplots_adjust(left=0.145, right=0.992, bottom=0.125, top=0.992,
                        wspace=0.58, hspace=0.28)
    save_figure(fig, "fig_case_three_cycles")

    return {
        "min_pressure_margin": float(np.nanmin(np.minimum(mu_p_H, mu_p_L))),
        "min_phase_margin": float(
            np.nanmin(phase_margin_df[[f"mu_sc_point_{i}" for i in range(1, 7)]].to_numpy())
        ),
    }


def build_power_duration_figure() -> dict[str, float]:
    envelope = pd.read_csv(RESULTS / "power_duration" / "power_duration_envelope.csv")
    reason_map = {
        "matched_working_mass_target": "Inventory reserve",
        "no_physical_solution": "Flow limit",
    }
    envelope["termination"] = envelope["event_kind"].map(reason_map).fillna(envelope["event_kind"])
    envelope.to_csv(DATA_DIR / "fig2a_power_duration_envelope.csv", index=False)

    normalized_time = np.linspace(0.0, 100.0, 101)
    mass_flow = pd.DataFrame({"normalized_discharge_time_pct": normalized_time})
    representative_powers = [2, 4, 6, 9]
    for power in representative_powers:
        run_dir = RESULTS / "power_duration" / "runs" / f"P_{power}p0000MW"
        traj = pd.read_csv(run_dir / "trajectory.csv")
        active = traj.loc[traj["q_kg_s"].notna(), ["time_h", "q_kg_s"]].copy()
        active = active.sort_values("time_h").drop_duplicates("time_h", keep="last")
        local_time = active["time_h"].to_numpy(float)
        local_time = 100.0 * (local_time - local_time[0]) / (local_time[-1] - local_time[0])
        mass_flow[f"q_{power}_MW_kg_s"] = np.interp(
            normalized_time, local_time, active["q_kg_s"].to_numpy(float)
        )
    mass_flow.to_csv(DATA_DIR / "fig2b_mass_flow_profiles.csv", index=False)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.50, 1.72))

    ax_a.fill_between(
        envelope["target_power_MW"],
        0.0,
        envelope["duration_h"],
        color=BLUE,
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    ax_a.plot(envelope["target_power_MW"], envelope["duration_h"], color=BLUE, lw=1.25, zorder=2)
    inventory = envelope["event_kind"].eq("matched_working_mass_target")
    flow_limit = envelope["event_kind"].eq("no_physical_solution")
    ax_a.scatter(
        envelope.loc[inventory, "target_power_MW"],
        envelope.loc[inventory, "duration_h"],
        s=17,
        marker="o",
        color=GREEN,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
        label="Inventory reserve",
    )
    ax_a.scatter(
        envelope.loc[flow_limit, "target_power_MW"],
        envelope.loc[flow_limit, "duration_h"],
        s=15,
        marker="s",
        color=ORANGE,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
        label="Flow limit",
    )
    ax_a.set_xlabel("Constant net power (MW)")
    ax_a.set_ylabel("Discharge duration (h)")
    ax_a.set_xticks(np.arange(2, 10, 1))
    ax_a.set_ylim(0, 4.75)
    ax_a.legend(loc="upper right", ncol=1, handlelength=1.2, borderpad=0.24, labelspacing=0.18)
    add_panel_label(ax_a, "(a)", x=0.018, y=0.03, va="bottom")

    profile_colors = [BLUE, GREEN, ORANGE, PURPLE]
    for power, color in zip(representative_powers, profile_colors):
        ax_b.plot(
            mass_flow["normalized_discharge_time_pct"],
            mass_flow[f"q_{power}_MW_kg_s"],
            color=color,
        )
    ax_b.axhline(500.0, color=RED, ls="--", lw=0.85)
    ax_b.set_xlabel("Normalized discharge time (%)")
    ax_b.set_ylabel("Mass flow rate (kg/s)")
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(70, 550)
    label_x = 8.0
    label_offsets = {2: 6.0, 4: 13.0, 6: 14.0, 9: -7.0}
    label_vertical_alignment = {2: "bottom", 4: "bottom", 6: "bottom", 9: "top"}
    for power, color in zip(representative_powers, profile_colors):
        q_label = np.interp(
            label_x,
            mass_flow["normalized_discharge_time_pct"],
            mass_flow[f"q_{power}_MW_kg_s"],
        )
        ax_b.text(
            label_x,
            q_label + label_offsets[power],
            f"{power} MW",
            color=color,
            fontsize=6.2,
            ha="left",
            va=label_vertical_alignment[power],
        )
    ax_b.text(
        97.0,
        512.0,
        "Maximum flow rate",
        color=RED,
        fontsize=6.2,
        ha="right",
        va="bottom",
    )
    add_panel_label(ax_b, "(b)", x=0.985, y=0.03, ha="right", va="bottom")

    for ax in (ax_a, ax_b):
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=7.8)
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)
    fig.subplots_adjust(left=0.145, right=0.992, bottom=0.29, top=0.985, wspace=0.58)
    save_figure(fig, "fig_case_power_duration")

    return {
        "duration_2MW_h": float(envelope.loc[envelope.target_power_MW.eq(2), "duration_h"].iloc[0]),
        "duration_9MW_h": float(envelope.loc[envelope.target_power_MW.eq(9), "duration_h"].iloc[0]),
        "energy_2MW_MWh": float(envelope.loc[envelope.target_power_MW.eq(2), "net_energy_MWh"].iloc[0]),
        "energy_9MW_MWh": float(envelope.loc[envelope.target_power_MW.eq(9), "net_energy_MWh"].iloc[0]),
    }


def build_ablation_table() -> dict[str, float]:
    ablation = pd.read_csv(RESULTS / "well_ablation" / "well_ablation_summary.csv")
    variant_map = {
        "完整井筒": "Full wellbore",
        "无摩擦绝热井筒": "Frictionless adiabatic",
        "无井筒": "No wellbore",
    }
    ablation.insert(0, "model", ablation["variant"].map(variant_map))
    ablation["delta_p_H_MPa"] = ablation["p_H_max_MPa"] - ablation["p_H_min_MPa"]
    ablation["delta_p_L_MPa"] = ablation["p_L_max_MPa"] - ablation["p_L_min_MPa"]
    base_discharge = float(ablation.loc[ablation["model"].eq("Full wellbore"), "total_discharge_MWh"].iloc[0])
    ablation["discharge_energy_change_pct_vs_full"] = 100.0 * (
        ablation["total_discharge_MWh"] / base_discharge - 1.0
    )
    ablation.to_csv(DATA_DIR / "table_well_ablation.csv", index=False)

    frictionless = ablation.loc[ablation["model"].eq("Frictionless adiabatic")].iloc[0]
    no_well = ablation.loc[ablation["model"].eq("No wellbore")].iloc[0]
    full = ablation.loc[ablation["model"].eq("Full wellbore")].iloc[0]
    return {
        "frictionless_discharge_change_pct": float(frictionless["discharge_energy_change_pct_vs_full"]),
        "frictionless_efficiency_change_pct": float(
            100.0 * (
                frictionless["aggregate_efficiency"] / full["aggregate_efficiency"] - 1.0
            )
        ),
        "no_well_discharge_change_pct": float(no_well["discharge_energy_change_pct_vs_full"]),
    }


def build_summary() -> None:
    cycles = pd.read_csv(RESULTS / "three_cycles" / "cycle_summary.csv")
    corrected = pd.read_json(
        RESULTS / "three_cycles" / "corrected_initial_state.json", typ="series"
    )
    m1 = build_three_cycle_figure()
    m2 = build_power_duration_figure()
    m3 = build_ablation_table()
    summary = pd.DataFrame(
        [
            {
                "initial_discharge_h": corrected["duration_h"],
                "initial_discharge_mass_t": corrected["transferred_mass_t"],
                "cycle1_charge_mass_t": cycles.loc[0, "charge_mass_t"],
                "cycle1_discharge_mass_t": cycles.loc[0, "discharge_mass_t"],
                "cycle1_mass_difference_pct": cycles.loc[0, "charge_discharge_mass_difference_pct"],
                "cycle1_working_mass_t": cycles.loc[0, "working_mass_t"],
                "cycle3_charge_mass_t": cycles.loc[2, "charge_mass_t"],
                "cycle3_discharge_mass_t": cycles.loc[2, "discharge_mass_t"],
                "cycle3_mass_difference_pct": cycles.loc[2, "charge_discharge_mass_difference_pct"],
                "cycle3_working_mass_t": cycles.loc[2, "working_mass_t"],
                "cycle1_efficiency_pct": 100.0 * cycles.loc[0, "single_cycle_efficiency"],
                "cycle3_efficiency_pct": 100.0 * cycles.loc[2, "single_cycle_efficiency"],
                **m1,
                **m2,
                **m3,
            }
        ]
    )
    summary.to_csv(DATA_DIR / "paper_reported_metrics.csv", index=False)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    build_summary()
    print(f"Figures written to {FIG_DIR}")
    print(f"Source tables written to {DATA_DIR}")
