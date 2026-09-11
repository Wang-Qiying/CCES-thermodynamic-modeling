"""g>=0, fixed scales; numerical residuals and model warnings are separate."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Constraint:
    name: str
    group: str
    location: str
    value: float
    limit: float
    physical_residual: float
    scale: float
    normalized_margin: float
    active: bool
    evaluated: bool
    violated: bool
    reason: str

    def row(self):
        return asdict(self)


def evaluate_constraints(model, mode, x, states, wp, diagnostic):
    c, rows = model.cfg, []
    s = dict(c["scales"])
    s["water_mass"] = s["water_mass_fraction"]*c["water"]["mass_total"]

    def add(name, group, loc, value, limit, scale, lower=True, active=True, reason="", normalizer=None):
        evaluated = active and limit is not None and value is not None and math.isfinite(value)
        val = float(value) if active and value is not None else math.nan
        lim = float(limit) if limit is not None else math.nan
        g = (val-lim if lower else lim-val) if evaluated else math.nan
        denom = float(s[scale] if normalizer is None else normalizer)
        if not math.isfinite(denom) or denom <= 0:
            raise ValueError(f"Constraint {name} has invalid normalization denominator {denom}")
        rows.append(Constraint(name, group, loc, val, lim, g, denom, g/denom,
                               active, evaluated, bool(evaluated and g < 0),
                               reason if reason else ("" if evaluated else "not applicable in this stage" if not active else "threshold unavailable; not evaluated")))

    for i in ("H", "L"):
        cc, st = c["caverns"][i], states[i]
        pressure_span = cc["p_max"]-cc["p_min"]
        add(f"{i}.p_min", "cavern_pressure", i, st.p, cc["p_min"], "pressure",
            normalizer=pressure_span)
        add(f"{i}.p_max", "cavern_pressure", i, st.p, cc["p_max"], "pressure", False,
            normalizer=pressure_span)
        add(f"{i}.T_max", "thermal", i, st.T, cc["T_max"], "temperature", False)
        add(f"{i}.T_min", "thermal", i, st.T, cc["T_min"], "temperature")
        add(f"{i}.dpdt", "engineering", i, abs(diagnostic[f"dp_{i}"]), c["engineering"]["pressure_rate_max"], "pressure_rate", False)
        add(f"{i}.dTdt", "engineering", i, abs(diagnostic[f"dT_{i}"]), c["engineering"]["temperature_rate_max"], "temperature_rate", False)

    phase_states = {"H": [(states["H"].p,states["H"].T)], "L": [(states["L"].p,states["L"].T)]}
    for j in range(1,7):
        st = wp["points"].get(j)
        phase_states[f"point{j}"] = [] if st is None else [(st.p,st.T)]
    for name in ("take", "inject"):
        well = wp[name]
        phase_states[name+"_well"] = [] if well is None else [(r["p"],r["T"]) for r in well.profile]
        pmax = max((r["p"] for r in well.profile), default=math.nan) if well else math.nan
        add(name+"_well.rating", "engineering", name+"_well", pmax, c["well"]["pressure_rating"], "pressure", False, mode!="idle")
    phase_states["hx_path"] = [] if wp["hx"] is None else [(r["p"],r["T_CO2"]) for r in wp["hx"].profile]
    for name, locations in phase_states.items():
        active = c["phase"]["enabled"] and bool(locations)
        p_sc_min = model.fluid.pcrit+c["phase"]["pressure_above_critical"]
        T_sc_min = model.fluid.Tcrit+c["phase"]["temperature_above_critical"]
        add(f"phase.{name}.p", "phase", name, min((p for p,T in locations), default=math.nan),
            p_sc_min, "pressure", active=active, normalizer=p_sc_min)
        add(f"phase.{name}.T", "phase", name, min((T for p,T in locations), default=math.nan),
            T_sc_min, "temperature", active=active, normalizer=T_sc_min)

    for stage, machine in [("charge","compressor"),("discharge","turbine")]:
        active = mode == stage
        mc = c[machine]
        flow_span = c["operation"]["q_max"]-c["operation"]["q_min"]
        ratio_span = mc["ratio_max"]-1.0
        add(f"{machine}.q_min", "equipment", machine, wp["q"], c["operation"]["q_min"], "flow",
            active=active, normalizer=flow_span)
        add(f"{machine}.q_max", "equipment", machine, wp["q"], c["operation"]["q_max"], "flow", False,
            active, normalizer=flow_span)
        add(f"{machine}.ratio_min", "equipment", machine, wp["ratio"], 1.0, "ratio",
            active=active, normalizer=ratio_span)
        add(f"{machine}.ratio_max", "equipment", machine, wp["ratio"], mc["ratio_max"], "ratio", False,
            active, normalizer=ratio_span)
        add(f"{machine}.power_max", "equipment", machine,
            wp["P_comp"] if stage=="charge" else wp["P_gross"], mc["power_max"], "power", False,
            active, normalizer=mc["power_max"])
        st = wp["points"].get(2 if stage=="charge" else 6)
        add(f"{machine}.T_out_max", "equipment", machine, None if st is None else st.T,
            mc["T_out_max"], "temperature", False, active,
            normalizer=mc["T_out_max"] if mc["T_out_max"] is not None else None)
    add("net_power.positive", "equipment", "grid", wp["P_net"], 0.0, "power",
        active=mode=="discharge" and c["operation"]["require_positive_net_power"],
        normalizer=c["turbine"]["power_max"])
    water_active = c["water"].get("enabled", True)
    for name, mass, T in [("hot",x[5],x[6]),("cold",c["water"]["mass_total"]-x[5],x[7])]:
        add(f"water.{name}.mass_min", "water_hx", name, mass, c["water"]["min_fraction"]*c["water"]["mass_total"], "water_mass", active=water_active)
        add(f"water.{name}.capacity", "water_hx", name, mass, c["water"][name+"_capacity"], "water_mass", False, water_active)
        add(f"water.{name}.T_max", "water_hx", name, T, c["water"]["T_max"], "temperature", False, water_active)
        add(f"water.{name}.T_min", "engineering", name, T, c["water"]["T_min"], "temperature", active=water_active)
    add("pump.NPSH", "engineering", "pump", None, c["water"]["npsh_required"], "length", active=water_active and mode!="idle",
        reason="NPSHa needs suction geometry and losses; no NPSH assessment implemented")
    for stage in ("charge", "discharge"):
        hx = wp["hx"] if mode == stage else None
        add(f"hx.{stage}.approach", "water_hx", stage+"_hx", None if hx is None else hx.min_approach,
            c["exchangers"]["min_approach"], "approach",
            active=water_active and mode==stage and c["exchangers"][stage]["UA"]>0,
            reason="not applicable for configured zero-UA exchanger" if mode==stage and c["exchangers"][stage]["UA"]==0 else "")
    return rows


def margin_summary(rows):
    eligible = [r for r in rows if r.active and r.evaluated]
    result = {}
    for group in sorted({r.group for r in rows}):
        candidates = [r for r in eligible if r.group == group]
        low = min(candidates, key=lambda r:r.normalized_margin) if candidates else None
        result[group] = (low.normalized_margin,low.name) if low else (math.nan,"")
    low = min(eligible, key=lambda r:r.normalized_margin) if eligible else None
    result["all_evaluated_applicable"] = (low.normalized_margin,low.name) if low else (math.nan,"")
    return result
