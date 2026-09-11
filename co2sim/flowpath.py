import numpy as np
from scipy.optimize import brentq, least_squares, root
from .wells import map_well
from .machines import compress, expand
from .exchangers import exchange, evaluate_heat
from .water import electric_auxiliary
from .errors import ModelError, NoPhysicalSolution, NumericalFailure, PropertyFailure, ModeInapplicable


def bounded_match(evaluate, target, lower, upper, cfg):
    """Ascending log grid; choose the first contiguous valid sign-changing bracket.

    Failed trials never masquerade as roots. Search bounds are numerical, not ratings.
    """
    failures, valid, bracket = [], [], None
    # The injection-bottom pressure is normally monotone in machine outlet
    # pressure. Try the full physical interval first; retain the deterministic
    # grid fallback for non-monotone or invalid endpoint cases.
    endpoints=[]
    for p in (lower,upper):
        try:
            result=evaluate(float(p))
            endpoints.append((float(p),result["inject"].outlet.p-target))
        except ModelError as exc:
            failures.append({"p":float(p),"kind":exc.kind,"reason":str(exc)})
    if len(endpoints)==2 and endpoints[0][1]*endpoints[1][1] <= 0:
        bracket=(lower,upper)
        valid.extend(endpoints)
    previous = None
    for p in ([] if bracket is not None else np.geomspace(lower, upper, cfg["numerics"]["pressure_bracket_points"])):
        try:
            result = evaluate(float(p))
            r = result["inject"].outlet.p-target
        except ModelError as exc:
            failures.append({"p": float(p), "kind": exc.kind, "reason": str(exc)})
            previous = None
            continue
        valid.append((float(p), r))
        if previous is not None and previous[1]*r <= 0:
            bracket = (previous[0], float(p))
            break
        previous = (float(p), r)
    if bracket is None:
        if not valid and failures and all(f["kind"] == "mode_inapplicable" for f in failures):
            raise ModeInapplicable(f"All pressure trials have inapplicable heat direction: {failures[:3]}")
        if any(f["kind"] == "numerical_failure" for f in failures):
            raise NumericalFailure(f"Incomplete pressure bracket due to failed numerical trials: {failures[:3]}")
        if any(f["kind"] == "property_failure" for f in failures):
            raise PropertyFailure(f"Incomplete pressure bracket due to property failures: {failures[:3]}")
        if not valid and failures and all(f["kind"] == "property_failure" for f in failures):
            raise PropertyFailure(f"All pressure trials failed property evaluation: {failures[:2]}")
        if not valid and failures and any(f["kind"] == "numerical_failure" for f in failures):
            raise NumericalFailure(f"No valid pressure trials: {failures[:2]}")
        raise NoPhysicalSolution(f"No pressure-matching bracket in [{lower},{upper}] Pa; valid residuals={valid}; failed trials={failures[:3]}")
    try:
        root = brentq(lambda p: evaluate(p)["inject"].outlet.p-target, *bracket,
                      xtol=cfg["numerics"]["pressure_xtol"], rtol=1e-11)
    except ModelError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise NumericalFailure(f"Pressure root failed: {exc}") from exc
    result = evaluate(root)
    result["pressure_residual"] = result["inject"].outlet.p-target
    result["pressure_bracket"] = bracket
    result["trial_failures"] = failures
    return result


def solve_flow_for_power(model, x, target_power, q_min=None, q_max=None):
    """Return discharge flow that matches a prescribed net electric power.

    This is a quasi-steady algebraic control layer.  It deliberately keeps no
    continuation state so rejected adaptive-integrator trials cannot alter the
    subsequent solution.
    """
    if target_power <= 0:
        raise ValueError("Constant-power discharge requires target_power > 0")
    op = model.cfg["operation"]
    lower = float(op["q_min"] if q_min is None else q_min)
    upper = float(op["q_max"] if q_max is None else q_max)
    if not 0 < lower < upper:
        raise ValueError("Constant-power flow bounds must satisfy 0 < q_min < q_max")

    def residual(q):
        return model.net_power(x, float(q))-target_power

    def grid_fallback():
        # A small deterministic grid recovers a valid contiguous interval when
        # a locally proposed endpoint is outside the property/work-point domain.
        valid = []
        for q in np.linspace(lower, upper, 13):
            try:
                valid.append((float(q), residual(float(q))))
            except ModelError:
                valid.append((float(q), None))
        for (qa, fa), (qb, fb) in zip(valid[:-1], valid[1:]):
            if fa is not None and fb is not None and fa*fb <= 0:
                return float(brentq(residual, qa, qb, xtol=1e-7, rtol=1e-11))
        vals = [(q, f/1e6) for q, f in valid if f is not None]
        raise NoPhysicalSolution(
            f"No constant-power flow bracket for target={target_power/1e6:.6g} MW; "
            f"valid residuals (q, MW)={vals}")

    try:
        flo = residual(lower)
    except ModelError:
        return grid_fallback()
    if abs(flo) <= 1.0:
        return lower
    if flo > 0:
        raise NoPhysicalSolution(
            f"Target net power {target_power/1e6:.6g} MW outside instantaneous "
            f"range: q_min={lower:g} kg/s already produces "
            f"{(flo+target_power)/1e6:.6g} MW")

    # Net turbine power is close to proportional to flow at a fixed cavern
    # state.  Start from that physical scaling and expand only as far as needed;
    # this avoids evaluating q_max for every low-power RK stage.
    p_lower = flo+target_power
    q_trial = float(np.clip(lower*target_power/max(p_lower, 1.0),
                            lower*(1+1e-8), upper))
    qa, fa = lower, flo
    while True:
        try:
            fb = residual(q_trial)
        except ModelError:
            return grid_fallback()
        if abs(fb) <= 1.0:
            return q_trial
        if fb > 0:
            qb = q_trial
            break
        if q_trial >= upper*(1-1e-12):
            raise NoPhysicalSolution(
                f"Target net power {target_power/1e6:.6g} MW outside instantaneous "
                f"range: q_max={upper:g} kg/s produces "
                f"{(fb+target_power)/1e6:.6g} MW")
        qa, fa = q_trial, fb
        p_trial = fb+target_power
        scaled = q_trial*target_power/max(p_trial, 1.0)*1.01
        q_trial = float(min(upper, max(q_trial*1.15, scaled)))
    try:
        return float(brentq(residual, qa, qb, xtol=1e-7, rtol=1e-11))
    except (ValueError, RuntimeError) as exc:
        raise NumericalFailure(f"Constant-power flow root failed: {exc}") from exc


def solve_workpoint(model, mode, x, q, states=None):
    cfg, f = model.cfg, model.fluid
    states = model.states(x) if states is None else states
    if mode == "idle":
        return dict(mode=mode, q=0.0, mw=0.0, points={}, take=None, inject=None, hx=None,
                    fluid_work=0.0, P_comp=0.0, P_gross=0.0, P_pump=0.0,
                    P_other=cfg["auxiliary"]["idle_power"], P_grid=cfg["auxiliary"]["idle_power"],
                    P_net=0.0, pressure_residual=np.nan, ratio=np.nan, trial_failures=[])
    if mode not in ("charge", "discharge") or q <= 0:
        raise ValueError("Active modes require positive prescribed CO2 flow")
    source, destination = ("L", "H") if mode == "charge" else ("H", "L")
    take = map_well(f, states[source], q, cfg["caverns"][source]["depth"], True, cfg)
    upper = cfg["numerics"]["pressure_search_max"]
    lower = cfg["numerics"]["pressure_search_min"]
    if mode == "charge":
        lower = max(lower, take.outlet.p*(1+1e-9))
        target = states[destination].p
        if cfg["exchangers"][mode]["UA"] == 0:
            def zero_ua_trial(p2):
                s2, power_fluid, power = compress(f,take.outlet,p2,q,cfg)
                hx = exchange(f,s2,q,x[7],mode,cfg,model.rho_ref)
                inject = map_well(f,hx.outlet,q,cfg["caverns"][destination]["depth"],False,cfg)
                return dict(inject=inject,hx=hx,fluid_work=power_fluid,P_comp=power,P_gross=0.0,
                            ratio=p2/take.outlet.p,points={1:take.outlet,2:s2,3:hx.outlet})
            if upper <= lower:
                raise NoPhysicalSolution("Empty pressure search range with required machine pressure ordering")
            wp = bounded_match(zero_ua_trial, target, lower, upper, cfg)
            mw = q*cfg["exchangers"][mode]["beta"]
            pump, other = electric_auxiliary(mode, mw, wp["P_comp"], cfg)
            wp.update(mode=mode, q=q, mw=mw, take=take, P_pump=pump, P_other=other,
                      P_grid=wp["P_comp"]+pump+other, P_net=0.0)
            return wp
        Cw = q*cfg["exchangers"][mode]["beta"]*cfg["water"]["cp"]
        def coupled(values, full=False):
            p2, fraction = values
            s2, power_fluid, power = compress(f,take.outlet,p2,q,cfg)
            hxc=cfg["exchangers"]
            dp=hxc["dp_ref"]*(q/hxc["q_ref"])**2*model.rho_ref/s2.rho
            co2_cap=q*(s2.h-f.pt(p2-dp,x[7]).h)
            water_cap=Cw*(s2.T-x[7])
            Q=fraction*min(co2_cap,water_cap)
            hx = evaluate_heat(f,s2,q,x[7],mode,cfg,model.rho_ref,Q)
            inject = map_well(f,hx.outlet,q,cfg["caverns"][destination]["depth"],False,cfg)
            result = dict(inject=inject,hx=hx,fluid_work=power_fluid,P_comp=power,P_gross=0.0,
                          ratio=p2/take.outlet.p,points={1:take.outlet,2:s2,3:hx.outlet})
            if full:
                return result
            return np.array([(inject.outlet.p-target)/1e6,hx.UA_residual/cfg["exchangers"][mode]["UA"]])
        pguess = min(upper*.99,max(lower*1.02,target-600*cfg["environment"]["g"]*cfg["caverns"][destination]["depth"]))
        ntu = cfg["exchangers"][mode]["UA"]/Cw
        guess = np.array([pguess,min(.995,max(.5,1-np.exp(-ntu)))])
        def scaled(values,full=False):
            return coupled((values[0]*1e6,values[1]),full)
        try:
            solved = root(scaled,[guess[0]/1e6,guess[1]],method="hybr",options={"xtol":1e-9,"maxfev":30})
        except ModelError:
            raise
        except (ValueError,RuntimeError) as exc:
            raise NumericalFailure(f"Coupled charge pressure/UA solve failed: {exc}") from exc
        candidate = solved.x
        residual = scaled(candidate)
        # HYBR occasionally declares convergence with a sub-pascal residual that
        # is still above the configured reporting tolerance. Polish only then.
        if (not solved.success or not lower < candidate[0]*1e6 < upper or not 0 < candidate[1] < 1 or
                abs(residual[0])*1e6 > cfg["numerics"]["pressure_xtol"] or
                abs(residual[1])*cfg["exchangers"][mode]["UA"] > cfg["numerics"]["UA_residual_tolerance"]):
            candidate=np.clip(candidate,[lower/1e6+1e-10,1e-6],[upper/1e6-1e-10,1-1e-8])
            polished = least_squares(scaled,candidate,
                bounds=([lower/1e6,1e-6],[upper/1e6,1-1e-8]),x_scale=[1,.01],
                xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=20)
            candidate = polished.x
            solved = polished
        wp = scaled(candidate,True)
        wp["pressure_residual"] = wp["inject"].outlet.p-target
        wp["pressure_bracket"] = (lower,upper)
        wp["trial_failures"] = []
        if (not solved.success or not lower < candidate[0]*1e6 < upper or not 0 < candidate[1] < 1 or
                abs(wp["pressure_residual"])>cfg["numerics"]["pressure_xtol"] or
                abs(wp["hx"].UA_residual)>cfg["numerics"]["UA_residual_tolerance"]):
            raise NumericalFailure(f"Coupled charge solve did not meet tolerances: {solved.message}; residuals="
                                   f"({wp['pressure_residual']} Pa, {wp['hx'].UA_residual} W/K)")
    else:
        hx = exchange(f, take.outlet, q, x[6], mode, cfg, model.rho_ref)
        def trial(p6):
            s6, power_fluid, power = expand(f, hx.outlet, p6, q, cfg)
            inject = map_well(f, s6, q, cfg["caverns"][destination]["depth"], False, cfg)
            return dict(inject=inject, hx=hx, fluid_work=power_fluid, P_comp=0.0, P_gross=power,
                        ratio=hx.outlet.p/p6, points={4:take.outlet, 5:hx.outlet, 6:s6})
        upper = min(upper, hx.outlet.p*(1-1e-9))
        if cfg["phase"]["enabled"]:
            lower = max(lower,f.pcrit+cfg["phase"]["pressure_above_critical"])
    if upper <= lower:
        raise NoPhysicalSolution("Empty pressure search range with required machine pressure ordering")
    if mode == "discharge":
        wp = bounded_match(trial, states[destination].p, lower, upper, cfg)
    mw = q*cfg["exchangers"][mode]["beta"]
    power = wp["P_comp"] if mode == "charge" else wp["P_gross"]
    pump, other = electric_auxiliary(mode, mw, power, cfg)
    wp.update(mode=mode, q=q, mw=mw, take=take, P_pump=pump, P_other=other,
              P_grid=wp["P_comp"]+pump+other if mode == "charge" else 0.0,
              P_net=wp["P_gross"]-pump-other if mode == "discharge" else 0.0)
    return wp
