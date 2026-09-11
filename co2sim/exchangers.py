from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .errors import ModeInapplicable, NumericalFailure, ModelError


@dataclass
class HXResult:
    outlet: object
    Q: float
    water_out_T: float
    dp: float
    min_approach: float
    UA_required: float
    UA_residual: float
    energy_residual: float
    profile: list


def evaluate_heat(fluid, inlet, q, water_T, mode, cfg, rho_ref, Q):
    """Evaluate the counterflow discretization at a prescribed total heat rate."""
    hxc, w = cfg["exchangers"], cfg["water"]
    design = hxc[mode]
    mw = design["beta"]*q
    Cw = mw*w["cp"]
    direction = -1 if mode == "charge" else 1
    zero_duty = design["UA"] == 0 and Q == 0
    if direction*(water_T-inlet.T) <= 0 and not zero_duty:
        raise ModeInapplicable(f"{mode}: CO2 inlet={inlet.T:.5f} K, water inlet={water_T:.5f} K; wrong heat direction")
    dp = hxc["dp_ref"]*(q/hxc["q_ref"])**2*rho_ref/inlet.rho
    pout = inlet.p-dp
    n = int(hxc["segments"])
    pos = np.linspace(0, 1, n+1)
    pressures = inlet.p-dp*pos
    h = inlet.h+direction*Q*pos/q
    Tc = np.array([fluid.temperature_ph(p, hi) for p, hi in zip(pressures, h)])
    # Water runs from x=1 (its inlet) to x=0 (its outlet).
    Tw = water_T-direction*Q*(1-pos)/Cw
    delta = direction*(Tw-Tc)
    if delta.min() <= 0 and not zero_duty:
        raise ModeInapplicable(f"{mode}: prescribed heat rate crosses temperatures; min delta={delta.min()} K")
    if zero_duty:
        outlet = fluid.ph(pout,inlet.h)
        profile = [dict(x=float(xx),p=float(p),h=float(inlet.h),
                        T_CO2=float(t),T_water=float(water_T),delta_T=float(direction*(water_T-t)))
                   for xx,p,t in zip(pos,pressures,Tc)]
        return HXResult(outlet,0.0,float(water_T),dp,float(delta.min()),0.0,0.0,0.0,profile)
    a, b = delta[:-1], delta[1:]
    diff = a-b
    reciprocal_lmtd = np.empty(n)
    close = abs(diff) < 1e-8*np.maximum(a, b)
    reciprocal_lmtd[close] = 1/((a[close]+b[close])/2)
    reciprocal_lmtd[~close] = np.log(a[~close]/b[~close])/diff[~close]
    required = Q/n*reciprocal_lmtd.sum()
    outlet = fluid.ph(pout, h[-1])
    residual = q*direction*(outlet.h-inlet.h)-Cw*(-direction)*(Tw[0]-water_T)
    profile = [dict(x=float(xx), p=float(p), h=float(hi), T_CO2=float(t), T_water=float(tw), delta_T=float(d))
               for xx,p,hi,t,tw,d in zip(pos, pressures, h, Tc, Tw, delta)]
    return HXResult(outlet, float(Q), float(Tw[0]), dp, float(delta.min()), float(required),
                    float(required-design["UA"]), float(residual), profile)


def exchange(fluid, inlet, q, water_T, mode, cfg, rho_ref):
    hxc, w = cfg["exchangers"], cfg["water"]
    design = hxc[mode]
    mw = design["beta"]*q
    Cw = mw*w["cp"]
    direction = -1 if mode == "charge" else 1
    if design["UA"] == 0:
        return evaluate_heat(fluid,inlet,q,water_T,mode,cfg,rho_ref,0.0)
    if direction*(water_T-inlet.T) <= 0:
        raise ModeInapplicable(f"{mode}: CO2 inlet={inlet.T:.5f} K, water inlet={water_T:.5f} K; wrong heat direction")
    dp = hxc["dp_ref"]*(q/hxc["q_ref"])**2*rho_ref/inlet.rho
    pout = inlet.p-dp
    n = int(hxc["segments"])
    pos = np.linspace(0, 1, n+1)  # spatial coordinate in CO2 flow direction
    pressures = inlet.p-dp*pos

    def profiles(Q):
        h = inlet.h+direction*Q*pos/q
        Tc = np.array([fluid.temperature_ph(p, hi) for p, hi in zip(pressures, h)])
        # Water runs from x=1 (its inlet) to x=0 (its outlet).
        Tw = water_T-direction*Q*(1-pos)/Cw
        delta = direction*(Tw-Tc)
        return h, Tc, Tw, delta

    # Reversible end-temperature caps only bracket the root; do not clip delivered Q.
    end_h = fluid.pt(pout, water_T).h
    cap = min(q*direction*(end_h-inlet.h), Cw*direction*(water_T-inlet.T))
    if cap <= 0 or profiles(0)[3].min() <= 0:
        raise ModeInapplicable(f"{mode}: pressure-drop profile does not support positive counterflow transfer")
    cap = float(cap)
    if profiles(cap)[3].min() <= 1e-9:
        cap = brentq(lambda Q: profiles(Q)[3].min()-1e-9, 0, cap, xtol=1e-7)

    def ua(Q):
        return evaluate_heat(fluid,inlet,q,water_T,mode,cfg,rho_ref,Q).UA_required

    upper = cap*(1-1e-12)
    if not ua(upper) > design["UA"]:
        raise NumericalFailure("HX positive-temperature bracket cannot reach installed UA")
    try:
        Q = brentq(lambda Q: ua(Q)-design["UA"], 0, upper, xtol=cfg["numerics"]["heat_xtol"], rtol=1e-11)
    except ModelError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise NumericalFailure(f"HX UA root failed: {exc}") from exc
    return evaluate_heat(fluid,inlet,q,water_T,mode,cfg,rho_ref,Q)
