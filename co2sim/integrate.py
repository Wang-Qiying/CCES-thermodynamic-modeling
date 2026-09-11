"""Adaptive RK45, event localization on accepted dense interpolants.

Each RHS evaluates the algebraic working point. Only accepted steps affect history.
Boundary zeros are allowed if stationary or inward; negative crossings terminate.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import RK45
from scipy.optimize import brentq
from .constraints import evaluate_constraints
from .errors import ModelError, NumericalFailure
from .model import ENERGY_NAMES


@dataclass
class Stage:
    mode: str
    q: float
    start: float
    end: float
    y0: np.ndarray
    yend: np.ndarray
    segments: list
    event: dict

    def at(self, t):
        if abs(t-self.start) < 1e-9:
            return self.y0.copy()
        for a,b,dense in self.segments:
            if a-1e-9 <= t <= b+1e-9:
                return dense(t)
        if abs(t-self.end) < 1e-8:
            return self.yend.copy()
        raise ValueError(f"Time {t} outside stored stage [{self.start},{self.end}]")

    def flow(self, x):
        """Return prescribed or state-feedback mass flow for this stage."""
        return float(self.q(x)) if callable(self.q) else float(self.q)


def run_stage(model, mode, x0, q, duration, start=0.0, energy0=None, progress=True,
              terminal_state_event=None, terminal_kind="state_target",
              terminal_reason="Requested state target reached",
              constraint_margin_floor=0.0):
    num = model.cfg["numerics"]
    y0 = np.r_[x0, np.zeros(len(ENERGY_NAMES)) if energy0 is None else energy0]
    segments = []
    last_key, last_eval = None, None

    def evaluate(y):
        nonlocal last_key, last_eval
        key = y[:8].tobytes()
        if key != last_key:
            q_now = float(q(y[:8])) if callable(q) else float(q)
            last_eval = model.evaluate(mode, y[:8], q_now)
            last_key = key
        return last_eval

    def margins(y):
        st, wp, dx, de, diag = evaluate(y)
        return {r.name:r for r in evaluate_constraints(model, mode, y[:8], st, wp, diag) if r.active and r.evaluated}

    def rhs(t,y):
        st, wp, dx, de, diag = evaluate(y)
        return np.r_[dx,de]

    def result(t, y, kind, reason, triggers=None):
        event = dict(mode=mode, time=float(t), kind=kind, reason=reason, constraints=triggers or [])
        if progress:
            print(f"{mode}: t={t/3600:.6f} h, {kind}: {reason}", flush=True)
        return Stage(mode,q,start,float(t),y0,y.copy(),segments,event)

    try:
        previous = margins(y0)
    except ModelError as exc:
        return result(start,y0,exc.kind,str(exc))
    margin_floor = float(constraint_margin_floor)
    if margin_floor < 0:
        raise ValueError("constraint_margin_floor must be nonnegative")
    bad = [
        r.name for r in previous.values()
        if r.normalized_margin < margin_floor-num["event_margin_tolerance"]
    ]
    if bad:
        kind = "constraint_violation" if margin_floor == 0 else "margin_floor_violation"
        reason = (
            "Initial working point violates constraints"
            if margin_floor == 0
            else f"Initial working point is below the normalized margin floor {margin_floor:g}"
        )
        return result(start,y0,kind,reason,bad)
    previous_terminal = None
    if terminal_state_event is not None:
        previous_terminal = float(terminal_state_event(y0))
        if previous_terminal <= 0:
            return result(start, y0, terminal_kind, terminal_reason)
    if duration <= 0:
        return result(start,y0,"duration_limit","Requested duration is zero")
    atol = np.r_[num["atol"], np.full(len(ENERGY_NAMES),num["energy_atol"])]
    solver = RK45(rhs,start,y0,start+duration,rtol=num["rtol"],atol=atol,max_step=num["max_step"])
    next_progress = start+1800
    while solver.status == "running":
        tprev, yprev = solver.t, solver.y.copy()
        try:
            solver.step()
            if solver.status == "failed":
                raise NumericalFailure("Adaptive RK45 failed to advance")
            current = margins(solver.y)
        except ModelError as exc:
            # Rejected/invalid tentative evaluations never modify accepted history.
            # Localize a failure by successively reducing the permitted step.
            reduced = min(solver.max_step, max(solver.h_abs, 1e-9))/4
            if reduced < num["event_time_xtol"]:
                return result(tprev,yprev,exc.kind,str(exc))
            try:
                solver = RK45(rhs,tprev,yprev,start+duration,rtol=num["rtol"],atol=atol,max_step=reduced)
            except ModelError as exc2:
                return result(tprev,yprev,exc2.kind,str(exc2))
            continue
        dense = solver.dense_output()
        crossings = []
        for name,r in current.items():
            if r.normalized_margin < margin_floor-num["event_margin_tolerance"]:
                if previous[name].normalized_margin <= margin_floor:
                    root = tprev
                else:
                    root = brentq(lambda t:margins(dense(t))[name].normalized_margin-margin_floor,
                                  tprev,solver.t,xtol=num["event_time_xtol"])
                crossings.append((root,name))
        terminal_crossing = None
        current_terminal = None
        if terminal_state_event is not None:
            current_terminal = float(terminal_state_event(solver.y))
            if current_terminal <= 0:
                terminal_crossing = (
                    tprev if previous_terminal <= 0 else
                    brentq(lambda t: float(terminal_state_event(dense(t))),
                           tprev, solver.t, xtol=num["event_time_xtol"])
                )
        if terminal_crossing is not None and (
                not crossings or terminal_crossing <= min(t for t, _ in crossings)):
            yend = dense(terminal_crossing)
            if terminal_crossing > tprev:
                segments.append((tprev, terminal_crossing, dense))
            return result(terminal_crossing, yend, terminal_kind, terminal_reason)
        if crossings:
            tend = min(t for t,n in crossings)
            yend = dense(tend)
            triggers = [n for t,n in crossings if abs(t-tend) <= 2*num["event_time_xtol"]]
            if tend > tprev:
                segments.append((tprev,tend,dense))
            kind = "operating_boundary" if margin_floor == 0 else "margin_floor"
            reason = (
                "First outward constraint crossing"
                if margin_floor == 0
                else f"First normalized margin reached the floor {margin_floor:g}"
            )
            return result(tend,yend,kind,reason,triggers)
        segments.append((tprev,solver.t,dense))
        previous = current
        if terminal_state_event is not None:
            previous_terminal = current_terminal
        if progress and solver.t >= next_progress:
            states = evaluate(solver.y)[0]
            print(f"  {mode} {solver.t/3600:.3f} h: pH={states['H'].p/1e6:.4f}, pL={states['L'].p/1e6:.4f} MPa",flush=True)
            next_progress += 1800
    return result(solver.t,solver.y,"duration_limit","Reached prescribed stage duration")


def run_process(model, progress=True):
    op = model.cfg["operation"]
    trial = run_stage(model,"charge",model.x0,op["q_charge"],op["active_duration_max"],progress=progress)
    logs = [dict(trial.event, role="charge_preparation_trial")]
    if trial.end <= trial.start:
        return [], trial, logs
    tcharge_requested = trial.end*op["charge_mass_fraction"]  # fixed q -> mass fraction equals time fraction
    tcharge = tcharge_requested
    reversible_boundary = None
    if op.get("enforce_reversible_charge",False):
        # Charging can remain locally feasible after L has fallen too low to
        # accept the return flow while the turbine outlet stays supercritical.
        # Screen the post-idle discharge work point on the already-integrated
        # charge trajectory; no parameter sweep or repeated charge integration.
        def reversible_at(t):
            candidate = trial.at(t)
            idle_probe = run_stage(model,"idle",candidate[:8],0.0,op["idle_duration"],t,candidate[8:],False)
            if idle_probe.event["kind"] != "duration_limit":
                return False
            discharge_probe = run_stage(model,"discharge",idle_probe.yend[:8],op["q_discharge"],1.0,
                                        idle_probe.end,idle_probe.yend[8:],False)
            return discharge_probe.event["kind"] == "duration_limit" and discharge_probe.end > discharge_probe.start

        if not reversible_at(tcharge_requested):
            high, low = tcharge_requested, None
            # Find the upper edge of the feasible interval before polishing it.
            for fraction in np.linspace(0.9,0.0,10):
                candidate_t = tcharge_requested*float(fraction)
                if reversible_at(candidate_t):
                    low = candidate_t
                    break
                high = candidate_t
            if low is None:
                logs.append(dict(mode="charge",time=0.0,kind="no_reversible_charge_state",
                    reason="No post-idle discharge work point found on the feasible charge trajectory",constraints=[]))
                return [],trial,logs
            while high-low > max(1.0,model.cfg["numerics"]["event_time_xtol"]):
                mid = 0.5*(low+high)
                if reversible_at(mid): low = mid
                else: high = mid
            reversible_boundary = low
            tcharge = reversible_boundary*op.get("reversible_reserve_fraction",0.98)
    yc = trial.at(tcharge)
    actual_segments = [(a,min(b,tcharge),d) for a,b,d in trial.segments if a<tcharge]
    fraction = tcharge/trial.end
    reason = f"{100*fraction:g}% of feasible trial transferred mass; boundary reserve retained"
    if reversible_boundary is not None:
        reason = (f"Reversible endpoint retained at {100*fraction:g}% of charge-side trial; "
                  "post-idle discharge feasibility reserve applied")
    charge = Stage("charge",op["q_charge"],0.0,tcharge,trial.y0,yc,actual_segments,
                   dict(mode="charge",time=tcharge,kind="preparation_snapshot",
                        reason=reason,
                        constraints=[]))
    logs.append(charge.event)
    stages = [charge]
    idle = run_stage(model,"idle",yc[:8],0.0,op["idle_duration"],tcharge,yc[8:],progress)
    stages.append(idle)
    logs.append(idle.event)
    if idle.event["kind"] != "duration_limit":
        return stages,trial,logs
    charged_mass = abs(charge.yend[0]-charge.y0[0])
    discharge_duration = op["active_duration_max"]
    if op.get("match_discharge_mass_to_charge",False):
        discharge_duration = min(discharge_duration,charged_mass/op["q_discharge"])
    discharge = run_stage(model,"discharge",idle.yend[:8],op["q_discharge"],discharge_duration,idle.end,idle.yend[8:],progress)
    if (op.get("match_discharge_mass_to_charge",False) and
            discharge.event["kind"] == "duration_limit" and
            abs(abs(discharge.yend[0]-discharge.y0[0])-charged_mass) <= max(1e-6,charged_mass*1e-10)):
        discharge.event.update(kind="matched_charge_mass_target",
            reason="Returned the same CO2 mass transferred during charge")
    stages.append(discharge)
    logs.append(discharge.event)
    return stages,trial,logs
