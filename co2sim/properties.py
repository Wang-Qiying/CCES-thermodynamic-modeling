"""No ideal-gas fallback, tabular surrogate, or imposed phase."""
from dataclasses import dataclass
import CoolProp.CoolProp as CP
from CoolProp import AbstractState
from .errors import PropertyFailure, NumericalFailure, ModelError


@dataclass(frozen=True)
class State:
    p: float
    T: float
    h: float
    s: float
    rho: float
    u: float
    viscosity: float
    sound: float


class CO2:
    def __init__(self, fluid="HEOS::CO2", h_tolerance=1e-7, flash_iterations=6, p_tolerance=.001):
        backend, name = fluid.split("::")
        self._as = AbstractState(backend, name)
        self.pcrit = self._as.p_critical()
        self.Tcrit = self._as.T_critical()
        self.h_tolerance = h_tolerance
        self.flash_iterations = flash_iterations
        self.p_tolerance = p_tolerance

    def _read(self):
        f = self._as
        return State(f.p(), f.T(), f.hmass(), f.smass(), f.rhomass(),
                     f.umass(), f.viscosity(), f.speed_sound())

    def _update(self, pair, a, b):
        try:
            self._as.update(pair, float(a), float(b))
            return self._read()
        except (ValueError, RuntimeError) as exc:
            raise PropertyFailure(f"HEOS flash failed: pair={pair}, inputs=({a}, {b}): {exc}") from exc

    def pt(self, p, T):
        return self._update(CP.PT_INPUTS, p, T)

    def ph(self, p, h):
        self._flash_ph(p,h)
        try:
            return self._read()
        except (ValueError,RuntimeError) as exc:
            raise PropertyFailure(f"HEOS p,h state diagnostics: {exc}") from exc

    def _flash_ph(self,p,h):
        try:
            self._as.update(CP.HmassP_INPUTS,float(h),float(p))
            # Refine the same HEOS p(rho,T), h(rho,T) equations with their Jacobian.
            # Direct rho,T evaluation avoids the PT-flash tolerance near the critical region.
            # No state/property value is overwritten or corrected algebraically.
            for _ in range(self.flash_iterations):
                error = self._as.hmass()-h
                perror = self._as.p()-p
                if abs(error) <= self.h_tolerance and abs(perror) <= self.p_tolerance:
                    return
                f = self._as
                pr = f.first_partial_deriv(CP.iP,CP.iDmass,CP.iT)
                pt = f.first_partial_deriv(CP.iP,CP.iT,CP.iDmass)
                hr = f.first_partial_deriv(CP.iHmass,CP.iDmass,CP.iT)
                ht = f.first_partial_deriv(CP.iHmass,CP.iT,CP.iDmass)
                det = pr*ht-pt*hr
                dr = (-perror*ht+pt*error)/det
                dT = (-pr*error+hr*perror)/det
                self._as.update(CP.DmassT_INPUTS,float(f.rhomass()+dr),float(f.T()+dT))
            raise NumericalFailure(f"HEOS p,h refinement did not converge; h residual={error} J/kg")
        except (ValueError,RuntimeError) as exc:
            if isinstance(exc,ModelError):
                raise
            raise PropertyFailure(f"HEOS p,h flash p={p}, h={h}: {exc}") from exc

    def ps(self, p, s):
        return self._update(CP.PSmass_INPUTS, p, s)

    def du(self, rho, u):
        return self._update(CP.DmassUmass_INPUTS, rho, u)

    def temperature_ph(self, p, h):
        self._flash_ph(p,h)
        return self._as.T()

    def pressure_rate(self, rho, u, drho, du):
        self.du(rho, u)
        f = self._as
        dp = f.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)*drho
        dp += f.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)*du
        dT = f.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)*drho
        dT += f.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)*du
        return dp, dT
