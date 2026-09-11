"""Vertical quasi-steady well, explicit midpoint in space, real-fluid p,h flashes."""
from dataclasses import dataclass
import numpy as np
from .errors import NoPhysicalSolution


@dataclass
class WellResult:
    outlet: object
    profile: list
    heat: float
    inventory: float
    energy_residual: float
    max_velocity: float
    max_mach: float
    kinetic_change: float


def darcy_factor(Re, relative_roughness):
    if Re <= 0:
        return 0.0
    if Re <= 2300:
        return 64.0/Re
    # Haaland turbulent approximation; linear transition 2300--4000.
    turbulent = (-1.8*np.log10((relative_roughness/3.7)**1.11+6.9/Re))**-2
    if Re >= 4000:
        return turbulent
    a = (Re-2300)/1700
    return (1-a)*64/Re+a*turbulent


def map_well(fluid, inlet, q, depth, upward, cfg):
    if q <= 0:
        raise NoPhysicalSolution("Well flow equations require q>0; idle wells are N/A")
    w, env = cfg["well"], cfg["environment"]
    if not w.get("enabled", True):
        # System-level ablation: the machine is connected directly to the
        # cavern representative state.  Keep a valid WellResult so the rest of
        # the flow-path and reporting code remains identical across variants.
        row = {"s": 0.0, "z": 0.0, "p": inlet.p, "T": inlet.T,
               "h": inlet.h, "rho": inlet.rho, "velocity": 0.0,
               "mach": 0.0, "Re": 0.0, "f_D": 0.0,
               "T_geo": inlet.T, "heat_per_length": 0.0}
        return WellResult(inlet, [row], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    D, g = w["diameter"], env["g"]
    A = np.pi*D*D/4
    n = int(w["segments"])
    ds = depth/n
    sign = 1 if upward else -1
    z0 = -depth if upward else 0.0

    def rhs(s, p, h):
        st = fluid.ph(p, h)
        v = q/(st.rho*A)
        Re = q*D/(A*st.viscosity)
        f = darcy_factor(Re, w["roughness"]/D) if w["friction"] else 0.0
        z = z0+sign*s
        Tg = env["T_surface"]-env["geothermal_gradient"]*z
        heat_length = w["U"]*np.pi*D*(Tg-st.T)
        dp = -st.rho*g*sign-f*st.rho*v*v/(2*D)
        dh = heat_length/q-g*sign
        row = {"s": s, "z": z, "p": st.p, "T": st.T, "h": st.h,
               "rho": st.rho, "velocity": v, "mach": v/st.sound, "Re": Re,
               "f_D": f, "T_geo": Tg, "heat_per_length": heat_length}
        return dp, dh, row, st

    p, h = inlet.p, inlet.h
    profile, heat, inventory = [], 0.0, 0.0
    for k in range(n):
        s = k*ds
        dp, dh, row, _ = rhs(s, p, h)
        profile.append(row)
        dpm, dhm, mid, _ = rhs(s+ds/2, p+dp*ds/2, h+dh*ds/2)
        profile.append(mid)
        p, h = p+dpm*ds, h+dhm*ds
        heat += mid["heat_per_length"]*ds
        inventory += mid["rho"]*A*ds
    _, _, row, outlet = rhs(depth, p, h)
    profile.append(row)
    residual = q*(outlet.h-inlet.h+g*sign*depth)-heat
    kinetic = (profile[-1]["velocity"]**2-profile[0]["velocity"]**2)/2
    return WellResult(outlet, profile, heat, inventory, residual,
                      max(r["velocity"] for r in profile), max(r["mach"] for r in profile), kinetic)
