import json
from pathlib import Path
import numpy as np
import yaml
from .properties import CO2
from .caverns import initial_state, recover, geometry, heat_and_rock, STATE_NAMES
from .water import tank_derivatives
from .flowpath import solve_workpoint

# Quadratures are bookkeeping; they do not enter the eight-state physics RHS.
ENERGY_NAMES = ["E_grid", "E_net", "E_idle", "E_fluid_work", "E_geo", "E_well_heat", "E_tank_loss",
                "E_machine_loss", "E_aux", "E_potential_transfer", "E_tank_conditioning",
                "E_HX_charge", "E_HX_discharge", "E_pump_charge", "E_pump_discharge",
                "E_conditioning_charge", "E_conditioning_idle", "E_conditioning_discharge",
                "E_comp_fluid", "E_comp_electric", "E_turbine_fluid", "E_turbine_gross",
                "E_other_charge", "E_other_discharge"]


class Model:
    def __init__(self, cfg, property_model=None):
        self.cfg = cfg
        self.fluid = property_model or CO2(cfg["fluid"],cfg["numerics"]["property_h_tolerance"],cfg["numerics"]["property_flash_max_iterations"],cfg["numerics"]["property_p_tolerance"])
        self.x0, self.total_mass = initial_state(cfg, self.fluid)
        self.geometry = {i: geometry(c, cfg["rock"]) for i,c in cfg["caverns"].items()}
        hx = cfg["exchangers"]
        self.rho_ref = self.fluid.pt(hx["rho_ref_p"], hx["rho_ref_T"]).rho
        if any(float(s) <= 0 for s in cfg["scales"].values()):
            raise ValueError("Every normalization scale must be positive")

    def states(self, x):
        return recover(x, self.total_mass, self.cfg, self.fluid)

    def workpoint(self, mode, x, q):
        return solve_workpoint(self, mode, x, q)

    def derivative(self, x, wp, states=None):
        cfg = self.cfg
        states = self.states(x) if states is None else states
        heat, dr, external = heat_and_rock(cfg, self.geometry, states, x)
        mode, q = wp["mode"], wp["q"]
        dx = np.zeros(8)
        dx[1:3] = [heat["H"], heat["L"]]
        if mode == "charge":
            dx[0] = q
            dx[1] += q*wp["inject"].outlet.h
            dx[2] -= q*states["L"].h
        elif mode == "discharge":
            dx[0] = -q
            dx[1] -= q*states["H"].h
            dx[2] += q*wp["inject"].outlet.h
        dx[3:5] = [dr["H"], dr["L"]]
        dx[5:8], tank_loss, tank_conditioning = tank_derivatives(x, mode, wp, cfg)
        rates = {}
        for i,k in [("H",1),("L",2)]:
            m = x[0] if i == "H" else self.total_mass-x[0]
            dm = dx[0] if i == "H" else -dx[0]
            rates[f"dp_{i}"], rates[f"dT_{i}"] = self.fluid.pressure_rate(
                states[i].rho, states[i].u, dm/cfg["caverns"][i]["volume"], (dx[k]-states[i].u*dm)/m)
        well_heat = sum(wp[k].heat for k in ["take","inject"]) if mode != "idle" else 0.0
        work = wp["fluid_work"]*(1 if mode == "charge" else -1)
        machine_loss = (wp["P_comp"]-wp["fluid_work"] if mode == "charge" else
                        wp["fluid_work"]-wp["P_gross"] if mode == "discharge" else 0.0)
        # The cavern representative elevations are z=-depth; include mass-transfer PE.
        pe_dot = cfg["environment"]["g"]*(cfg["caverns"]["L"]["depth"]-cfg["caverns"]["H"]["depth"])*dx[0]
        hxQ=wp["hx"].Q if wp["hx"] is not None else 0.0
        de = np.array([wp["P_grid"], wp["P_net"], wp["P_grid"] if mode == "idle" else 0.0,
                       work, sum(external.values()), well_heat, tank_loss, machine_loss,
                       wp["P_pump"]+wp["P_other"],pe_dot,tank_conditioning,
                       hxQ if mode=="charge" else 0.0,hxQ if mode=="discharge" else 0.0,
                       wp["P_pump"] if mode=="charge" else 0.0,
                       wp["P_pump"] if mode=="discharge" else 0.0,
                       tank_conditioning if mode=="charge" else 0.0,
                       tank_conditioning if mode=="idle" else 0.0,
                       tank_conditioning if mode=="discharge" else 0.0,
                       wp["fluid_work"] if mode=="charge" else 0.0,
                       wp["P_comp"] if mode=="charge" else 0.0,
                       wp["fluid_work"] if mode=="discharge" else 0.0,
                       wp["P_gross"] if mode=="discharge" else 0.0,
                       wp["P_other"] if mode=="charge" else 0.0,
                       wp["P_other"] if mode=="discharge" else 0.0])
        diagnostic = dict(**rates, Q_H=heat["H"], Q_L=heat["L"], tank_loss=tank_loss,
                          geo_heat=sum(external.values()),well_heat=well_heat,
                          tank_conditioning=tank_conditioning)
        return dx, de, diagnostic

    def evaluate(self, mode, x, q):
        states = self.states(x)
        wp = solve_workpoint(self, mode, x, q, states)
        dx, de, diag = self.derivative(x, wp, states)
        return states, wp, dx, de, diag

    def net_power(self, x, q_d):
        return self.workpoint("discharge", x, q_d)["P_net"]

    def stored_energy(self, x):
        w = self.cfg["water"]
        return (x[1]+x[2]+self.geometry["H"]["C_rock"]*x[3]+self.geometry["L"]["C_rock"]*x[4]
                +w["cp"]*(x[5]*x[6]+(w["mass_total"]-x[5])*x[7]))

    def save_snapshot(self, path, x, time=0.0, energy=None):
        data = dict(state_names=STATE_NAMES, x=list(map(float,x[:8])), time=float(time),
                    total_CO2_mass=self.total_mass, config=self.cfg,
                    energy_names=ENERGY_NAMES,
                    energy=None if energy is None else list(map(float, energy)))
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_snapshot(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(d["config"])
        if not np.isclose(model.total_mass, d["total_CO2_mass"], rtol=1e-12):
            raise ValueError("Snapshot inventory inconsistent with EOS/configuration")
        return model, np.array(d["x"]), d["time"], d["energy"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
