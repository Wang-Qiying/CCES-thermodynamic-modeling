import numpy as np

STATE_NAMES = ["m_H", "U_H", "U_L", "T_r_H", "T_r_L", "M_w_hot", "T_w_hot", "T_w_cold"]


def geometry(c, rock):
    height = c["volume"]/(np.pi*c["radius"]**2)
    area = 2*np.pi*c["radius"]*height + 2*np.pi*c["radius"]**2
    return {"height": height, "area": area,
            "C_rock": rock["density"]*rock["cp"]*area*rock["thickness"],
            "K_rock": rock["conductivity"]*area/rock["thickness"]}


def initial_state(cfg, fluid):
    states = {i: fluid.pt(c["p_initial"], c["T_initial"]) for i, c in cfg["caverns"].items()}
    mass = {i: s.rho*cfg["caverns"][i]["volume"] for i, s in states.items()}
    w = cfg["water"]
    rock_by_cavern = cfg["rock"].get("T_initial_by_cavern", {})
    x = np.array([mass["H"], mass["H"]*states["H"].u, mass["L"]*states["L"].u,
                  rock_by_cavern.get("H",cfg["rock"]["T_initial"]),
                  rock_by_cavern.get("L",cfg["rock"]["T_initial"]),
                  w["mass_total"]*w["initial_hot_fraction"],
                  w.get("T_hot_initial",w.get("T_initial")),
                  w.get("T_cold_initial",w.get("T_initial"))])
    return x, mass["H"]+mass["L"]


def recover(x, total_mass, cfg, fluid):
    m = {"H": x[0], "L": total_mass-x[0]}
    return {i: fluid.du(m[i]/cfg["caverns"][i]["volume"], x[k]/m[i])
            for i, k in [("H", 1), ("L", 2)]}


def heat_and_rock(cfg, geometry_map, states, x):
    heat, dr, external = {}, {}, {}
    for i, k in [("H", 3), ("L", 4)]:
        geo = geometry_map[i]
        env = cfg["environment"]
        Tinf = env["T_surface"]+env["geothermal_gradient"]*cfg["caverns"][i]["depth"]
        heat[i] = cfg["rock"]["alpha"]*geo["area"]*(x[k]-states[i].T)
        external[i] = geo["K_rock"]*(Tinf-x[k])
        dr[i] = (external[i]-heat[i])/geo["C_rock"]
    return heat, dr, external
