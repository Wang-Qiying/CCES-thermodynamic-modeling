def tank_derivatives(x, mode, wp, cfg):
    w = cfg["water"]
    if not w.get("enabled", True):
        return (0.0, 0.0, 0.0), 0.0, 0.0
    if not w.get("enabled", True):
        return (0.0, 0.0, 0.0), 0.0, 0.0
    Mh, Th, Tc = x[5:8]
    Mc = w["mass_total"]-Mh
    cp = w["cp"]
    hot_loss = w["loss_conductance"]*(Th-cfg["environment"]["T_ambient"])
    cold_loss = w["loss_conductance"]*(Tc-cfg["environment"]["T_ambient"])
    dMh, hot_in, cold_in = 0.0, 0.0, 0.0
    if mode == "charge":
        dMh = wp["mw"]
        hot_in = dMh*cp*(wp["hx"].water_out_T-Th)
    elif mode == "discharge":
        dMh = -wp["mw"]
        cold_in = -dMh*cp*(wp["hx"].water_out_T-Tc)
    if w.get("fixed_temperatures",False):
        conditioning = hot_loss+cold_loss
        if mode == "charge":
            conditioning += dMh*cp*(Th-wp["hx"].water_out_T)
        elif mode == "discharge":
            conditioning += (-dMh)*cp*(Tc-wp["hx"].water_out_T)
        return (dMh,0.0,0.0),hot_loss+cold_loss,conditioning
    dTh = (hot_in-hot_loss)/(Mh*cp)
    dTc = (cold_in-cold_loss)/(Mc*cp)
    return (dMh,dTh,dTc),hot_loss+cold_loss,0.0


def electric_auxiliary(mode, mw, gross_or_input, cfg):
    if not cfg["water"].get("enabled", True):
        return 0.0, cfg["auxiliary"][mode]["P0"]+cfg["auxiliary"][mode]["fraction"]*gross_or_input if mode != "idle" else cfg["auxiliary"]["idle_power"]
    if mode == "idle":
        return 0.0, cfg["auxiliary"]["idle_power"]
    w, a = cfg["water"], cfg["auxiliary"][mode]
    if not w.get("enabled", True):
        return 0.0, a["P0"]+a["fraction"]*gross_or_input
    pump = mw*w["pump_dp"]/(w["density"]*w["pump_eta_total"])
    return pump, a["P0"]+a["fraction"]*gross_or_input
