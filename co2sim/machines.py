def compress(fluid, inlet, p_out, q, cfg):
    c = cfg["compressor"]
    ideal = fluid.ps(p_out, inlet.s)
    outlet = fluid.ph(p_out, inlet.h+(ideal.h-inlet.h)/c["eta_is"])
    fluid_work = q*(outlet.h-inlet.h)
    electricity = fluid_work/(c["eta_motor"]*c["eta_mech"])
    return outlet, fluid_work, electricity


def expand(fluid, inlet, p_out, q, cfg):
    c = cfg["turbine"]
    ideal = fluid.ps(p_out, inlet.s)
    outlet = fluid.ph(p_out, inlet.h-c["eta_is"]*(inlet.h-ideal.h))
    fluid_work = q*(inlet.h-outlet.h)
    electricity = fluid_work*c["eta_generator"]*c["eta_mech"]
    return outlet, fluid_work, electricity
