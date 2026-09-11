"""Human-readable documentation generated from the actual outputs."""
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from .results import write_json


def flatten(d,prefix=""):
    for k,v in d.items():
        name=f"{prefix}.{k}" if prefix else k
        if isinstance(v,dict): yield from flatten(v,name)
        else: yield name,v


def parameter_catalog(model,folder):
    cfg=model.cfg
    units={
        "fluid":"backend::fluid", "g":"m/s2", "T_ambient":"K", "T_surface":"K", "geothermal_gradient":"K/m",
        "volume":"m3","radius":"m","depth":"m","p_initial":"Pa","T_initial":"K","p_min":"Pa","p_max":"Pa","T_max":"K","T_min":"K",
        "alpha":"W/(m2 K)","density":"kg/m3","cp":"J/(kg K)","conductivity":"W/(m K)","thickness":"m",
        "diameter":"m","roughness":"m","U":"W/(m2 K)","segments":"1","friction":"boolean","pressure_rating":"Pa","mach_warning":"1","kinetic_head_warning_fraction":"1",
        "eta_is":"1","eta_motor":"1","eta_mech":"1","power_max":"W","ratio_max":"1","T_out_max":"K","eta_generator":"1",
        "UA":"W/K","beta":"kg_water/kg_CO2","min_approach":"K","design_approach_diagnostic":"K","dp_ref":"Pa","q_ref":"kg/s","rho_ref_p":"Pa","rho_ref_T":"K",
        "mass_total":"kg","initial_hot_fraction":"1","min_fraction":"1","hot_capacity":"kg","cold_capacity":"kg","loss_conductance":"W/K","fixed_temperatures":"boolean","T_hot_initial":"K","T_cold_initial":"K",
        "absolute_pressure":"Pa","pump_dp":"Pa","pump_eta_total":"1","npsh_required":"m","P0":"W","fraction":"1","idle_power":"W",
        "enabled":"boolean","pressure_above_critical":"Pa","temperature_above_critical":"K","q_charge":"kg/s","q_discharge":"kg/s","q_min":"kg/s","q_max":"kg/s",
        "active_duration_max":"s","charge_mass_fraction":"1","enforce_reversible_charge":"boolean","reversible_reserve_fraction":"1","match_discharge_mass_to_charge":"boolean","idle_duration":"s","require_positive_net_power":"boolean",
        "pressure_rate_max":"Pa/s","temperature_rate_max":"K/s","surface_CO2_inventory_volume":"m3",
        "pressure":"Pa","temperature":"K","power":"W","flow":"kg/s","water_mass_fraction":"1","ratio":"1","approach":"K","pressure_rate":"Pa/s","temperature_rate":"K/s","length":"m",
        "property_h_tolerance":"J/kg","property_p_tolerance":"Pa","property_flash_max_iterations":"1","max_step":"s","output_interval":"s","rtol":"1",
        "atol":"[kg,J,J,K,K,kg,K,K]","energy_atol":"J","pressure_xtol":"Pa","heat_xtol":"W","UA_residual_tolerance":"W/K","pressure_search_min":"Pa","pressure_search_max":"Pa",
        "pressure_bracket_points":"1","event_time_xtol":"s","event_margin_tolerance":"1"}
    assumed={"environment.g","well.segments","well.mach_warning","well.kinetic_head_warning_fraction",
             "exchangers.rho_ref_p","exchangers.rho_ref_T","water.density","scales.pressure_rate","scales.temperature_rate","scales.length"}
    boundary_provenance={
        "caverns.H.p_min":("文献初筛值","CEEGS D2.1 表9：1500 m盐穴最低压力9.71 MPa，约为岩静压力的0.3倍；场址设计仍需地质力学校准"),
        "caverns.H.p_max":("文献初筛值","CEEGS D2.1 表9：1500 m盐穴最高压力25.90 MPa，约为岩静压力的0.8倍；场址设计还受最小主应力、压裂和井筒承压限制"),
        "caverns.L.p_min":("文献规则修正值","CEEGS D2.1 表9的800 m盐穴力学下限为5.18 MPa；本算例为保持超临界CO2，取max(5.18 MPa, 临界压力+0.20 MPa)并圆整为7.60 MPa"),
        "caverns.L.p_max":("文献初筛值","CEEGS D2.1 表9：800 m盐穴最高压力13.81 MPa，约为岩静压力的0.8倍；场址设计还受最小主应力、压裂和井筒承压限制"),
        "caverns.H.T_max":("暂定筛选值","80 degC是未经过盐岩热-力耦合或井筒材料校核的暂定上限，不应视为工程安全边界"),
        "caverns.L.T_max":("暂定筛选值","80 degC是未经过盐岩热-力耦合或井筒材料校核的暂定上限，不应视为工程安全边界"),
        "phase.pressure_above_critical":("工程裕量假设","CO2临界压力来自HEOS物性；其上0.20 MPa裕量为本算例假设，尚未按控制误差和近临界物性敏感性标定"),
        "phase.temperature_above_critical":("工程裕量假设","CO2临界温度来自HEOS物性；其上1 K裕量为本算例假设，尚未按控制误差和近临界物性敏感性标定"),
        "compressor.ratio_max":("设备包络假设","压比上限3.0未由具体压缩机性能图标定；只能作为概念模型筛选值"),
        "compressor.power_max":("设备包络假设","10 MW为未绑定具体机型的暂定电输入上限；不是由本算例热力状态反算的额定功率"),
        "turbine.ratio_max":("设备包络假设","膨胀比上限3.0未由具体膨胀机性能图标定；只能作为概念模型筛选值"),
        "turbine.power_max":("设备包络假设","5 MW为未绑定具体机型的暂定毛发电上限；不是由本算例热力状态反算的额定功率"),
    }
    rows=[]
    for key,value in flatten(cfg):
        source="用户给定"; note="用户建议的初步假设算例；不是已验证工程设计值"
        if value is None:
            source="待标定"; note="缺少可靠阈值或几何数据，默认未评估；不算通过"
        elif key=="fluid":
            source="文献依据"; note="用户要求 HEOS::CO2；CoolProp/Span-Wagner EOS；见 README 参考资料"
        elif key in assumed or key.startswith("numerics."):
            source="假设"; note="本实现的参考点、数值设置或适用性提示；不是工程安全标准"
            if key=="numerics.max_step":source="用户给定";note="用户建议的初始自适应积分最大步长；区别于输出间隔"
        elif key in ("water.hot_capacity","water.cold_capacity"):
            source="计算得到";note="(1-min_fraction)*mass_total；容量与总水量和保有量相容"
        if key in boundary_provenance:
            source,note=boundary_provenance[key]
        leaf=key.split('.')[-1]
        unit="K" if key.startswith("rock.T_initial_by_cavern.") else units[leaf]
        rows.append(dict(parameter=key,value=json.dumps(value,ensure_ascii=False),unit=unit,source=source,note=note))
    derived={
        "derived.CO2_mass_total":(model.total_mass,"kg","两穴 HEOS 初态密度乘体积求和"),
        "derived.H_mass":(model.x0[0],"kg","rho_H*V_H"),
        "derived.L_mass":(model.total_mass-model.x0[0],"kg","rho_L*V_L"),
        "derived.H_U":(model.x0[1],"J","m_H*u_H"),"derived.L_U":(model.x0[2],"J","m_L*u_L"),
        "derived.HX_rho_reference":(model.rho_ref,"kg/m3","HEOS::CO2 at fixed rho_ref_p,rho_ref_T"),
        "derived.p_critical":(model.fluid.pcrit,"Pa","CoolProp HEOS CO2 临界常数"),
        "derived.T_critical":(model.fluid.Tcrit,"K","CoolProp HEOS CO2 临界常数"),
        "derived.water_mass_min":(cfg["water"]["mass_total"]*cfg["water"]["min_fraction"],"kg","min_fraction*mass_total"),
        "derived.hot_mass_initial":(model.x0[5],"kg","initial_hot_fraction*mass_total"),
        "derived.cold_mass_initial":(cfg["water"]["mass_total"]-model.x0[5],"kg","mass_total-hot_mass_initial"),
        "derived.water_saturation_temperature":(CP.PropsSI("T","P",cfg["water"]["absolute_pressure"],"Q",0,"HEOS::Water"),"K","水罐绝对压力下饱和温度；仅诊断，不是 NPSH 评估")}
    for i,g in model.geometry.items():
        for k,val in g.items():derived[f"derived.{i}.{k}"]=(val,{"height":"m","area":"m2","C_rock":"J/K","K_rock":"W/K"}[k],"用户给定圆柱及等效围岩关系计算")
    for k,(v,u,n) in derived.items():rows.append(dict(parameter=k,value=v,unit=u,source="计算得到",note=n))
    pd.DataFrame(rows).to_csv(Path(folder)/"parameter_catalog.csv",index=False,encoding="utf-8-sig")
    unavailable=[dict(parameter=k,reason="未提供阈值/数据") for k,v in flatten(cfg) if v is None]
    unavailable += [dict(parameter="machine_internal_phase_path",reason="只检查实际进出口；未建立机内完整流道相态模型"),
                    dict(parameter="fixed_speed_machine_map",reason="给定流量协议，不验证固定转速设备的实际工作范围"),
                    dict(parameter="geomechanical_integrity",reason="未建立完整地质力学、疲劳或井筒结构认证模型")]
    write_json(Path(folder)/"engineering_assessment.json",dict(unassessed=unavailable,notes="null 阈值及缺少模型的项目均未计入综合裕度"))


def generate_report(model,summary,validation,folder):
    folder=Path(folder)
    parameter_catalog(model,folder)
    versions={k:importlib.metadata.version(k) for k in ("numpy","scipy","CoolProp","matplotlib","pandas","PyYAML")}
    write_json(folder/"runtime_versions.json",versions)
    root=folder.parent
    installed=sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions() if d.metadata["Name"].lower() not in ("pip","setuptools"))
    (root/"requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k,v in installed)+"\n",encoding="utf-8")
    if summary["status"]!="initial_charge_infeasible":
        h=summary["heat_recovery_assessment"]
        ev=summary["events"]
        n=summary["numerical_diagnostics"]
        closure=summary.get("cycle_closure",{})
        electric=summary.get("electric_loss_assessment",{})
        discharge_reason=("返回与充电相同的 CO2 质量" if ev[-1].get("kind")=="matched_charge_mass_target"
                          else ev[-1].get("reason",""))
        text=f"""# 恒温水罐方案仿真说明

## 运行结果

采用冷罐 30 °C、热罐 60 °C，二者在全过程保持恒温。固定 UA 换热器只要求沿程温差为正；原 5 K 作为设计参考保留，不再作为停机约束。

| 阶段 | 时长 h | CO2 转移质量 kg | 结束原因 |
|---|---:|---:|---|
| 充电 | {summary['duration_h']['charge']:.6f} | {summary['transferred_CO2_kg']['charge']:.3f} | 采用可逆性筛选后的充电状态 |
| 静置 | {summary['duration_h']['idle']:.6f} | 0 | 达到 2 h |
| 放电 | {summary['duration_h']['discharge']:.6f} | {summary['transferred_CO2_kg']['discharge']:.3f} | {discharge_reason} |

充电试运行首先触及 `{', '.join(ev[0]['constraints'])}`；实际充电状态说明为“{ev[1].get('reason','')}”。放电按充电转移质量反算时长。充电和放电分别转移 {closure.get('charged_CO2_kg',0):.3f} kg 与 {closure.get('discharged_CO2_kg',0):.3f} kg，质量差为 {closure.get('CO2_transfer_mismatch_kg',0):.6g} kg。电网购电 {summary['charge_purchase_MWh']:.6f} MWh，净放电 {summary['energy_MWh']['E_net']:.6f} MWh，电力边界下的单次循环效率为 {100*summary['electricity_recovery_ratio']:.3f}%。CO2 与水的转移质量已经闭合，但围岩和穴内热状态尚未达到周期稳态；初末储能差为 {summary['stored_energy_change_J']/3.6e9:.6f} MWh，因此仍需多周期收敛后才能称为严格稳态往返效率。净放电能量密度为 {summary['net_energy_density_kWh_m3']:.9f} kWh/m³。

## 盐穴容量利用

盐穴始终被 CO2 占据，因此这里的“体积利用”用循环工作质量和压力窗口利用率表征。循环工作质量占总库存 {100*summary['capacity_utilization']['working_mass_fraction_of_inventory']:.3f}%；相对于初始温度下高压穴接收能力、低压穴采出能力和最长运行时间三者中的最小参考容量，利用率为 {100*summary['capacity_utilization']['reference_working_mass_utilization']:.3f}%。充电期间高压穴初始至压力上限的压力裕度利用率为 {100*summary['capacity_utilization']['H_charge_pressure_headroom_utilization']:.3f}%，低压穴初始至压力下限的压力裕度利用率为 {100*summary['capacity_utilization']['L_charge_pressure_headroom_utilization']:.3f}%。参考质量容量采用初始穴温的等温估算，最终运行容量由瞬态约束和静置后可放电条件决定。

## 电功与损失分解

| 项目 | 能量 / MWh |
|---|---:|
| 压缩机传给 CO2 的流体功 | {electric.get('compressor_fluid_work_MWh',0):.6f} |
| 压缩机电动机/机械损失 | {electric.get('compressor_motor_mechanical_loss_MWh',0):.6f} |
| 充电水泵耗电 | {electric.get('charge_water_pump_MWh',0):.6f} |
| 其他充电辅机 | {electric.get('charge_other_auxiliary_MWh',0):.6f} |
| 膨胀机从 CO2 获得的流体功 | {electric.get('turbine_fluid_work_MWh',0):.6f} |
| 膨胀机/发电机机械电气损失 | {electric.get('turbine_generator_mechanical_loss_MWh',0):.6f} |
| 发电机毛发电 | {electric.get('generator_gross_MWh',0):.6f} |
| 放电水泵耗电 | {electric.get('discharge_water_pump_MWh',0):.6f} |
| 其他放电辅机 | {electric.get('discharge_other_auxiliary_MWh',0):.6f} |
| 净放电 | {electric.get('net_discharge_MWh',0):.6f} |

本基准删除了没有设备清单或文献标定依据的通用固定辅机，只保留显式水泵、电动机、机械传动和发电机损失。恒温水罐补热仍是独立热量账户，不计入上述电网购电。

## 热回收是否有用

| 项目 | 能量 |
|---|---:|
| 充电时冷水从 CO2 吸热 | {h['charge_CO2_to_water_MWh_th']:.6f} MWh_th |
| 充电水泵耗电 | {h['charge_water_pump_MWh_e']:.6f} MWh_e |
| “吸热减充电泵耗”指标 | {h['requested_heat_minus_pump_MWh_equivalent']:.6f} MWh（能量等值） |
| 放电再热提供给 CO2 | {h['discharge_reheat_MWh_th']:.6f} MWh_th |
| 放电水泵耗电 | {h['discharge_water_pump_MWh_e']:.6f} MWh_e |
| 全周期水泵耗电 | {h['total_water_pump_MWh_e']:.6f} MWh_e |
| “吸热减全周期泵耗”指标 | {h['charge_heat_minus_all_cycle_pumps_MWh_equivalent']:.6f} MWh（能量等值） |
| 充电阶段恒温补热 | {h['conditioning_charge_MWh_th']:.6f} MWh_th |
| 静置恒温补热 | {h['conditioning_idle_MWh_th']:.6f} MWh_th |
| 放电回水恒温处理 | {h['conditioning_discharge_MWh_th']:.6f} MWh_th（负值为排热） |

按“冷水吸热减去充电水泵耗电”的局部指标，结果为正，泵耗只占回收热的 {100*h['pump_fraction_of_recovered_heat']:.3f}%。即使把放电水泵也计入，全周期水泵耗电仅占回收热的 {100*h['all_cycle_pump_fraction_of_recovered_heat']:.3f}%，净值仍为正。因此**水循环用于吸收 CO2 热量本身是有效的**。

但恒温热罐方案不是自给的热回收闭环。充电换热器回水远低于 60 °C，维持热罐恒温需要额外补热 {h['conditioning_charge_MWh_th']:.3f} MWh_th；它大于从 CO2 回收的 {h['charge_CO2_to_water_MWh_th']:.3f} MWh_th。放电再热量也高于本次充电回收热量，差额来自预置热水、恒温补热和未闭合的末态库存。因此不能把外部补热产生的发电归功于压缩热回收。

热量与电量虽都用 MWh 表示，但温度品位不同。严谨效率或经济性还需给出恒温加热/冷却设备的能源来源、效率和耗电。本阶段只做一阶能量核算。

## 运行裕度曲线

`fig2_operating_margins.png/svg/pdf` 的横轴是从实际充电开始的累计时间，覆盖充电—静置—放电；两条竖虚线分别标记充电结束和静置结束。每条线是对应约束在每个输出时刻的归一化裕度，停用阶段断线。水平 0 线是运行边界，负值未截断。综合最小值只包含已评估且在该阶段适用的项目。

换热温差曲线相对于 0 K 温差交叉边界计算，固定显示尺度仍为 5 K。充电最小温差接近夹点但保持正值；这不代表满足原来的 5 K 设计参考。5 K 是初步设计/经济筛选值，不是热力学必要条件。

## 守恒与数值误差

| 检查 | 最大误差 |
|---|---:|
| CO2 总质量 | {n['CO2_mass_error_kg']:.6g} kg |
| 水总质量 | {n['water_mass_error_kg']:.6g} kg |
| 压力匹配 | {n['max_pressure_match_Pa']:.6g} Pa |
| 换热器能量 | {n['max_HX_energy_residual_W']:.6g} W |
| 换热器 UA 方程 | {n['max_HX_UA_residual_W_per_K']:.6g} W/K |
| 井筒能量 | {n['max_well_energy_residual_W']:.6g} W |
| 系统瞬时能量 | {n['max_system_rate_residual_W']:.6g} W |
| 全过程积分能量 | {n['integrated_energy_residual_J']:.6g} J |

"""
        if validation:
            text+=f"完整验证状态：`module_checks_passed = {validation['module_checks_passed']}`。时间步及空间分段加密的逐状态差异见 `validation.json`。\n\n"
        else:
            text+="本次未运行 `--validate` 的加密复算；上述为基准轨迹实际残差。\n\n"
        text+="""## 边界

水罐温度固定时，水量仍守恒并随流量转移。模型显式计算为使换热器回水达到接收罐固定温度所需的外部热量及环境散热补偿。泵电功、其他辅机和电机/发电机损失不反馈到水温。

井筒承压、最低温度、变化率、NPSH、完整机内相态和地质力学限制仍未评估，也不计入综合裕度。详见 `engineering_assessment.json` 和 `unevaluated_constraints.csv`。
"""
        (folder/"检查说明.md").write_text(text,encoding="utf-8")
        return
    if "hx.charge.approach" not in summary["events"][-1]["constraints"]:
        (folder/"检查说明.md").write_text("# 初始充电工作点未通过\n\n尚未执行非零时长阶段。完整原因、类别与数值见下列事件；不能据此声称物理循环通过。\n\n```json\n"+json.dumps(summary["events"],ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
        return
    df=pd.read_csv(folder/"preflight_workpoint.csv")
    r=df.iloc[0]
    n=summary["numerical_diagnostics"]
    v=validation
    rate_percent=100*summary["neglected_well_inventory_fraction_range"][0] if summary["neglected_well_inventory_fraction_range"][0] is not None else np.nan
    text=f"""# 本阶段检查说明：基础模型已运行，原始算例未能启动充电

## 实际结论

程序在 **t=0** 检查到 `hx.charge.approach` 越界。100 kg/s 候选充电工作点的实际最小温差为 **{r.hx_approach:.9f} K**，要求为 **5 K**；归一化裕度 **{(r.hx_approach-5)/5:.9f}**。压力匹配已经找到解，停止原因是运行约束越界，不是压力求根失败。

没有执行非零时长的实际充电、静置或放电，没有 90% 可行充电快照，CO2/水实际转移质量均为 0。购电、静置耗电和净放电能量均为 0；净放电能量密度为 0 kWh/m³。电能回收比因分母为 0 **未定义**，不能报告 0% 往返效率。初末动态状态相同。

`trajectory.csv` 只有未运行初态；`preflight_workpoint.csv` 保存被拒绝候选工作点的功率、流量与导数。`constraints.csv`、`constraint_minima.csv` 的活跃设备值是 **t=0 候选可行性检查**，不是实际执行记录。后者的每项最小值均发生于 t=0，不能解释为一条非零时长轨迹的最小值。

图1展示共同初态和未启动结果；图2展示未经截断的起点裕度；图3展示候选充电状态1–3，4–6留空。没有可定义的充电结束和静置结束时刻，因此未画虚假的阶段分界线或完整循环。绘图模块已包含可行轨迹的共享时间轴、多子图、阶段标注、零裕度线和不适用阶段断线逻辑。

## 为什么触发温差约束

| 候选状态 | 压力 MPa | 温度 °C |
|---|---:|---:|
| 状态1：采出井出口 | {r.point1_p/1e6:.6f} | {r.point1_T-273.15:.6f} |
| 状态2：压缩机出口 | {r.point2_p/1e6:.6f} | {r.point2_T-273.15:.6f} |
| 状态3：换热器出口 | {r.point3_p/1e6:.6f} | {r.point3_T-273.15:.6f} |

冷水入口 30 °C，水侧流量 25 kg/s，换热量 {r.Q_hx/1e6:.6f} MW。水出口为 {r.water_return_T-273.15:.6f} °C，已接近状态2温度。有限 UA 方程在此处给出接近温度夹点的解，它并不自动满足另外指定的 5 K 运行要求。候选压缩机电功率只有 {r.P_comp/1000:.4f} kW，初始两穴压差较小；回收热量不全来自压缩机电功，也包含从 CO2 初始热库存取出的热量，不能用它与压缩机电功的大小直接判断能量违规。

初始热罐水量 40000 kg 恰在下限，冷罐水量 1960000 kg 恰在容量上限，二者裕度为 0。充电方向使热罐接收、冷罐采出，因此这些零裕度没有被误判为停止事件。它们和负的换热裕度均原样保留。

## 数值验证

| 检查 | 实际误差 |
|---|---:|
| 压力匹配 | {n['max_pressure_match_Pa']:.6g} Pa |
| 换热器两侧能量 | {n['max_HX_energy_residual_W']:.6g} W |
| 离散换热方程 UA 残差 | {n['max_HX_UA_residual_W_per_K']:.6g} W/K |
| 井筒能量 | {n['max_well_energy_residual_W']:.6g} W |
| 水罐瞬时能量 | {n['max_water_energy_residual_W']:.6g} W |
| 系统瞬时能量 | {n['max_system_rate_residual_W']:.6g} W |

CO2 总质量与水总质量由互补库存构造，误差均为 0。这是构造守恒验证；不等于已经验证了未发生的循环。完整过程积分能量核算在零时长下只是恒等式，其非零时长验证标为未评估。

"""
    if v:
        hx=v["HX_independent_continuous_quadrature"]
        text+=f"""绝热无摩擦井筒沿程 `h+gz` 最大偏差为 **{v['adiabatic_frictionless_well']['max_h_plus_gz_deviation_J_kg']:.6g} J/kg**。绝热定容封闭盐穴的质量和内能变化均为 0；混合水罐解析解温度误差为 {v['mixed_receiving_tank_analytic']['temperature_error_K']:.6g} K。状态快照保存/载入误差为 0。

独立连续空间自适应积分得到换热量 {hx['Q_adaptive_quadrature_W']:.6f} W，与生产模型 20 段解的相对差为 {abs(hx['relative_difference']):.6g}。将生产模型换热量代入连续空间 UA 积分，残差为 {hx['UA_continuous_residual_W_per_K']:.6g} W/K；它是空间离散误差，与上表离散方程求根残差不同。

| 井筒段数 | 换热器段数 | 最小温差 K | 换热量 W | 压缩机电功率 W |
|---:|---:|---:|---:|---:|
"""
        for row in v["spatial_convergence_initial_workpoint"]:
            text+=f"| {row['well_segments']} | {row['hx_segments']} | {row['approach_K']:.9f} | {row['Q_W']:.6f} | {row['compressor_power_W']:.6f} |\n"
        text+=f"""
加密后起点仍不可行，负裕度结论稳定。**没有可行基准动态轨迹，因此不能给出该轨迹的时间/空间收敛结论。** 另外对隔离的静置模块进行了 30 s/15 s 最大步长的 2 h 测试：两次均到达时长上限，没有因初始水量恰在下限而立即终止。差异逐状态保存在 `validation.json`，该测试不计入实际过程及能量。

仅做了一次降低流量的有限调试：100 → {v['finite_flow_debug']['candidate_q']:.0f} kg/s，最小温差进一步减到 **{v['finite_flow_debug']['approach_K']:.9f} K**，仍越界。没有采用该值，也没有修改 UA、水初温、相区、库存或 5 K 阈值。未进行流量扫描或优化。

模块数值判据结果：**{v['module_checks_passed']}**。这不表示完整循环通过，也不表示工程安全已认证。

"""
    else:text+="本次未执行 --validate；以上只包含实际起点诊断。\n\n"
    text+=f"""## 能量边界和近似

储能量包括两穴 CO2 内能、等效围岩显热和两水罐显热。外部项包含远场围岩传热、井筒与地温换热、水罐散热、压缩/膨胀对流体的功；不同深度时计入 CO2 库存转移的重力势能变化。本算例两穴同深，净势能转移为 0。水泵、电机/发电机及机械损失、其他辅机的耗电均核算，但其热量视作排到所建热库存边界之外，**没有把这些热量反馈到水温**。

候选起点两根井筒沿程估算 CO2 库存占穴内总库存约 **{rate_percent:.6f}%**，为千分之一量级。地面设备/管道有效容积未知，因此总被忽略库存比例还不能完整评估；结果给出各地面状态密度，可按 `Σrho_j V_j / M_CO2` 补算。不能把仅井筒估算称作全部井筒和设备库存。井筒准稳态且不建立停井温度动态，静置时沿程结果不适用。

## 尚未评估的限制与下一步接口

井筒额定承压、盐穴最低温度、压力/温度变化率上限、水罐最低温度、泵 NPSH，以及膨胀机出口温度上限未给出可靠阈值；固定转速设备工作范围、完整机内相态与地质力学完整性也未验证。完整清单见 `engineering_assessment.json`，当前阶段适用但未评估条目见 `unevaluated_constraints.csv`。水泵压升和水罐绝对压力各为独立配置项。

相区筛选覆盖两穴、活跃地面端点、井筒全部离散节点及中点、换热器全部分段节点；并不证明压缩机/膨胀机整个内部流道或离散节点之间处处满足相区条件。马赫数和被忽略动能变化另作模型适用性警报，不混作工程约束越界。

所有裕度尺度在配置中固定。改变尺度会改变“谁最接近边界”的排序；综合值只涵盖已评估且适用条目，不涵盖未评估工程项目。

模块输入输出、快照恢复例子和后续接口见项目 `README.md`。下一阶段可使用 `Model.net_power(x, q_d)` 建立恒功率求根；本阶段没有实现或运行该扩展。应先由你复核初始换热条件与运行协议；当前交付保留原始参数，不代替你做设计取舍。
"""
    (folder/"检查说明.md").write_text(text,encoding="utf-8")
