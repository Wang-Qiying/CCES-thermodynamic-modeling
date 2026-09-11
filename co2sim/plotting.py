from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def make_plots(df,mf,summary,cfg,folder):
    folder = Path(folder)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[n for n in ["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"] if n in available],
                         "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
    t = df.time_h.to_numpy()
    tc = summary["duration_h"].get("charge",0)
    ti = tc+summary["duration_h"].get("idle",0)
    tend = max(t.max(),0.05)

    def decorate(axes):
        for ax in axes:
            for a,b,color,label in [(0,tc,"#EDF4FC","充电"),(tc,ti,"#F0F0F0","静置"),(ti,tend,"#FDF0E5","放电")]:
                if b>a:
                    ax.axvspan(a,b,color=color,zorder=-5)
            for tt in (tc,ti): ax.axvline(tt,color="#777777",linestyle="--",linewidth=0.8)
            ax.grid(alpha=.2)
            ax.set_xlim(0,tend)
        for a,b,label in [(0,tc,"充电"),(tc,ti,"静置"),(ti,tend,"放电")]:
            if b>a: axes[0].text((a+b)/2,1.02,label,transform=axes[0].get_xaxis_transform(),ha="center")
        axes[-1].set_xlabel("累计时间 / h（充电结束、静置结束以竖虚线标记）")

    def save(fig,name):
        fig.savefig(folder/(name+".png"),dpi=180,bbox_inches="tight")
        fig.savefig(folder/(name+".svg"),bbox_inches="tight")
        fig.savefig(folder/(name+".pdf"),bbox_inches="tight")
        plt.close(fig)

    if summary["status"] == "initial_charge_infeasible":
        preflight_plots(df,mf,summary,cfg,save)
        return

    fig,axes = plt.subplots(6,1,figsize=(12,15),sharex=True,layout="constrained")
    colors = {"H":"#1767A4","L":"#DC7434"}
    for i in ("H","L"):
        axes[0].plot(t,df[f"p_{i}"]/1e6,label=i,color=colors[i])
        for suffix in ("min","max"):
            axes[0].axhline(cfg["caverns"][i][f"p_{suffix}"]/1e6,ls=":",color=colors[i],alpha=.7)
        axes[1].plot(t,df[f"T_{i}"]-273.15,label=f"CO2 {i}",color=colors[i])
        axes[1].plot(t,df[f"T_r_{i}"]-273.15,label=f"围岩 {i}",color=colors[i],ls="--")
    axes[0].set_ylabel("穴压 / MPa")
    axes[1].set_ylabel("温度 / °C")
    axes[2].plot(t,df.q,label="CO2")
    axes[2].plot(t,df.mw,label="水")
    axes[2].set_ylabel("质量流量 / kg/s")
    for key,label in [("P_grid","电网购电"),("P_gross","发电机毛输出"),("P_net","净输出")]:
        axes[3].plot(t,df[key]/1e6,label=label)
    axes[3].set_ylabel("功率 / MW")
    axes[4].plot(t,df.M_w_hot/1e3,label="热罐")
    axes[4].plot(t,df.M_w_cold/1e3,label="冷罐")
    axes[4].axhline(cfg["water"]["mass_total"]*cfg["water"]["min_fraction"]/1e3,ls=":",color="gray",label="最低保有量")
    axes[4].set_ylabel("水量 / t")
    axes[5].plot(t,df.T_w_hot-273.15,label="热罐")
    axes[5].plot(t,df.T_w_cold-273.15,label="冷罐")
    axes[5].set_ylabel("水温 / °C")
    for ax in axes: ax.legend(loc="best",ncol=4,fontsize=8)
    decorate(axes)
    fig.suptitle("图1  闭式双盐穴 CO2 储能：实际采用的充电—静置—放电过程",fontsize=14)
    save(fig,"fig1_process")

    fig,axes = plt.subplots(5,1,figsize=(13,16),sharex=True,layout="constrained")
    important = [
        [("H.p_min","H 压力下限"),("H.p_max","H 压力上限"),("L.p_min","L 压力下限"),("L.p_max","L 压力上限")],
        [("H.T_max","H 温度上限"),("L.T_max","L 温度上限"),("phase.H.T","H 超临界温度"),("phase.L.T","L 超临界温度"),("phase.take_well.p","采出井超临界压力"),("phase.inject_well.T","注入井超临界温度")],
        [("compressor.power_max","压缩机功率"),("compressor.ratio_min","压缩机压比下限"),("compressor.ratio_max","压缩机压比上限"),("turbine.power_max","发电机功率"),("turbine.ratio_min","膨胀比下限"),("turbine.ratio_max","膨胀比上限"),("net_power.positive","正净输出")],
        [("water.hot.mass_min","热罐最低水量"),("water.cold.mass_min","冷罐最低水量"),("water.hot.T_max","热罐最高温度"),("hx.charge.approach","充电换热最小温差"),("hx.discharge.approach","放电换热最小温差")]
    ]
    for ax,entries in zip(axes,important):
        for name,label in entries:
            dat = mf[mf.name==name]
            ax.plot(dat.time_h,dat.normalized_margin,label=label,lw=1.5)
        ax.legend(ncol=3,fontsize=8,loc="best")
    for group,ls,label in [("thermal",":","穴温组最小"),("phase","--","相区组最小")]:
        axes[1].plot(t,df["margin_"+group],ls=ls,color="black",label=label,alpha=.6)
    axes[1].legend(ncol=3,fontsize=8)
    axes[4].plot(t,df.margin_all_evaluated_applicable,color="#9B273B",lw=2,label="全部已评估且适用约束的最小值")
    axes[4].legend(loc="best")
    for ax,label in zip(axes,["穴压","热状态 / 相区","设备","水蓄热 / 换热","综合最小"]):
        ax.axhline(0,color="black",lw=1)
        ax.set_ylabel(label+"\n归一化裕度 μ")
    decorate(axes)
    final_event = summary["events"][-1]
    stop = final_event["kind"]+"; "+", ".join(final_event["constraints"])
    axes[4].annotate("终止："+stop,
        xy=(t[-1],df.margin_all_evaluated_applicable.iloc[-1]),
        xytext=(-8,14),textcoords="offset points",ha="right",va="bottom",fontsize=8,
        bbox=dict(facecolor="white",alpha=.85,edgecolor="none"),
        arrowprops=dict(arrowstyle="->",color="#666666",lw=.8))
    fig.suptitle("图2  归一化运行约束裕度（不是已认证的地质安全裕度）\n尺度影响接近边界的排序；未评估项目不计入综合值；不适用阶段断线",fontsize=13)
    save(fig,"fig2_operating_margins")

    fig,axes = plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    for j in range(1,7):
        axes[0].plot(t,df[f"point{j}_p"]/1e6,label=f"状态 {j}")
        axes[1].plot(t,df[f"point{j}_T"]-273.15,label=f"状态 {j}")
    axes[0].set_ylabel("压力 / MPa")
    axes[1].set_ylabel("温度 / °C")
    for ax in axes: ax.legend(ncol=6,fontsize=8)
    decorate(axes)
    fig.suptitle("图3  地面状态点 1–6（停用时留空）")
    save(fig,"fig3_surface_states")
    # Cavern states and all six surface states on the same cumulative timeline.
    # Inactive surface points remain NaN, so charge/idle/discharge applicability
    # is visible directly without inventing values during idle.
    fig,axes = plt.subplots(2,1,figsize=(13,8),sharex=True,layout="constrained")
    axes[0].plot(t,df.p_H/1e6,color="#1767A4",lw=2.4,label="高压盐穴 H")
    axes[0].plot(t,df.p_L/1e6,color="#DC7434",lw=2.4,label="低压盐穴 L")
    axes[1].plot(t,df.T_H-273.15,color="#1767A4",lw=2.4,label="高压盐穴 H")
    axes[1].plot(t,df.T_L-273.15,color="#DC7434",lw=2.4,label="低压盐穴 L")
    point_labels = {
        1:"点1 采出井口/压缩机前", 2:"点2 压缩机后", 3:"点3 充电换热器后",
        4:"点4 采出井口/再热前", 5:"点5 再热器后", 6:"点6 膨胀机后"}
    point_colors = ["#2A9D8F","#6A4C93","#8AB17D","#E76F51","#F4A261","#7F5539"]
    for j,color in zip(range(1,7),point_colors):
        axes[0].plot(t,df[f"point{j}_p"]/1e6,color=color,ls="--",lw=1.6,label=point_labels[j])
        axes[1].plot(t,df[f"point{j}_T"]-273.15,color=color,ls="--",lw=1.6,label=point_labels[j])
    # Put the physical/operating envelope on the same axes as the state curves.
    # Cavern limits use the cavern colour; the supercritical envelope is shared
    # by every CO2 state, including the well and heat-exchanger profiles.
    for i,color in (("H","#1767A4"),("L","#DC7434")):
        cc=cfg["caverns"][i]
        axes[0].axhline(cc["p_min"]/1e6,color=color,ls=":",lw=1.1,
                       label=f"{i} 穴压下限")
        axes[0].axhline(cc["p_max"]/1e6,color=color,ls="-.",lw=1.1,
                       label=f"{i} 穴压上限")
    p_phase=(summary["critical"]["p"]+cfg["phase"]["pressure_above_critical"])/1e6
    T_phase=summary["critical"]["T"]+cfg["phase"]["temperature_above_critical"]-273.15
    axes[0].axhline(p_phase,color="#222222",ls="--",lw=1.2,label="超临界压力下限")
    axes[1].axhline(T_phase,color="#222222",ls="--",lw=1.2,label="超临界温度下限")
    shown_Tmax=[]
    for i,color in (("H","#1767A4"),("L","#DC7434")):
        Tmax=cfg["caverns"][i]["T_max"]
        if Tmax is not None and Tmax not in shown_Tmax:
            axes[1].axhline(Tmax-273.15,color=color,ls="-.",lw=1.1,
                           label=f"{i} 穴温上限")
            shown_Tmax.append(Tmax)
    axes[0].set_ylabel("CO2 压力 / MPa")
    axes[1].set_ylabel("CO2 温度 / °C")
    for ax in axes:
        ax.legend(ncol=4,fontsize=8,loc="best")
    decorate(axes)
    fig.suptitle("图4  两个盐穴与地面点位 1–6 的 CO2 压力、温度全过程变化\n地面点位在对应设备停用时留空",fontsize=13)
    save(fig,"fig4_all_key_states")


def preflight_plots(df,mf,summary,cfg,save):
    """An infeasible start must not be drawn as a fictitious three-stage cycle."""
    r = df.iloc[0]
    fig,ax = plt.subplots(2,3,figsize=(13,8),layout="constrained")
    a = ax.ravel()
    a[0].bar(["H","L"],[r.p_H/1e6,r.p_L/1e6],color=["#1767A4","#DC7434"])
    for j,i in enumerate(("H","L")):
        a[0].plot([j-.3,j+.3],[cfg["caverns"][i]["p_min"]/1e6]*2,"k--")
        a[0].plot([j-.3,j+.3],[cfg["caverns"][i]["p_max"]/1e6]*2,"k:")
    a[0].set_ylabel("穴压 / MPa"); a[0].set_title("共同初态；虚线/点线为上下限")
    a[1].bar(["CO2 H","CO2 L","围岩 H","围岩 L"],[r.T_H-273.15,r.T_L-273.15,r.T_r_H-273.15,r.T_r_L-273.15])
    a[1].set_ylabel("温度 / °C")
    a[2].bar(["实际 CO2","实际水"],[0,0]); a[2].set_ylim(0,120)
    a[2].set_ylabel("实际流量 / kg/s")
    a[2].text(.5,.6,"未运行：实际转移量为 0\n候选充电流量 100 kg/s",ha="center",transform=a[2].transAxes)
    a[3].bar(["电网购电","毛发电","净输出"],[0,0,0]); a[3].set_ylim(0,.1)
    a[3].set_ylabel("实际功率 / MW")
    a[3].text(.5,.6,"未启动，购电与放电量均为 0\n候选工作点功率见诊断数据",ha="center",transform=a[3].transAxes)
    a[4].bar(["热罐","冷罐"],[r.M_w_hot/1e3,r.M_w_cold/1e3],color=["#DC7434","#1767A4"])
    a[4].set_ylabel("初始水量 / t")
    a[5].bar(["热罐","冷罐"],[r.T_w_hot-273.15,r.T_w_cold-273.15],color=["#DC7434","#1767A4"])
    a[5].set_ylabel("初始水温 / °C")
    for axis in a: axis.grid(axis="y",alpha=.2)
    fig.suptitle("图1  初态与未启动结果：充电起点不满足运行条件\n充电、静置、放电时长均为 0；没有可绘制的实际过程曲线",fontsize=14)
    save(fig,"fig1_process")

    groups = [
        ("穴压",[("H.p_min","H 压力下限"),("H.p_max","H 压力上限"),("L.p_min","L 压力下限"),("L.p_max","L 压力上限")]),
        ("热状态与相区",[("H.T_max","H 温度上限"),("L.T_max","L 温度上限"),("phase.take_well.p","采出井相区压力"),("phase.inject_well.T","注入井相区温度")]),
        ("候选充电设备",[("compressor.q_min","流量下限"),("compressor.ratio_min","压比下限"),("compressor.power_max","功率上限"),("compressor.T_out_max","排气温度上限")]),
        ("水蓄热与换热",[("water.hot.mass_min","热罐最低水量"),("water.cold.capacity","冷罐容量上限"),("water.hot.T_max","热水温度上限"),("hx.charge.approach","换热最小温差")])]
    fig,axes = plt.subplots(5,1,figsize=(12,13),layout="constrained")
    for ax,(title,entries) in zip(axes,groups):
        names,vals = [],[]
        for name,label in entries:
            rr = mf[mf.name==name].iloc[0]
            names.append(label); vals.append(rr.normalized_margin)
        bars = ax.barh(names,vals,color=["#BF3943" if v<0 else "#367EAA" for v in vals],height=.6)
        for b,v in zip(bars,vals):
            if np.isfinite(v):
                ax.annotate(f" {v:.4f}",(v,b.get_y()+b.get_height()/2),va="center",ha="left" if v>=0 else "right",fontsize=9)
            else:
                ax.text(0,b.get_y()+b.get_height()/2,"不适用 / 未评估",va="center")
        ax.axvline(0,color="black",lw=1); ax.set_title(title,loc="left"); ax.grid(axis="x",alpha=.2)
        finite=[v for v in vals if np.isfinite(v)]
        lo=min(0,min(finite)) if finite else -1
        hi=max(finite) if finite else 1
        ax.set_xlim(lo-0.2*(hi-lo+1),hi+0.2*(hi-lo+1))
        ax.set_xlabel("归一化运行约束裕度 μ（原始值，未截断）")
    low = mf[mf.active & mf.evaluated].normalized_margin.min()
    axes[4].axis("off")
    ev=summary["events"][-1]
    diagnosis=(f"实际最小温差 {r.hx_approach:.6f} K；要求至少 5 K。" if np.isfinite(r.hx_approach) else "工作点求解失败，活跃设备裕度未评估。")
    axes[4].text(0,1,f"候选起点综合最小裕度：{low:.6f}；停止类别 {ev['kind']}\n"
        +diagnosis+"触发："+", ".join(ev["constraints"])+"\n"
        "初始热罐最低水量、冷罐容量裕度恰为 0，允许向可行方向转移；它们不是停止原因。\n"
        "综合值仅含已评估且适用项目；停用设备不适用；未评估工程阈值不计入。\n"
        "归一化尺度会影响接近边界的排序。这不是已认证的地质安全裕度。\n"
        "没有充电结束/静置结束时刻，故不绘制虚假的阶段分界线或时间曲线。",va="top",fontsize=11,linespacing=1.8)
    fig.suptitle("图2  t = 0 候选充电工作点：归一化运行约束裕度\n全程未启动；以初始裕度诊断替代不存在的全程曲线",fontsize=14)
    save(fig,"fig2_operating_margins")

    fig,axes = plt.subplots(2,1,figsize=(10,7),layout="constrained")
    for j in range(1,7):
        axes[0].scatter(j,r[f"point{j}_p"]/1e6,color="#1767A4")
        axes[1].scatter(j,r[f"point{j}_T"]-273.15,color="#DC7434")
    for ax in axes:
        ax.set_xticks(range(1,7),[f"状态 {j}" for j in range(1,7)]); ax.grid(alpha=.2)
        ax.text(.7,.6,"4–6 不适用：未进入放电",transform=ax.transAxes)
    axes[0].set_ylabel("压力 / MPa"); axes[1].set_ylabel("温度 / °C")
    fig.suptitle("图3  候选充电起点的地面状态诊断（尚未运行）")
    save(fig,"fig3_surface_states")

