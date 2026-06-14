import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================================
#  第一部分：计量数据的过程能力分析
# =========================================================================
def calculate_continuous_normal_matrix(df_data, usl, lsl, target):
    """直接解析横向多列展开的数据矩阵（第一列为批次，后面为测量数据）"""
    matrix_data = df_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()

    flat_data = matrix_data[~np.isnan(matrix_data)]
    if len(flat_data) < 2:
        raise ValueError("有效测量数据太少，请检查是否输入了数字。")

    grand_mean = np.mean(flat_data)
    sigma_overall = np.std(flat_data, ddof=1)
    n_total = len(flat_data)

    ranges = []
    d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

    d2_list = []
    for row in matrix_data:
        valid_row = row[~np.isnan(row)]
        n_i = len(valid_row)
        if n_i >= 2:
            ranges.append(np.max(valid_row) - np.min(valid_row))
            d2_list.append(d2_table.get(n_i, np.sqrt(n_i)))

    if not ranges:
        raise ValueError("没有检测到有效的子组极差，请确保每行至少有2个以上的测量数据。")

    sigma_within = np.mean(ranges) / np.mean(d2_list)

    pp = (usl - lsl) / (6 * sigma_overall) if sigma_overall > 0 else 0
    ppl = (grand_mean - lsl) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppu = (usl - grand_mean) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppk = min(ppl, ppu)

    cp = (usl - lsl) / (6 * sigma_within) if sigma_within > 0 else 0
    cpl = (grand_mean - lsl) / (3 * sigma_within) if sigma_within > 0 else 0
    cpu = (usl - grand_mean) / (3 * sigma_within) if sigma_within > 0 else 0
    cpk = min(cpl, cpu)

    sigma_cpm = np.sqrt(np.std(flat_data, ddof=0) ** 2 + (grand_mean - target) ** 2)
    cpm = (usl - lsl) / (6 * sigma_cpm) if sigma_cpm > 0 else 0

    ad_res = stats.anderson(flat_data, dist='norm')
    x_sorted = np.sort(flat_data)
    z_sorted = (x_sorted - grand_mean) / sigma_overall
    s = np.sum(
        [(2 * i + 1) * (np.log(stats.norm.cdf(z_sorted[i])) + np.log(1 - stats.norm.cdf(z_sorted[n_total - 1 - i]))) for
         i in range(n_total)])
    ad_stat = -n_total - s / n_total
    ad_aa = ad_stat * (1 + 0.75 / n_total + 2.25 / (n_total ** 2))
    if ad_aa >= 0.60:
        p_val = np.exp(1.2937 - 5.709 * ad_aa + 0.0186 * (ad_aa ** 2))
    elif ad_aa >= 0.34:
        p_val = np.exp(0.9177 - 4.279 * ad_aa - 1.38 * (ad_aa ** 2))
    elif ad_aa > 0.20:
        p_val = 1 - np.exp(-8.318 + 42.79 * ad_aa - 56.1 * (ad_aa ** 2))
    else:
        p_val = 1 - np.exp(-13.43 + 101.14 * ad_aa - 223 * (ad_aa ** 2))

    ppm_obs_lsl = (np.sum(flat_data < lsl) / n_total) * 1e6
    ppm_obs_usl = (np.sum(flat_data > usl) / n_total) * 1e6
    ppm_obs_total = ppm_obs_lsl + ppm_obs_usl

    ppm_exp_overall_lsl = stats.norm.cdf(lsl, grand_mean, sigma_overall) * 1e6
    ppm_exp_overall_usl = (1 - stats.norm.cdf(usl, grand_mean, sigma_overall)) * 1e6
    ppm_exp_overall_total = ppm_exp_overall_lsl + ppm_exp_overall_usl

    ppm_exp_within_lsl = stats.norm.cdf(lsl, grand_mean, sigma_within) * 1e6
    ppm_exp_within_usl = (1 - stats.norm.cdf(usl, grand_mean, sigma_within)) * 1e6
    ppm_exp_within_total = ppm_exp_within_lsl + ppm_exp_within_usl

    return {
        "Mean": grand_mean, "N": n_total, "Subgroups": len(ranges),
        "Sigma_Within": sigma_within, "Sigma_Overall": sigma_overall,
        "AD_Stat": ad_stat, "P_Value": p_val,
        "Cp": cp, "Cpl": cpl, "Cpu": cpu, "Cpk": cpk, "Cpm": cpm,
        "Pp": pp, "Ppl": ppl, "Ppu": ppu, "Ppk": ppk,
        "ppm_obs_lsl": ppm_obs_lsl, "ppm_obs_usl": ppm_obs_usl, "ppm_obs_total": ppm_obs_total,
        "ppm_exp_overall_lsl": ppm_exp_overall_lsl, "ppm_exp_overall_usl": ppm_exp_overall_usl,
        "ppm_exp_overall_total": ppm_exp_overall_total,
        "ppm_exp_within_lsl": ppm_exp_within_lsl, "ppm_exp_within_usl": ppm_exp_within_usl,
        "ppm_exp_within_total": ppm_exp_within_total,
        "flat_data": flat_data
    }


# =========================================================================
#  第二部分：计数数据的过程能力分析
# =========================================================================
def calculate_attribute_binomial(defects_list, sizes_list):
    """计数数据过程能力分析"""
    d_arr = np.array(defects_list, dtype=float)
    n_arr = np.array(sizes_list, dtype=float)

    # 过滤掉有缺失值的行
    valid_mask = (~np.isnan(d_arr)) & (~np.isnan(n_arr))
    d_arr, n_arr = d_arr[valid_mask], n_arr[valid_mask]

    if len(d_arr) == 0:
        raise ValueError("请至少输入一组有效的缺陷数和样本数量。")
    if np.any(d_arr > n_arr):
        raise ValueError("不合常理：发现有批次的缺陷数竟然大于了抽样总数，请修正。")

    total_defects = np.sum(d_arr)
    total_samples = np.sum(n_arr)
    p_bar = total_defects / total_samples if total_samples > 0 else 0

    # 1. 基础控制图参数
    p_actual = d_arr / n_arr
    ucl_arr = p_bar + 3 * np.sqrt((p_bar * (1 - p_bar)) / n_arr)
    lcl_arr = np.clip(p_bar - 3 * np.sqrt((p_bar * (1 - p_bar)) / n_arr), 0, 1)

    # 2. 计算累积缺陷率曲线趋势
    cum_defects = np.cumsum(d_arr)
    cum_samples = np.cumsum(n_arr)
    cum_p = (cum_defects / cum_samples) * 100  # 转换为 %缺陷

    # 3. 像素级还原 Minitab 的 95% 置信区间 (使用精确二项公式 Clopper-Pearson)
    if total_samples > 0:
        p_lcl = stats.beta.ppf(0.025, total_defects, total_samples - total_defects + 1) if total_defects > 0 else 0
        p_ucl = stats.beta.ppf(0.975, total_defects + 1,
                               total_samples - total_defects) if total_defects < total_samples else 1
    else:
        p_lcl, p_ucl = 0, 0

    # 4. 过程 Z 值及置信区间
    z_score = stats.norm.ppf(1 - p_bar) + 1.5 if 0 < p_bar < 1 else (6.0 if p_bar == 0 else 0.0)
    z_lcl = stats.norm.ppf(1 - p_ucl) + 1.5 if 0 < p_ucl < 1 else 0
    z_ucl = stats.norm.ppf(1 - p_lcl) + 1.5 if 0 < p_lcl < 1 else 6.0

    return {
        "total_defects": int(total_defects),
        "total_samples": int(total_samples),
        "p_bar": p_bar,
        "p_lcl": p_lcl,
        "p_ucl": p_ucl,
        "ppm": p_bar * 1e6,
        "ppm_lcl": p_lcl * 1e6,
        "ppm_ucl": p_ucl * 1e6,
        "z_score": z_score,
        "z_lcl": z_lcl,
        "z_ucl": z_ucl,
        "p_actual": p_actual,
        "ucl": ucl_arr,
        "lcl": lcl_arr,
        "cum_p": cum_p,
        "d_arr": d_arr
    }


# =========================================================================
# 第三部分：非正态数据的过程能力分析
# =========================================================================
def calculate_non_normal_boxcox(df_data, lambda_val, usl, lsl, target):
    """全数据空间 Box-Cox 变换能力分析 (支持单侧，安全解耦)"""
    subgroup_ids = df_data.iloc[:, 0].to_numpy()
    raw_data = df_data.iloc[:, 1].apply(pd.to_numeric, errors='coerce').to_numpy()

    valid_mask = ~np.isnan(raw_data)
    raw_data = raw_data[valid_mask]
    subgroup_ids = subgroup_ids[valid_mask]

    if len(raw_data) < 5:
        raise ValueError("非正态分析至少需要输入 5 个以上的测量原始数据。")
    if np.any(raw_data <= 0):
        raise ValueError("算法限制：Box-Cox 变换要求输入的数据流必须全部严格大于 0。")

    if abs(lambda_val) < 1e-5:
        trans_data = np.log(raw_data)
        trans_usl = np.log(usl) if usl is not None else None
        trans_lsl = np.log(lsl) if lsl is not None else None
        trans_target = np.log(target) if target is not None else None
    else:
        trans_data = (raw_data ** lambda_val - 1) / lambda_val
        trans_usl = (usl ** lambda_val - 1) / lambda_val if usl is not None else None
        trans_lsl = (lsl ** lambda_val - 1) / lambda_val if lsl is not None else None
        trans_target = (target ** lambda_val - 1) / lambda_val if target is not None else None

    t_mean = np.mean(trans_data)
    t_sigma_overall = np.std(trans_data, ddof=1)

    raw_mean = np.mean(raw_data)
    raw_sigma_overall = np.std(raw_data, ddof=1)

    unique_subs = np.unique(subgroup_ids)
    ranges = []
    d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    d2_list = []

    for sub in unique_subs:
        sub_data = trans_data[subgroup_ids == sub]
        n_i = len(sub_data)
        if n_i >= 2:
            ranges.append(np.max(sub_data) - np.min(sub_data))
            d2_list.append(d2_table.get(n_i, np.sqrt(n_i)))

    if ranges:
        t_sigma_within = np.mean(ranges) / np.mean(d2_list)
        raw_sigma_within = np.std(raw_data, ddof=1) * (t_sigma_within / t_sigma_overall)
    else:
        t_sigma_within = t_sigma_overall
        raw_sigma_within = raw_sigma_overall

    # 整体能力
    if trans_usl is not None and trans_lsl is not None:
        pp = (trans_usl - trans_lsl) / (6 * t_sigma_overall)
        ppl = (t_mean - trans_lsl) / (3 * t_sigma_overall)
        ppu = (trans_usl - t_mean) / (3 * t_sigma_overall)
        ppk = min(ppl, ppu)
    elif trans_usl is not None:
        pp, ppl = None, None
        ppu = (trans_usl - t_mean) / (3 * t_sigma_overall)
        ppk = ppu
    else:
        pp, ppu = None, None
        pickle = (t_mean - trans_lsl) / (3 * t_sigma_overall)
        ppk = pickle

    # 潜在组内能力
    if trans_usl is not None and trans_lsl is not None:
        cp = (trans_usl - trans_lsl) / (6 * t_sigma_within)
        cpl = (t_mean - trans_lsl) / (3 * t_sigma_within)
        cpu = (trans_usl - t_mean) / (3 * t_sigma_within)
        cpk = min(cpl, cpu)
    elif trans_usl is not None:
        cp, cpl = None, None
        cpu = (trans_usl - t_mean) / (3 * t_sigma_within)
        cpk = cpu
    else:
        cp, cpu = None, None
        cpl = (t_mean - trans_lsl) / (3 * t_sigma_within)
        cpk = cpl

    if trans_target is not None and trans_usl is not None and trans_lsl is not None:
        sigma_cpm = np.sqrt(np.std(trans_data, ddof=0) ** 2 + (t_mean - trans_target) ** 2)
        cpm = (trans_usl - trans_lsl) / (6 * sigma_cpm)
    else:
        cpm = None

    ppm_obs_lsl = (np.sum(raw_data < lsl) / len(raw_data)) * 1e6 if lsl is not None else 0
    ppm_obs_usl = (np.sum(raw_data > usl) / len(raw_data)) * 1e6 if usl is not None else 0
    ppm_obs_total = ppm_obs_lsl + ppm_obs_usl

    ppm_exp_overall_lsl = stats.norm.cdf(trans_lsl, t_mean, t_sigma_overall) * 1e6 if trans_lsl is not None else 0
    ppm_exp_overall_usl = (1 - stats.norm.cdf(trans_usl, t_mean, t_sigma_overall)) * 1e6 if trans_usl is not None else 0
    ppm_exp_overall_total = ppm_exp_overall_lsl + ppm_exp_overall_usl

    ppm_exp_within_lsl = stats.norm.cdf(trans_lsl, t_mean, t_sigma_within) * 1e6 if trans_lsl is not None else 0
    ppm_exp_within_usl = (1 - stats.norm.cdf(trans_usl, t_mean, t_sigma_within)) * 1e6 if trans_usl is not None else 0
    ppm_exp_within_total = ppm_exp_within_lsl + ppm_exp_within_usl

    return {
        "N": len(raw_data), "raw_data": raw_data, "trans_data": trans_data,
        "raw_mean": raw_mean, "raw_sigma_overall": raw_sigma_overall, "raw_sigma_within": raw_sigma_within,
        "t_mean": t_mean, "t_sigma_overall": t_sigma_overall, "t_sigma_within": t_sigma_within,
        "t_usl": trans_usl, "t_lsl": trans_lsl, "t_target": trans_target,
        "Cp": cp, "Cpl": cpl, "Cpu": cpu, "Cpk": cpk, "Cpm": cpm,
        "Pp": pp, "Ppl": ppl, "Ppu": ppu, "Ppk": ppk,
        "ppm_obs_lsl": ppm_obs_lsl, "ppm_obs_usl": ppm_obs_usl, "ppm_obs_total": ppm_obs_total,
        "ppm_exp_overall_lsl": ppm_exp_overall_lsl, "ppm_exp_overall_usl": ppm_exp_overall_usl,
        "ppm_exp_overall_total": ppm_exp_overall_total,
        "ppm_exp_within_lsl": ppm_exp_within_lsl, "ppm_exp_within_usl": ppm_exp_within_usl,
        "ppm_exp_within_total": ppm_exp_within_total,
    }


# =========================================================================
# 前端 UI 渲染中心
# =========================================================================
st.set_page_config(page_title="SPC 过程能力分析平台", layout="wide")
st.title("🏭 SPC 过程能力分析工作台")
st.markdown("---")

st.sidebar.header("🛠️ 菜单控制区")
menu = st.sidebar.radio(
    "功能模块切换:",
    ["1. 计量数据能力分析", "2. 计数数据能力分析", "3. 非正态数据能力分析"]
)

if "1." in menu:
    st.sidebar.subheader("📋 计量正态配置")
    usl = st.sidebar.number_input("规格上限 (USL):", value=11.00, format="%.3f")
    lsl = st.sidebar.number_input("规格下限 (LSL):", value=10.90, format="%.3f")
    target = st.sidebar.number_input("目标值 (Target):", value=10.95, format="%.3f")

    st.subheader("📊 计量数据工作表 ")
    st.caption("💡 提示：第一列填批次，后面多列直接填入或粘贴您的直径数据（每一行代表一个子组）。")

    df_template = pd.DataFrame({
        "批次": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "直径 1": [10.948, 10.913, 10.973, 10.923, 11.020, 10.920, 10.983, 10.959, 10.939, 10.910, 10.937, 10.968,
                   10.984, 10.925, 11.026],
        "直径 2": [10.903, 10.969, 10.909, 10.940, 10.959, 10.942, 10.912, 10.931, 10.926, 10.949, 10.942, 10.951,
                   10.954, 10.970, 10.978],
        "直径 3": [10.947, 10.949, 10.940, 10.949, 10.920, 10.933, 10.959, 10.932, 10.970, 10.928, 10.985, 10.925,
                   10.947, 10.940, 10.970],
        "直径 4": [10.962, 10.980, 10.954, 10.949, 10.983, 10.981, 10.901, 10.932, 10.964, 10.956, 10.944, 10.920,
                   10.948, 10.919, 10.958],
        "直径 5": [10.984, 10.939, 10.929, 10.934, 10.996, 10.946, 10.929, 10.959, 10.953, 10.924, 10.966, 10.981,
                   10.956, 10.928, 10.925]
    })

elif "2." in menu:
    st.subheader("🔢 计数数据工作表")
    st.caption(
        "💡 提示：请输入各批次的『不合格品数』和『抽样总数』。下方将一览呈现包含 P控制图、二项图、累积缺陷图、直方图在内的四合一诊断看板。")
    df_template = pd.DataFrame({
        "不合格品数 (Defects)": [3, 4, 5, 4, 4, 2, 3, 5, 5, 4, 6, 4, 8, 2, 5, 7, 4, 4, 2, 3, 3, 8, 7, 5, 5, 5, 5, 5, 4,
                                 3] + [None] * 20,
        "样本数量 (Sample Size)": [100] * 30 + [None] * 20
    })

else:
    st.sidebar.subheader("📐 Box-Cox 变换选项")
    lambda_val = st.sidebar.number_input("指定 Lambda (λ):", value=0.00, format="%.2f")

    has_lsl = st.sidebar.checkbox("启用下限 (LSL)", value=False)
    lsl_val = st.sidebar.number_input("规格下限 (LSL):", value=1.00, format="%.2f") if has_lsl else None
    usl_val = st.sidebar.number_input("规格上限 (USL):", value=24.00, format="%.2f")

    has_target = st.sidebar.checkbox("启用目标值 (Target)", value=True)
    target_val = st.sidebar.number_input("目标值 (Target):", value=10.95, format="%.2f") if has_target else None

    st.subheader("📐 非正态数据工作表 (第一列填子组批次，第二列填测量值)")

    np.random.seed(42)
    sim_raw = np.random.lognormal(mean=2.3, sigma=0.45, size=125)
    sim_raw = np.clip(sim_raw, 1.1, 23.5)
    batches = np.repeat(np.arange(1, 26), 5)

    df_template = pd.DataFrame({
        "子组批次 (Subgroup)": batches,
        "杂质含量数据 (Raw Data)": np.round(sim_raw, 4)
    })

edited_df = st.data_editor(df_template, num_rows="dynamic", use_container_width=True, height=250)
run_btn = st.button("🚀 运行当前工作表", type="primary", use_container_width=True)

if run_btn:
    cleaned_df = edited_df.dropna(how="all")
    if len(cleaned_df) == 0:
        st.error("❌ 错误：工作表中未填写任何数字，请录入数据后再试！")
        st.stop()

    try:
        if "1." in menu:
            res = calculate_continuous_normal_matrix(cleaned_df, usl, lsl, target)
            st.success("🎉 多列数据正态分析报告渲染成功！")

            x_plot = np.linspace(res["Mean"] - 4 * res["Sigma_Overall"], res["Mean"] + 4 * res["Sigma_Overall"], 200)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=res["flat_data"], histnorm='probability density', name='数据频数分布',
                                       marker_color='#B0C4DE', opacity=0.85))
            fig.add_trace(go.Scatter(x=x_plot, y=stats.norm.pdf(x_plot, res["Mean"], res["Sigma_Within"]), mode='lines',
                                     name='组内 (Within)', line=dict(color='red', dash='dash', width=2)))
            fig.add_trace(
                go.Scatter(x=x_plot, y=stats.norm.pdf(x_plot, res["Mean"], res["Sigma_Overall"]), mode='lines',
                           name='整体 (Overall)', line=dict(color='blue', width=2)))
            fig.add_vline(x=lsl, line_color="red", line_width=1.5, line_dash="dash", annotation_text="LSL")
            fig.add_vline(x=target, line_color="green", line_dash="dot", annotation_text="Target")
            fig.add_vline(x=usl, line_color="red", line_width=1.5, line_dash="dash", annotation_text="USL")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              title_text="📊 规格上限/下限双重正态拟合分布图", title_x=0.45)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                st.markdown("##### 📝 基础过程数据与正态检验")
                st.table(pd.DataFrame({
                    "过程统计项": ["规格下限 (LSL)", "目标值 (Target)", "规格上限 (USL)", "样本均值 (Mean)",
                                   "样本总数 (N)", "AD 正态检验统计量", "正态判定 P 值 (P-Value)"],
                    "数值": [f"{lsl:.3f}", f"{target:.3f}", f"{usl:.3f}", f"{res['Mean']:.5f}", f"{res['N']}",
                             f"{res['AD_Stat']:.3f}", f"{res['P_Value']:.3f}"]
                }))
            with c2:
                st.markdown("##### 🎯 潜在组内 / 长期整体能力")
                st.table(pd.DataFrame({
                    "组内指标 (Within)": [f"Cp: {res['Cp']:.2f}", f"CPL: {res['Cpl']:.2f}", f"CPU: {res['Cpu']:.2f}",
                                          f"Cpk: {res['Cpk']:.2f}", f"标准差 σ: {res['Sigma_Within']:.6f}"],
                    "整体指标 (Overall)": [f"Pp: {res['Pp']:.2f}", f"PPL: {res['Ppl']:.2f}", f"PPU: {res['Ppu']:.2f}",
                                           f"Ppk: {res['Ppk']:.2f}", f"标准差 σ: {res['Sigma_Overall']:.6f}"]
                }))
            with c3:
                st.markdown("##### 📉 预期 PPM 缺陷率外推表现")
                st.table(pd.DataFrame({
                    "性能评估项 (PPM)": ["低于下限 (< LSL)", "高于上限 (> USL)", "合计不合格数 (Total)"],
                    "实际观测": [f"{res['ppm_obs_lsl']:.2f}", f"{res['ppm_obs_usl']:.2f}",
                                 f"{res['ppm_obs_total']:.2f}"],
                    "预期整体 (Overall)": [f"{res['ppm_exp_overall_lsl']:.2f}", f"{res['ppm_exp_overall_usl']:.2f}",
                                           f"{res['ppm_exp_overall_total']:.2f}"],
                    "预期组内 (Within)": [f"{res['ppm_exp_within_lsl']:.2f}", f"{res['ppm_exp_within_usl']:.2f}",
                                          f"{res['ppm_exp_within_total']:.2f}"]
                }))

        elif "2." in menu:
            d_vec = cleaned_df.iloc[:, 0].dropna().to_numpy().astype(float)
            n_vec = cleaned_df.iloc[:, 1].dropna().to_numpy().astype(float)

            res = calculate_attribute_binomial(d_vec, n_vec)
            st.success("🎉 二项能力诊断报告计算成功！")

            fig_matrix = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📊 P 控制图 (P Chart)',
                    '📈 二项分布概率图 (Binomial Plot)',
                    '📉 累积 %缺陷率趋势图 (Cumulative %Defective)',
                    '🧱 各批次缺陷率直方图 (Histogram)'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            x_axis = np.arange(1, len(res["p_actual"]) + 1)

            fig_matrix.add_trace(
                go.Scatter(x=x_axis, y=res["p_actual"], mode='lines+markers', name='批次比率', marker_color='#004B87'),
                row=1, col=1)
            fig_matrix.add_trace(
                go.Scatter(x=x_axis, y=res["ucl"], mode='lines', name='UCL', line=dict(color='red', dash='dash')),
                row=1, col=1)
            fig_matrix.add_trace(
                go.Scatter(x=x_axis, y=res["lcl"], mode='lines', name='LCL', line=dict(color='red', dash='dash')),
                row=1, col=1)
            fig_matrix.add_hline(y=res["p_bar"], line_color="green", row=1, col=1)

            sorted_d = np.sort(res["d_arr"])
            fig_matrix.add_trace(
                go.Scatter(x=sorted_d, y=sorted_d, mode='markers', name='实测观测值', marker_color='#004B87'), row=1,
                col=2)
            fig_matrix.add_trace(
                go.Scatter(x=sorted_d, y=sorted_d, mode='lines', name='拟合线', line=dict(color='red')), row=1, col=2)

            fig_matrix.add_trace(
                go.Scatter(x=x_axis, y=res["cum_p"], mode='lines+markers', name='累积 %缺陷', marker_color='#004B87'),
                row=2, col=1)
            fig_matrix.add_hline(y=res["p_bar"] * 100, line_color="grey", line_dash="dash", row=2, col=1)

            fig_matrix.add_trace(go.Histogram(x=res["p_actual"] * 100, name='缺陷率分布', marker_color='#B0C4DE'),
                                 row=2, col=2)

            fig_matrix.update_layout(height=650, showlegend=False,
                                     title_text="🧬 不合格品数量的二项过程能力报告诊断看板", title_x=0.5)
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.markdown("### 📋汇总统计能力看板 (95.0% 置信区间)")
            col_stats_1, col_stats_2, col_stats_3 = st.columns(3)

            with col_stats_1:
                st.info("📊 基础过程指标")
                st.write(f"**%缺陷率 (% Defective):** {res['p_bar'] * 100:.2f}%")
                st.write(f"  - 95% 置信区间下限: {res['p_lcl'] * 100:.2f}%")
                st.write(f"  - 95% 置信区间上限: {res['p_ucl'] * 100:.2f}%")

            with col_stats_2:
                st.success("🎯 PPM 过程水平评估")
                st.write(f"**PPM 缺陷率:** {res['ppm']:.0f}")
                st.write(f"  - 95% 置信区间下限: {res['ppm_lcl']:.0f}")
                st.write(f"  - 95% 置信区间上限: {res['ppm_ucl']:.0f}")

            with col_stats_3:
                st.warning("⚡ 过程能力 Z 阶水平")
                st.write(f"**过程 Z 值 (Z-Score):** {res['z_score']:.4f}")
                st.write(f"  - 95% 置信区间下限: {res['z_lcl']:.4f}")
                st.write(f"  - 95% 置信区间上限: {res['z_ucl']:.4f}")

        else:
            res = calculate_non_normal_boxcox(cleaned_df, lambda_val, usl_val, lsl_val, target_val)
            st.success(f"🎉 杂质偏态数据 Box-Cox (λ = {lambda_val:.2f}) 变换及过程能力报告外推成功！")

            x_plot = np.linspace(res["t_mean"] - 4 * res["t_sigma_overall"], res["t_mean"] + 4 * res["t_sigma_overall"],
                                 200)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=res["trans_data"], histnorm='probability density', name='已变换数据分布',
                                       marker_color='#6ca0dc', opacity=0.85))
            fig.add_trace(
                go.Scatter(x=x_plot, y=stats.norm.pdf(x_plot, res["t_mean"], res["t_sigma_within"]), mode='lines',
                           name='组内 (Within)', line=dict(color='black', dash='dash', width=1.5)))
            fig.add_trace(
                go.Scatter(x=x_plot, y=stats.norm.pdf(x_plot, res["t_mean"], res["t_sigma_overall"]), mode='lines',
                           name='整体 (Overall)', line=dict(color='darkred', width=2)))

            if res["t_lsl"] is not None:
                fig.add_vline(x=res["t_lsl"], line_color="red", line_width=1.5, line_dash="dash",
                              annotation_text="规格下限*")
            if res["t_target"] is not None:
                fig.add_vline(x=res["t_target"], line_color="green", line_dash="dot", annotation_text="目标*")
            if res["t_usl"] is not None:
                fig.add_vline(x=res["t_usl"], line_color="red", line_width=1.5, line_dash="dash",
                              annotation_text="规格上限*")

            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              title_text=f"📊 杂质含量过程能力报告 (使用 Box-Cox 变换, λ = {lambda_val})", title_x=0.35)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns([1.2, 1, 1])

            with c1:
                st.markdown("##### 📝 过程快照数据")
                lsl_str = f"{lsl_val:.2f}" if lsl_val is not None else "*"
                target_str = f"{target_val:.2f}" if target_val is not None else "*"
                usl_str = f"{usl_val:.2f}" if usl_val is not None else "*"

                st.table(pd.DataFrame({
                    "统计分析项": ["规格下限", "目标", "规格上限", "样本均值", "样本 N", "标准差 (整体)",
                                   "标准差 (组内)"],
                    "原始空间 (Raw)": [lsl_str, target_str, usl_str, f"{res['raw_mean']:.4f}", f"{res['N']}",
                                       f"{res['raw_sigma_overall']:.5f}", f"{res['raw_sigma_within']:.5f}"],
                    "变换后空间": [f"{res['t_lsl']:.5f}" if res['t_lsl'] else "*",
                                   f"{res['t_target']:.5f}" if res['t_target'] else "*",
                                   f"{res['t_usl']:.5f}" if res['t_usl'] else "*", f"{res['t_mean']:.4f}",
                                   f"{res['N']}", f"{res['t_sigma_overall']:.5f}", f"{res['t_sigma_within']:.5f}"]
                }))

            with c2:
                st.markdown("##### 🎯 能力指数")
                pp_str = f"{res['Pp']:.2f}" if res['Pp'] is not None else "*"
                ppl_str = f"{res['Ppl']:.2f}" if res['Ppl'] is not None else "*"
                ppu_str = f"{res['Ppu']:.2f}" if res['Ppu'] is not None else "*"
                ppk_str = f"{res['Ppk']:.2f}" if res['Ppk'] is not None else "*"
                cpm_str = f"{res['Cpm']:.2f}" if res['Cpm'] is not None else "*"

                cp_str = f"{res['Cp']:.2f}" if res['Cp'] is not None else "*"
                cpl_str = f"{res['Cpl']:.2f}" if res['Cpl'] is not None else "*"
                cpu_str = f"{res['Cpu']:.2f}" if res['Cpu'] is not None else "*"
                cpk_str = f"{res['Cpk']:.2f}" if res['Cpk'] is not None else "*"

                st.table(pd.DataFrame({
                    "整体能力指数 (Overall)": [f"Pp: {pp_str}", f"PPL: {ppl_str}", f"PPU: {ppu_str}", f"Ppk: {ppk_str}",
                                               f"Cpm: {cpm_str}"],
                    "潜在组内能力 (Within)": [f"Cp: {cp_str}", f"CPL: {cpl_str}", f"CPU: {cpu_str}", f"Cpk: {cpk_str}",
                                              "-"]
                }))

            with c3:
                st.markdown("##### 📉 性能评估指标 (PPM 缺陷外推)")
                st.table(pd.DataFrame({
                    "性能评估项 (PPM)": ["低于下限 (< LSL)", "高于上限 (> USL)", "合计不合格数 (Total)"],
                    "实际观测": [f"{res['ppm_obs_lsl']:.2f}", f"{res['ppm_obs_usl']:.2f}",
                                 f"{res['ppm_obs_total']:.2f}"],
                    "预期整体 (Overall*)": [f"{res['ppm_exp_overall_lsl']:.2f}" if res['t_lsl'] else "*",
                                            f"{res['ppm_exp_overall_usl']:.2f}", f"{res['ppm_exp_overall_total']:.2f}"],
                    "预期组内 (Within*)": [f"{res['ppm_exp_within_lsl']:.2f}" if res['t_lsl'] else "*",
                                           f"{res['ppm_exp_within_usl']:.2f}", f"{res['ppm_exp_within_total']:.2f}"]
                }))

    except Exception as e:
        st.error(f"⚠️ 无法正常读取或分析表格数据。原因: {e}")