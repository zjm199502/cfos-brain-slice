"""脑区活动与行为相关（v0.8.1）：每只动物填一个行为分数（如冻结时间 %），
逐脑区算 c-Fos 指标（默认阳性细胞密度）与行为分数的 Spearman 秩相关，Benjamini-Hochberg 校正，
输出按相关强弱排序的表、相关系数图和散点图。参考 eLife reviewed preprint 107908 的做法。

统计单位是动物：同一动物的多张脑片先合并（细胞数相加、面积相加，再算密度），与“组间比较”相同。
每组 8 只左右时相关系数很不稳定，所以一定给出 95% 置信区间（Fisher z 变换，Spearman 用
Bonett & Wright 2000 的标准误 sqrt((1 + r²/2) / (n − 3))），并做 FDR 校正。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import bh_fdr, per_animal, check_samples, metric_options, _plt, GROUP_COLORS

ALL_ANIMALS = "全部动物"
MIN_ANIMALS = 5          # 少于这么多只动物有数据的脑区不计算
WARN_ANIMALS = 8         # 少于这么多只时提醒结果不稳定


@dataclass
class BehaviorParams:
    metric: str = ""                 # c-Fos 指标，沿用“组间比较”的设置
    hemisphere: str = "左右合计"
    min_area_mm2: float = 0.05
    max_depth: int = 0
    behavior_name: str = "行为分数"   # 例如“冻结时间 %”
    group: str = ALL_ANIMALS         # 用哪些动物：全部动物，或某一个组
    alpha: float = 0.05

    def to_dict(self):
        return asdict(self)


def behavior_table(project) -> pd.DataFrame:
    """每只动物一行：动物、组别、行为分数（没填的为空）。"""
    rows = {}
    for r in project.sample_rows():
        a = r["动物"]
        if a and a not in rows:
            rows[a] = {"动物": a, "组别": r["组别"], "行为分数": project.behavior.get(a)}
    return pd.DataFrame(list(rows.values()), columns=["动物", "组别", "行为分数"])


def spearman_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    r = float(np.clip(r, -0.9999, 0.9999))
    z = np.arctanh(r)
    se = np.sqrt((1 + r * r / 2) / (n - 3))
    k = st.norm.ppf(0.5 + conf / 2)
    return float(np.tanh(z - k * se)), float(np.tanh(z + k * se))


def correlate(animal: pd.DataFrame, behavior: dict, params: BehaviorParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """animal：stats.per_animal 的输出。返回 (相关结果, 动物×脑区 指标宽表)。"""
    df = animal[animal["半脑"] == params.hemisphere].copy()
    if params.group != ALL_ANIMALS:
        df = df[df["组别"] == params.group]
    if params.max_depth:
        df = df[df["层级"] <= params.max_depth]
    df = df[df["面积 (mm²)"] >= params.min_area_mm2]
    df["行为分数"] = df["动物"].map(behavior)
    df = df[df["行为分数"].notna() & df[params.metric].notna()]
    rows = []
    for (acr, name, lvl), g in df.groupby(["脑区缩写", "脑区名称", "层级"], sort=False):
        n = len(g)
        x, y = g[params.metric].to_numpy(float), g["行为分数"].to_numpy(float)
        if n < MIN_ANIMALS or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        r, p = st.spearmanr(x, y)
        lo, hi = spearman_ci(r, n)
        rows.append({"脑区缩写": acr, "脑区名称": name, "层级": int(lvl), "动物数": n,
                     "Spearman r": r, "95% CI 下限": lo, "95% CI 上限": hi, "p 值": p})
    res = pd.DataFrame(rows, columns=["脑区缩写", "脑区名称", "层级", "动物数", "Spearman r", "95% CI 下限",
                                      "95% CI 上限", "p 值"])
    res["q 值（FDR）"] = bh_fdr(res["p 值"].to_numpy()) if len(res) else []
    res["显著"] = np.where(res["q 值（FDR）"] < params.alpha, "是", "")
    res["方向"] = np.where(res["Spearman r"] > 0, "越活跃，行为分数越高", "越活跃，行为分数越低")
    res = res.sort_values(["q 值（FDR）", "p 值"], key=lambda s: s.fillna(9)).reset_index(drop=True)
    wide = df.pivot_table(index=["组别", "动物", "行为分数"], columns="脑区缩写", values=params.metric).reset_index()
    return res, wide


def rank_plot(res: pd.DataFrame, params: BehaviorParams, path, n=20):
    """相关最强的脑区：Spearman r 和 95% 置信区间。"""
    plt = _plt()
    top = res.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.32 * len(top) + 1.2)))
    yy = np.arange(len(top))
    col = np.where(top["显著"] == "是", "#d62728", "#7f7f7f")
    ax.errorbar(top["Spearman r"], yy, xerr=[top["Spearman r"] - top["95% CI 下限"], top["95% CI 上限"] - top["Spearman r"]],
                fmt="none", ecolor="#bbbbbb", elinewidth=1.5, capsize=2)
    ax.scatter(top["Spearman r"], yy, c=col, s=36, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(top["脑区缩写"])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(f"Spearman r（{params.metric} 与 {params.behavior_name}）")
    ax.set_title(f"相关最强的 {len(top)} 个脑区（红点：q<{params.alpha}；横线：95% 置信区间）", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def scatter_grid(res: pd.DataFrame, wide: pd.DataFrame, params: BehaviorParams, path, n=6):
    plt = _plt()
    top = res.head(n)
    if top.empty:
        return
    k = len(top); cols = min(3, k); rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    groups = sorted(wide["组别"].unique())
    for ax, (_, r) in zip(axs.ravel(), top.iterrows()):
        for gi, g in enumerate(groups):
            sub = wide[wide["组别"] == g]
            ax.scatter(sub[r["脑区缩写"]], sub["行为分数"], s=30, color=GROUP_COLORS[gi % len(GROUP_COLORS)],
                       label=g, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{r['脑区缩写']}  r={r['Spearman r']:.2f}  q={r['q 值（FDR）']:.3f}", fontsize=10)
        ax.set_xlabel(params.metric, fontsize=8); ax.set_ylabel(params.behavior_name, fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axs.ravel()[k:]:
        ax.axis("off")
    if len(groups) > 1:
        axs.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"相关最强的 {k} 个脑区（每个点是一只动物）", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_behavior_correlation(project, out_dir, params: BehaviorParams, region_sec: pd.DataFrame | None = None,
                             progress=None):
    """计算并保存：脑区与行为相关.xlsx + 相关系数图 + 散点图。返回 (输出文件夹, 摘要文字, 图片路径列表)。"""
    from pathlib import Path
    from .export import build_tables, _write_xlsx
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if region_sec is None:
        if progress:
            progress(0, 3, "正在统计每张脑片的脑区结果…")
        skip, depth = project.export_skip_layers, project.export_max_depth
        project.export_skip_layers, project.export_max_depth = False, 0
        try:
            region_sec = build_tables(project).get("按脑区×脑片")
        finally:
            project.export_skip_layers, project.export_max_depth = skip, depth
    if region_sec is None or region_sec.empty:
        raise ValueError("还没有按脑区的结果。请先完成细胞识别和“⑥ 脑图谱配准”。")
    used = set(region_sec["图片"].astype(str) + "/" + region_sec["脑片"].astype(str))
    errs, warns = check_samples({k: v for k, v in project.samples.items() if k in used})
    if errs:
        raise ValueError("\n".join(errs))
    # 只需要动物编号；没填组别的动物归到“未分组”
    samples = {k: {"animal": v.get("animal", ""), "group": v.get("group", "") or "未分组"}
               for k, v in project.samples.items()}
    animal = per_animal(region_sec, samples)
    if animal.empty:
        raise ValueError("还没有填写动物编号。请先在上面的表格里为每张脑片填写“动物编号”。")
    behavior = {a: float(v) for a, v in project.behavior.items() if v is not None and np.isfinite(float(v))}
    have = sorted(set(animal["动物"]) & set(behavior))
    if params.group != ALL_ANIMALS:
        have = sorted(set(animal.loc[animal["组别"] == params.group, "动物"]) & set(behavior))
    if len(have) < MIN_ANIMALS:
        raise ValueError(f"填了行为分数的动物只有 {len(have)} 只，至少需要 {MIN_ANIMALS} 只才能算相关"
                         f"（建议 ≥ {WARN_ANIMALS} 只）。请在表格的“行为分数”列填写。")
    params.metric = params.metric or metric_options(project.channel_names, grading=project.grading)[0]
    if progress:
        progress(1, 3, "正在逐脑区计算相关…")
    res, wide = correlate(animal, behavior, params)
    if res.empty:
        raise ValueError("没有可以计算的脑区：可能是面积筛选太严，或各动物的指标 / 行为分数完全相同。")
    out = res.copy()
    for c in ("Spearman r", "95% CI 下限", "95% CI 上限"):
        out[c] = out[c].round(3)
    groups_in = sorted(wide["组别"].unique())
    info = pd.DataFrame([
        ("c-Fos 指标", params.metric), ("行为指标", params.behavior_name),
        ("动物", params.group + ("（" + "、".join(groups_in) + "）" if params.group == ALL_ANIMALS else "")),
        ("相关方法", "Spearman 秩相关（对离群值和非线性单调关系更稳）"),
        ("置信区间", "95%，Fisher z 变换，标准误 sqrt((1 + r²/2)/(n − 3))（Bonett & Wright 2000）"),
        ("多重比较校正", "Benjamini-Hochberg（FDR），按所有计算的脑区校正"),
        ("统计单位", "动物（同一动物的多张脑片先合并：细胞数相加、面积相加，再算密度）"),
        ("半脑", params.hemisphere), ("纳入条件", f"该动物在该脑区面积 ≥ {params.min_area_mm2} mm²；"
                                          f"至少 {MIN_ANIMALS} 只动物有数据的脑区才计算"),
        ("脑区层级", "全部" if not params.max_depth else f"≤ 第 {params.max_depth} 级"),
        ("显著标准", f"q < {params.alpha}"), ("动物数", len(have)),
        ("提醒", "相关不等于因果；把不同组的动物放在一起算时，相关可能只是反映了组间差异，"
                 "建议同时看每个组内的相关。"),
    ], columns=["项目", "内容"])
    xlsx = out_dir / "脑区与行为相关.xlsx"
    _write_xlsx({"相关排序": out, "按动物数据": wide.round(3), "统计方法": info}, xlsx)
    if progress:
        progress(2, 3, "正在画图…")
    p1 = out_dir / "相关系数_前20个脑区.png"; rank_plot(res, params, p1)
    p2 = out_dir / "散点图_前6个脑区.png"; scatter_grid(res, wide, params, p2)
    nsig = int((res["显著"] == "是").sum())
    best = res.iloc[0]
    summary = (f"{len(have)} 只动物，计算了 {len(res)} 个脑区，q<{params.alpha} 的有 {nsig} 个。"
               f"相关最强：{best['脑区缩写']}（r = {best['Spearman r']:.2f}，95% CI {best['95% CI 下限']:.2f}～"
               f"{best['95% CI 上限']:.2f}）。")
    if len(have) < WARN_ANIMALS:
        summary += f"\n注意：动物少于 {WARN_ANIMALS} 只，相关系数很不稳定（置信区间很宽），仅供参考。"
    if params.group == ALL_ANIMALS and len(groups_in) > 1:
        summary += "\n提醒：几个组的动物放在一起算，相关可能只反映组间差异，建议再分别看每个组。"
    for w in warns:
        summary += "\n注意：" + w
    return out_dir, summary, [p1, p2]
