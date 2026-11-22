"""实验分组与组间统计：把每张脑片的脑区结果按动物合并，再按组比较。

数据流：按脑区×脑片（每张脑片每个脑区的细胞数和面积）
  → 按动物汇总（同一动物的所有脑片细胞数相加、面积相加，再算密度）
  → 按组汇总（各组动物的平均值、标准误、动物数）
  → 组间比较（每个脑区一个检验，Benjamini-Hochberg 校正多重比较）
统计单位是“动物”，不是脑片，避免把同一只动物的多张脑片当作独立样本（伪重复）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from . import region_sets as _RS
from scipy import stats as st

KEY_COLS = ["脑区缩写", "脑区名称", "层级", "半脑"]


@dataclass
class StatsParams:
    metric: str = ""                # 比较的指标，例如 "Red+ 密度 (个/mm²)"；空 = 第一个标记物通道的密度
    test: str = "welch"             # welch = Welch t 检验；mannwhitney = Mann-Whitney U（非参数）
    control: str = ""               # 对照组名称；多组时其他各组分别和它比较；空 = 第一个组
    hemisphere: str = "左右合计"     # 左右合计 / 左 / 右
    min_area_mm2: float = 0.05      # 每只动物在这个脑区至少有多少面积才纳入统计（太小的区域密度不可靠）
    max_depth: int = 0              # 只比较不超过第几级的脑区（0 = 全部）
    region_set: str = ""             # 比较哪些脑区：见 core/region_sets.py（空 = 旧设置，按 max_depth）
    alpha: float = 0.05             # 校正后 q 值小于它算显著
    region_ids: list[int] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def count_columns(df: pd.DataFrame) -> list[str]:
    """细胞数的列：“X+”和“X+Y 共标”。"""
    return [c for c in df.columns if (c.endswith("+") or c.endswith(" 共标") or c[-3:] in ("+ 弱", "+ 中", "+ 强"))
            and pd.api.types.is_numeric_dtype(df[c])]


def metric_options(channel_names, nucleus_name="DAPI", grading=None) -> list[str]:
    """界面上可选的指标：每个通道和每种共标的密度，以及双标占单标的百分比。
    v0.8.3：有 DAPI 通道时，“X+ 占 DAPI+ 的比例 %”放在最前面，作为推荐主指标（不受切片厚度、组织收缩、面积误差影响）。"""
    from itertools import combinations
    opts = []
    if nucleus_name in channel_names:
        for nm in channel_names:
            if nm != nucleus_name:
                opts.append(f"{nm}+ 占 {nucleus_name}+ 的比例 %")
    for nm in channel_names:
        if nm != nucleus_name:
            opts.append(f"{nm}+ 密度 (个/mm²)")
    markers = [n for n in channel_names if n != nucleus_name]
    for a, b in combinations(channel_names, 2):
        if nucleus_name in (a, b):
            continue
        opts.append(f"{a}+{b} 共标 密度 (个/mm²)")
        opts.append(f"{a}+{b} 共标 占 {a}+ 的比例 %")
        opts.append(f"{a}+{b} 共标 占 {b}+ 的比例 %")
    for nm in (grading or {}):
        if nm in channel_names:
            for lab in ("强", "中", "弱"):
                opts.append(f"{nm}+ {lab} 密度 (个/mm²)")
            opts.append(f"{nm}+ 强 占 {nm}+ 的比例 %")
    if nucleus_name in channel_names:
        opts.append(f"{nucleus_name}+ 密度 (个/mm²)")
    if not markers and channel_names:
        opts = [f"{channel_names[0]}+ 密度 (个/mm²)"]
    return opts


def add_metrics(t: pd.DataFrame) -> pd.DataFrame:
    """在已经汇总好的计数表上加密度和比例列。"""
    t = t.copy()
    area = t["面积 (mm²)"].replace(0, np.nan)
    for c in count_columns(t):
        t[f"{c} 密度 (个/mm²)"] = t[c] / area
        if c[-3:] in ("+ 弱", "+ 中", "+ 强") and c[:-2] in t.columns:
            t[f"{c} 占 {c[:-2]} 的比例 %"] = 100.0 * t[c] / t[c[:-2]].replace(0, np.nan)
        if c.endswith(" 共标"):
            parts = c[:-3].split("+")
            if len(parts) == 2:
                for p in parts:
                    if f"{p}+" in t.columns:
                        t[f"{c} 占 {p}+ 的比例 %"] = 100.0 * t[c] / t[f"{p}+"].replace(0, np.nan)
                # v0.8.3 推荐主指标：X+ 占 DAPI+ 的比例 = 有核的 X+（DAPI+X 共标）÷ DAPI+，用合计后的细胞数算
                if NUC in parts and f"{NUC}+" in t.columns:
                    x = parts[1] if parts[0] == NUC else parts[0]
                    t[f"{x}+ 占 {NUC}+ 的比例 %"] = 100.0 * t[c] / t[f"{NUC}+"].replace(0, np.nan)
    return t


NUC = "DAPI"


def dapi_note(animal: pd.DataFrame, metric: str) -> pd.DataFrame | None:
    """指标是“X+ 占 DAPI+ 的比例”时，按脑区给出提示：有动物 DAPI+ 不足 200 个，或是细胞极密的脑区。
    返回 KEY_COLS + “比例提示”；不是这个指标时返回 None。"""
    if not metric.endswith(f"占 {NUC}+ 的比例 %") or " 共标 " in metric or f"{NUC}+" not in animal.columns:
        return None
    from .coloc import MIN_NUCLEI, DAPI_NOTE
    from .regions import dense_regions
    dense = dense_regions()
    rows = []
    for k, g in animal.groupby(KEY_COLS, dropna=False):
        k = dict(zip(KEY_COLS, k))
        low = int((g[f"{NUC}+"].fillna(0) < MIN_NUCLEI).sum())
        if low:
            note = f"{DAPI_NOTE}（{low} 只动物 {NUC}+ 不足 {MIN_NUCLEI} 个）"
        elif k["脑区缩写"] in dense:
            note = f"{DAPI_NOTE}（细胞密集区，DAPI 容易漏识别）"
        else:
            note = ""
        rows.append({**k, "比例提示": note})
    return pd.DataFrame(rows)


def check_samples(samples: dict) -> tuple[list[str], list[str]]:
    """检查分组填写：返回 (错误, 提醒)。
    错误：同一只动物被填进了两个组（会被当成两只动物）。
    提醒：动物编号只差大小写、空格或全角/半角（如“M1”和“m1”），很可能是同一只。"""
    import unicodedata
    by_animal = {}
    for k, d in samples.items():
        a, g = (d or {}).get("animal", ""), (d or {}).get("group", "")
        if a and g:
            by_animal.setdefault(a, set()).add(g)
    errors = [f"动物“{a}”同时出现在 {len(gs)} 个组（{'、'.join(sorted(gs))}）里，请检查分组表。"
              for a, gs in by_animal.items() if len(gs) > 1]
    norm = {}
    for a in by_animal:
        n = unicodedata.normalize("NFKC", a).replace(" ", "").lower()
        norm.setdefault(n, []).append(a)
    warns = [f"动物编号 {'、'.join('“' + x + '”' for x in v)} 只差大小写、空格或全角半角，是否是同一只？"
             for v in norm.values() if len(v) > 1]
    return errors, warns


def per_animal(region_sec: pd.DataFrame, samples: dict) -> pd.DataFrame:
    """region_sec：按脑区×脑片表（含“图片”“脑片”列）。samples：{"图片/脑片": {"animal":..., "group":...}}。
    返回每只动物每个脑区的计数、面积、密度。未填动物编号的脑片不纳入。"""
    if region_sec is None or region_sec.empty:
        return pd.DataFrame()
    df = region_sec.copy()
    key = df["图片"].astype(str) + "/" + df["脑片"].astype(str)
    df["动物"] = key.map(lambda k: (samples.get(k) or {}).get("animal", "")).fillna("")
    df["组别"] = key.map(lambda k: (samples.get(k) or {}).get("group", "")).fillna("")
    df = df[(df["动物"] != "") & (df["组别"] != "")]
    if df.empty:
        return pd.DataFrame()
    counts = count_columns(df)
    g = df.groupby(["组别", "动物"] + KEY_COLS, as_index=False, dropna=False)
    t = g[counts + ["面积 (mm²)"]].sum()
    n_sec = (df[["组别", "动物", "图片", "脑片"]].drop_duplicates()
             .groupby(["组别", "动物"]).size().rename("脑片数").reset_index())
    t = t.merge(n_sec, on=["组别", "动物"], how="left")
    return add_metrics(t)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = st.false_discovery_control(p[ok], method="bh")
    return q


def _test(a: np.ndarray, b: np.ndarray, test: str):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if test == "mannwhitney":
        return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    if np.all(a == a[0]) and np.all(b == b[0]):
        return 1.0 if a[0] == b[0] else np.nan
    return float(st.ttest_ind(a, b, equal_var=False).pvalue)


def eligible_rows(animal: pd.DataFrame, params: StatsParams) -> pd.DataFrame:
    a = animal[animal["半脑"] == params.hemisphere]
    a = _RS.filter_table(a, params.region_set, params.max_depth, region_ids=params.region_ids)
    return a.loc[a["面积 (mm²)"] >= params.min_area_mm2]


def _apply_fdr(table: pd.DataFrame, alpha: float):
    table["q 值 (FDR 校正)"] = bh_fdr(table["p 值"].to_numpy())
    table["显著"] = np.where(table["q 值 (FDR 校正)"] < alpha, "是", "")


def _overall_comparison(eligible: pd.DataFrame, groups: list[str], params: StatsParams) -> pd.DataFrame:
    """One omnibus test per region, using the same expected groups for every region."""
    rows = []
    rank_test = params.test == "mannwhitney"
    for key, region in eligible.groupby(KEY_COLS, dropna=False):
        values = []
        row = dict(zip(KEY_COLS, key))
        row["检验"] = "Kruskal–Wallis H" if rank_test else "Welch ANOVA"
        for group in groups:
            v = region.loc[region["组别"] == group, params.metric].to_numpy(float)
            v = v[np.isfinite(v)]
            values.append(v)
            row[f"{group} 动物数"] = len(v)
        statistic = p = np.nan
        note = ""
        if any(len(v) < 2 for v in values):
            note = "未检验：至少一组的有效动物数不足 2"
        elif all(np.all(v == values[0][0]) for v in values):
            note = "未检验：所有动物的数值相同"
        elif not rank_test and any(np.all(v == v[0]) for v in values):
            note = "未检验：至少一组的组内方差为零"
        else:
            result = st.kruskal(*values) if rank_test else st.f_oneway(*values, equal_var=False)
            statistic, p = result.statistic, result.pvalue
            if rank_test and any(len(v) < 5 for v in values):
                note = "有组少于 5 只动物，卡方近似需谨慎解释"
        rows.append({**row, "统计量": statistic, "p 值": p, "说明": note})
    result = pd.DataFrame(rows)
    if not result.empty:
        _apply_fdr(result, params.alpha)
        result = result.sort_values("p 值", kind="stable").reset_index(drop=True)
    return result


def compare_groups(animal: pd.DataFrame, params: StatsParams, groups=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (按组汇总, 组间比较)。组间比较：每个“对照组 vs 其他组”一组行，每行一个脑区。"""
    if animal.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = params.metric or next((c for c in animal.columns if c.endswith("密度 (个/mm²)")), None)
    if m not in animal.columns:
        raise ValueError(f"没有这个指标：{m}")
    a = eligible_rows(animal, params)
    groups = sorted(animal["组别"].unique()) if groups is None else groups
    return _compare_eligible(a, params, m, groups)


def _compare_eligible(a, params, m, groups):
    ctrl = params.control if params.control in groups else (groups[0] if groups else "")
    # 按组汇总
    rows = []
    for (grp, *k), g in a.groupby(["组别"] + KEY_COLS, dropna=False):
        v = g[m].to_numpy(float); v = v[np.isfinite(v)]
        rows.append(dict(zip(["组别"] + KEY_COLS, [grp] + list(k)),
                         **{"动物数": len(v), "平均值": v.mean() if len(v) else np.nan,
                            "标准误 (SEM)": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                            "标准差": v.std(ddof=1) if len(v) > 1 else np.nan,
                            "各动物数值": "、".join(f"{x:.1f}" for x in v)}))
    gsum = pd.DataFrame(rows)
    # 组间比较
    comps = []
    for other in [g for g in groups if g != ctrl]:
        rows = []
        for k, g in a[a["组别"].isin([ctrl, other])].groupby(KEY_COLS, dropna=False):
            va = g.loc[g["组别"] == ctrl, m].to_numpy(float)
            vb = g.loc[g["组别"] == other, m].to_numpy(float)
            va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            ma, mb = va.mean() if len(va) else np.nan, vb.mean() if len(vb) else np.nan
            rows.append(dict(zip(KEY_COLS, k), **{
                "比较": f"{other} vs {ctrl}",
                "对照组": ctrl, "比较组": other,
                f"{ctrl} 平均": ma, f"{ctrl} 动物数": len(va),
                f"{other} 平均": mb, f"{other} 动物数": len(vb),
                "倍数变化": mb / ma if ma and not np.isnan(ma) else np.nan,
                "log2 倍数变化": np.log2((mb + 1e-9) / (ma + 1e-9)) if ma > 0 or mb > 0 else np.nan,
                "p 值": _test(va, vb, params.test)}))
        c = pd.DataFrame(rows)
        if c.empty:
            continue
        comps.append(c.sort_values("p 值"))
    comp = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    if not comp.empty:
        _apply_fdr(comp, params.alpha)
    note = dapi_note(a, m)
    if note is not None:
        gsum = gsum.merge(note, on=KEY_COLS, how="left") if not gsum.empty else gsum
        comp = comp.merge(note, on=KEY_COLS, how="left") if not comp.empty else comp
    for t in (gsum, comp):
        for col in t.columns:
            if pd.api.types.is_float_dtype(t[col]):
                if col not in ("p 值", "q 值 (FDR 校正)"):
                    t[col] = t[col].round(3)
    return gsum, comp


# ---------------------------------------------------------------- 图
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC",
              "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    else:
        import glob
        for path in glob.glob("/usr/share/fonts/**/wqy-microhei.ttc", recursive=True) + glob.glob("C:/Windows/Fonts/msyh.ttc"):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


GROUP_COLORS = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def volcano_plot(comp: pd.DataFrame, metric: str, path, alpha=0.05, label_top=12):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = comp["log2 倍数变化"].to_numpy(float)
    y = -np.log10(comp["p 值"].to_numpy(float).clip(1e-300))
    sig = comp["显著"].eq("是").to_numpy()
    ax.scatter(x[~sig], y[~sig], s=12, c="#b0b0b0", label="不显著")
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s=18, c="#d62728", label=f"升高（q<{alpha}）")
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s=18, c="#1f77b4", label=f"降低（q<{alpha}）")
    top = comp.assign(_y=y).sort_values("_y", ascending=False).head(label_top)
    for _, r in top.iterrows():
        ax.annotate(str(r["脑区缩写"]), (r["log2 倍数变化"], r["_y"]), fontsize=8, xytext=(3, 2), textcoords="offset points")
    ax.axvline(0, c="k", lw=0.5)
    ax.set_xlabel("log2 倍数变化（实验组 / 对照组）")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{comp['比较'].iloc[0]}：{metric}", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bar_plot(animal: pd.DataFrame, comp: pd.DataFrame, params: StatsParams, metric: str, path, n=15):
    """显著（或 p 最小）的前 n 个脑区：各组平均值 ± SEM，叠加每只动物的点。"""
    plt = _plt()
    regions = comp.sort_values("p 值").drop_duplicates("脑区缩写").head(n)["脑区缩写"].tolist()
    a = animal[(animal["半脑"] == params.hemisphere) & animal["脑区缩写"].isin(regions)]
    groups = sorted(a["组别"].unique().tolist())
    ctrl = params.control if params.control in groups else groups[0]
    groups = [ctrl] + [g for g in groups if g != ctrl]
    fig, ax = plt.subplots(figsize=(max(7, len(regions) * 0.75 + 2), 5))
    w = 0.8 / max(len(groups), 1)
    rng = np.random.default_rng(0)
    for gi, grp in enumerate(groups):
        means, sems = [], []
        for ri, reg in enumerate(regions):
            v = a[(a["组别"] == grp) & (a["脑区缩写"] == reg)][metric].to_numpy(float)
            v = v[~np.isnan(v)]
            means.append(v.mean() if len(v) else 0)
            sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            xs = ri + (gi - (len(groups) - 1) / 2) * w + rng.uniform(-w * 0.2, w * 0.2, len(v))
            ax.scatter(xs, v, s=10, c="k", zorder=3)
        xs = np.arange(len(regions)) + (gi - (len(groups) - 1) / 2) * w
        ax.bar(xs, means, w * 0.9, yerr=sems, capsize=2, color=GROUP_COLORS[gi % len(GROUP_COLORS)],
               alpha=0.8, label=grp)
    for ri, reg in enumerate(regions):
        q = comp[comp["脑区缩写"] == reg]["q 值 (FDR 校正)"].min()
        star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < params.alpha else ""
        if star:
            top = a[a["脑区缩写"] == reg][metric].max()
            ax.text(ri, top * 1.05, star, ha="center", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"差异最明显的 {len(regions)} 个脑区（{params.hemisphere}；* q<{params.alpha}，** q<0.01，*** q<0.001）", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_group_stats(project, out_dir, params: StatsParams, region_sec: pd.DataFrame | None = None, progress=None):
    """计算并保存：组间统计.xlsx + 火山图 + 柱状图。返回 (输出文件夹, 摘要文字, 图片路径列表)。"""
    from pathlib import Path
    from .export import build_tables, _write_xlsx
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if region_sec is None:
        if progress:
            progress(0, 3, "正在统计每张脑片的脑区结果…")
        region_sec = build_tables(project, intensity=False, all_regions=True).get("按脑区×脑片")
    if region_sec is None or region_sec.empty:
        raise ValueError("还没有按脑区的结果。请先完成细胞识别和“⑥ 脑图谱配准”。")
    covered = region_sec.loc[region_sec["半脑"] == params.hemisphere]
    if params.region_set == _RS.CUSTOM:
        covered = _RS.filter_table(covered, params.region_set, region_ids=params.region_ids)
        if covered.empty:
            raise ValueError("所选脑片在指定半脑未覆盖勾选的脑区。请调整脑区、半脑或脑片范围。")
    used = set(region_sec["图片"].astype(str) + "/" + region_sec["脑片"].astype(str))
    errs, warns = check_samples({k: v for k, v in project.samples.items() if k in used})
    if errs:
        raise ValueError("\n".join(errs))
    animal = per_animal(region_sec, project.samples)
    if animal.empty:
        raise ValueError("还没有填写分组。请在表格中为每张脑片填写“动物编号”和“组别”。")
    assigned = {r["组别"] for r in project.sample_rows() if r["组别"] and r["动物"]}
    groups = sorted(assigned | set(animal["组别"]))
    if len(groups) < 2:
        raise ValueError(f"只有一个组（{groups[0]}），至少需要两个组才能比较。")
    control = params.control if params.control in groups else groups[0]
    if progress:
        progress(1, 3, "正在做组间比较…")
    metric = params.metric or metric_options(project.channel_names, grading=project.grading)[0]
    params.metric = metric
    eligible = eligible_rows(animal, params)
    if metric not in eligible.columns:
        raise ValueError(f"没有这个指标：{metric}")
    if eligible.empty:
        raise ValueError("指定半脑、脑区和最小面积范围内没有可纳入的动物数据。")
    gsum, comp = _compare_eligible(eligible, params, metric, groups)
    overall = _overall_comparison(eligible, groups, params) if len(groups) >= 3 else pd.DataFrame()
    n_by_group = (eligible.loc[np.isfinite(eligible[metric]), :].groupby("组别")["动物"]
                  .nunique().reindex(groups, fill_value=0))
    show = eligible.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].round(3)
    info = pd.DataFrame([
        ("比较指标", metric), ("组别数量", len(groups)),
        ("与对照组比较", "Welch t 检验" if params.test == "welch" else "Mann-Whitney U 检验"),
        ("多重比较校正", "使用 Benjamini-Hochberg 分别校正两类检验：各脑区的总体检验，以及跨脑区和组别的对照比较。"),
        ("统计单位", "动物（同一动物的多张脑片先合并：细胞数相加、面积相加，再算密度）"),
        ("对照组", control), ("半脑", params.hemisphere),
        ("纳入条件", f"该动物在该脑区面积 ≥ {params.min_area_mm2} mm²"),
        ("脑区范围", _RS.describe(params.region_set, params.max_depth, region_ids=params.region_ids)),
        *_RS.selection_details(params.region_set, params.region_ids),
        ("显著标准", f"q < {params.alpha}"),
    ] + ([("多组总体检验", "Welch ANOVA" if params.test == "welch" else "Kruskal–Wallis H")]
         if len(groups) >= 3 else [])
      + [(f"组 {g} 有效动物数", int(n)) for g, n in n_by_group.items()], columns=["项目", "内容"])
    xlsx = out_dir / "组间统计.xlsx"
    tables = {"多组总体检验": overall} if len(groups) >= 3 else {}
    tables.update({"组间比较": comp, "按组汇总": gsum, "按动物×脑区": show, "统计方法": info})
    _write_xlsx(tables, xlsx)
    from .workflow_status import record_table_output
    record_table_output(project, "group", covered)
    if progress:
        progress(2, 3, "正在画图…")
    figs = []
    if not comp.empty:
        import re
        for k, (other, c) in enumerate(comp.dropna(subset=["p 值"]).groupby("比较组", sort=False), 1):
            label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", f"{other} vs {control}")
            safe = f"{k:02d}_{label[:60]}"
            p1 = out_dir / f"火山图_{safe}.png"; volcano_plot(c, metric, p1, params.alpha); figs.append(p1)
            p2 = out_dir / f"柱状图_{safe}.png"
            bar_plot(eligible[eligible["组别"].isin([control, other])], c, params, metric, p2)
            figs.append(p2)
    if not gsum.empty:
        from .heatmap import heatmap_figure, default_ap_levels
        levels = default_ap_levels(project)
        observed = set(gsum["组别"])
        others = [g for g in groups if g != control and g in observed]
        heatmaps = []
        for start in range(0, max(1, len(others)), 2):
            page = [control, *others[start:start + 2]]
            name = f"脑区热图_{start // 2 + 1:02d}.png" if len(others) > 2 else "脑区热图.png"
            path = out_dir / name
            heatmap_figure(gsum[gsum["组别"].isin(page)], comp, metric, control, levels, path, params.alpha)
            heatmaps.append(path)
        figs = heatmaps + figs
    nsig = int((comp["显著"] == "是").sum()) if not comp.empty else 0
    summary = (f"{len(groups)} 个组（" + "，".join(f"{g} {int(n)} 只" for g, n in n_by_group.items()) + "），"
               f"与对照组的比较中有 {nsig} 项达到 q<{params.alpha}。")
    if len(groups) >= 3:
        tested = int(overall["p 值"].notna().sum()) if not overall.empty else 0
        significant = int(overall["显著"].eq("是").sum()) if not overall.empty else 0
        summary += f"\n{tested} 个脑区完成多组总体检验，{significant} 个达到 q<{params.alpha}。"
    if (n_by_group < 3).any():
        summary += "\n注意：有的组少于 3 只动物，统计结果不可靠，仅供参考。"
    for w in warns:
        summary += "\n注意：" + w
    return out_dir, summary, figs
