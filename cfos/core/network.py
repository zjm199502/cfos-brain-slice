"""c-Fos 脑区功能网络分析（跨动物共激活网络）。

思路（Wheeler et al. 2013 等常用方法）：
- 在同一组内，以“动物”为样本，计算每两个脑区 c-Fos 指标的相关系数（Pearson 或 Spearman），得到相关矩阵；
- 只保留强的正相关（|r| 或 r 不低于阈值），得到网络：节点 = 脑区，边 = 强相关；
- 结论不能依赖某一个阈值，所以同时在多个阈值下计算；
- 节点指标：度（连了几条边）、介数中心性（处在多少最短路径上）、所属模块；度和介数都排在前 20% 的为枢纽脑区；
- 全局指标：网络密度、平均度、全局效率、聚类系数、模块度；
- 组间比较：
  · 每条边：Fisher z 检验比较两组的相关系数，FDR 校正；
  · 全局指标和每个脑区的度：随机打乱动物的组别（置换检验）得到 p 值；
- 稳健性：每组每次去掉一只动物重新算（留一法），看枢纽脑区是否稳定；对动物重抽样（bootstrap）给全局指标置信区间。

每组只有 7–9 只动物时，单条边的相关系数置信区间很宽（n=8 时 r=0.7 的 95% CI 约为 0.03–0.94），
所以应重点看跨阈值一致、留一法稳定的模式，单条边的显著性仅供参考。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import per_animal, bh_fdr, metric_options, KEY_COLS

MAJOR = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB"]
MAJOR_ZH = {"Isocortex": "新皮层", "OLF": "嗅区", "HPF": "海马结构", "CTXsp": "皮层下板", "STR": "纹状体",
            "PAL": "苍白球", "TH": "丘脑", "HY": "下丘脑", "MB": "中脑", "P": "脑桥", "MY": "延髓", "CB": "小脑",
            "其他": "其他"}


@dataclass
class NetParams:
    metric: str = ""                         # 空 = 第一个标记物的密度
    hemisphere: str = "左右合计"
    control: str = ""                        # 对照组（空 = 按名称排第一的组）
    depth: int = 7                           # 脑区粒度：截到第几级（取这一级及以上的最细脑区，互不重叠）
    min_area_mm2: float = 0.05
    min_coverage: float = 0.8                # 脑区至少要在每组这么多比例的动物中有数据
    method: str = "pearson"                  # pearson / spearman
    thresholds: list = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])
    main_threshold: float = 0.7              # 画网络图、判断枢纽用的主阈值
    positive_only: bool = True               # 只用正相关建网（负相关的生物学含义不明确，常见做法是只用正相关）
    n_perm: int = 1000
    n_boot: int = 500
    seed: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 脑区选择
def _structures():
    from .atlas import load_atlas
    st_ = load_atlas().structures
    id2acr = dict(zip(st_["id"], st_["acronym"]))
    anc = {r["acronym"]: [id2acr.get(int(x)) for x in str(r["path"]).split("/") if x] for _, r in st_.iterrows()}
    return anc


def major_of(acr: str, anc=None) -> str:
    anc = anc or _structures()
    for a in anc.get(acr, []):
        if a in MAJOR:
            return a
    return "其他"


def select_regions(animal: pd.DataFrame, depth: int) -> list[str]:
    """把图谱截到 depth 级后的“最细脑区”：层级 ≤ depth（不含皮层分层），且它的下级脑区都没有出现在数据中。
    这样选出的脑区互不包含，不会因为“上级包含下级”产生虚假的相关。"""
    anc = _structures()
    sub = animal[animal["层级"] <= depth]
    # 皮层分层（layer 1、2/3…）不作为单独节点：同一脑区各层之间高度相关，会造成虚假的“模块”
    sub = sub[~sub["脑区名称"].astype(str).str.contains(r"[Ll]ayer", regex=True)]
    present = set(sub["脑区缩写"])
    present.discard("root")
    has_child = set()
    for a in present:
        for p in anc.get(a, [])[:-1]:
            if p in present:
                has_child.add(p)
    keep = [a for a in present if a not in has_child and major_of(a, anc) != "其他"]
    order = {m: i for i, m in enumerate(MAJOR)}
    return sorted(keep, key=lambda a: (order.get(major_of(a, anc), 99), a))


def animal_matrix(animal: pd.DataFrame, params: NetParams, regions=None):
    """返回 (宽表：行 = (组别, 动物)，列 = 脑区，值 = 指标), 纳入的脑区列表。"""
    a = animal[animal["半脑"] == params.hemisphere]
    a = a[a["面积 (mm²)"] >= params.min_area_mm2]
    regions = regions or select_regions(a, params.depth)
    a = a[a["脑区缩写"].isin(regions)]
    wide = a.pivot_table(index=["组别", "动物"], columns="脑区缩写", values=params.metric, aggfunc="first")
    # 每组都要有足够动物覆盖
    ok = np.ones(wide.shape[1], bool)
    for g, sub in wide.groupby(level=0):
        ok &= (sub.notna().mean(axis=0).to_numpy() >= params.min_coverage)
        ok &= (sub.std(axis=0).fillna(0).to_numpy() > 0)
    cols = [c for c, k in zip(wide.columns, ok) if k]
    order = [r for r in regions if r in cols]
    return wide[order], order


# ---------------------------------------------------------------- 相关与网络
def corr(X: np.ndarray, method="pearson") -> np.ndarray:
    """X: 动物 × 脑区（可含 NaN，用成对完整的动物计算）。"""
    df = pd.DataFrame(X)
    r = df.corr(method=method, min_periods=4).to_numpy().copy()
    np.fill_diagonal(r, 1.0)
    return r


def corr_p(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """相关系数的双侧 p 值（t 分布近似，Spearman 同样适用于近似）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        p = 2 * st.t.sf(np.abs(t), np.clip(n - 2, 1, None))
    np.fill_diagonal(p, np.nan)
    return p


def pair_n(X: np.ndarray) -> np.ndarray:
    v = (~np.isnan(X)).astype(int)
    return v.T @ v


def adjacency(r: np.ndarray, thr: float, positive_only=True) -> np.ndarray:
    A = (r >= thr) if positive_only else (np.abs(r) >= thr)
    A = A & ~np.eye(len(r), dtype=bool)
    return A & np.isfinite(r)


def global_efficiency(A: np.ndarray) -> float:
    from scipy.sparse.csgraph import shortest_path
    n = len(A)
    if n < 2:
        return np.nan
    D = shortest_path(A.astype(float), unweighted=True, directed=False)
    with np.errstate(divide="ignore"):
        inv = 1.0 / D
    inv[~np.isfinite(inv)] = 0
    np.fill_diagonal(inv, 0)
    return float(inv.sum() / (n * (n - 1)))


def fast_metrics(A: np.ndarray) -> dict:
    """置换和重抽样中反复用到的、计算快的全局指标。"""
    n = len(A); deg = A.sum(1)
    return {"网络密度": A.sum() / max(n * (n - 1), 1), "平均度": float(deg.mean()), "全局效率": global_efficiency(A)}


def graph_metrics(A: np.ndarray, names: list[str], seed=0):
    """返回 (全局指标 dict, 节点表)。"""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from(zip(*np.nonzero(np.triu(A))))
    deg = np.array([d for _, d in G.degree()])
    btw = nx.betweenness_centrality(G, normalized=True)
    clus = nx.clustering(G)
    if G.number_of_edges():
        comms = nx.algorithms.community.louvain_communities(G, seed=seed)
        mod = nx.algorithms.community.modularity(G, comms)
    else:
        comms, mod = [{i} for i in range(len(names))], np.nan
    comm_of = {}
    for k, c in enumerate(sorted(comms, key=len, reverse=True), 1):
        for i in c:
            comm_of[i] = k if len(c) > 1 else 0
    b = np.array([btw[i] for i in range(len(names))])
    hub = (deg >= np.percentile(deg, 80)) & (b >= np.percentile(b, 80)) & (deg > 0)
    nodes = pd.DataFrame({"脑区缩写": names, "度": deg, "介数中心性": b.round(4),
                          "聚类系数": [round(clus[i], 3) for i in range(len(names))],
                          "模块": [comm_of[i] for i in range(len(names))], "枢纽脑区": np.where(hub, "是", "")})
    g = fast_metrics(A)
    g.update({"平均聚类系数": float(np.mean(list(clus.values()))) if len(clus) else np.nan, "模块度": mod,
              "模块数（≥2 个脑区）": sum(1 for c in comms if len(c) > 1),
              "孤立脑区数": int((deg == 0).sum())})
    return g, nodes


# ---------------------------------------------------------------- 组间比较
def fisher_z_compare(r1, n1, r2, n2):
    with np.errstate(invalid="ignore", divide="ignore"):
        z1, z2 = np.arctanh(np.clip(r1, -0.9999, 0.9999)), np.arctanh(np.clip(r2, -0.9999, 0.9999))
        se = np.sqrt(1.0 / np.clip(n1 - 3, 1e-9, None) + 1.0 / np.clip(n2 - 3, 1e-9, None))
        z = (z1 - z2) / se
        p = 2 * st.norm.sf(np.abs(z))
    bad = (n1 < 4) | (n2 < 4)
    z = np.where(bad, np.nan, z); p = np.where(bad, np.nan, p)
    return z, p


def run_network(animal: pd.DataFrame, params: NetParams, progress=None):
    """核心计算。返回 dict：regions、groups、R/P/N（每组）、global、nodes、edges_diff、perm、loo。"""
    rng = np.random.default_rng(params.seed)
    wide, regions = animal_matrix(animal, params)
    if len(regions) < 5:
        raise ValueError(f"满足条件的脑区只有 {len(regions)} 个，无法做网络分析。可以把“脑区粒度”调小一些、"
                         "降低“最小面积”，或检查是否所有动物都已配准。")
    groups = sorted(wide.index.get_level_values(0).unique())
    if params.control in groups:  # 对照组放在第一位：组间比较都是“其他组 vs 对照组”
        groups = [params.control] + [g for g in groups if g != params.control]
    X = {g: wide.loc[g].to_numpy(float) for g in groups}
    for g in groups:
        if len(X[g]) < 4:
            raise ValueError(f"组“{g}”只有 {len(X[g])} 只动物，至少需要 4 只（建议 ≥ 8 只）才能计算相关网络。")
    thr = sorted(set(params.thresholds) | {params.main_threshold})
    out = {"regions": regions, "groups": groups, "R": {}, "P": {}, "N": {}, "n_animals": {g: len(X[g]) for g in groups}}
    grows, nrows = [], []
    steps = len(groups) * 3 + 2; k = 0

    def tick(txt):
        nonlocal k
        k += 1
        if progress:
            progress(k, steps, txt)

    for g in groups:
        tick(f"正在计算 {g} 组的相关矩阵和网络…")
        R = corr(X[g], params.method); N = pair_n(X[g]); P = corr_p(R, N)
        out["R"][g], out["P"][g], out["N"][g] = R, P, N
        for t in thr:
            gm, nodes = graph_metrics(adjacency(R, t, params.positive_only), regions, params.seed)
            grows.append({"组别": g, "阈值": t, **gm})
            nodes.insert(0, "阈值", t); nodes.insert(0, "组别", g)
            nrows.append(nodes)
        # bootstrap：对动物有放回重抽样，给全局指标置信区间（主阈值）
        tick(f"{g} 组：重抽样估计置信区间…")
        boots = []
        n = len(X[g])
        for _ in range(params.n_boot):
            idx = rng.integers(0, n, n)
            if len(set(idx)) < 4:
                continue
            boots.append(fast_metrics(adjacency(corr(X[g][idx], params.method), params.main_threshold,
                                                params.positive_only)))
        bt = pd.DataFrame(boots)
        for row in grows:
            if row["组别"] == g and row["阈值"] == params.main_threshold:
                for m in bt.columns:
                    lo, hi = np.nanpercentile(bt[m], [2.5, 97.5])
                    row[f"{m} 95% CI"] = f"{lo:.3g} – {hi:.3g}"
        # 留一法：每次去掉一只动物，看枢纽脑区是否稳定
        tick(f"{g} 组：留一法检查枢纽脑区稳定性…")
        hubc = np.zeros(len(regions)); degs = []
        for i in range(n):
            keep = np.arange(n) != i
            _, nd = graph_metrics(adjacency(corr(X[g][keep], params.method), params.main_threshold,
                                            params.positive_only), regions, params.seed)
            hubc += (nd["枢纽脑区"] == "是").to_numpy(); degs.append(nd["度"].to_numpy())
        out.setdefault("loo", {})[g] = (hubc / n, np.array(degs))
    gl = pd.DataFrame(grows)
    nodes = pd.concat(nrows, ignore_index=True)
    # 留一法稳定性写进主阈值的节点表
    for g in groups:
        frac, degs = out["loo"][g]
        m = (nodes["组别"] == g) & (nodes["阈值"] == params.main_threshold)
        nodes.loc[m, "留一法中为枢纽的比例"] = np.round(frac, 2)
        nodes.loc[m, "留一法度 范围"] = [f"{a}–{b}" for a, b in zip(degs.min(0), degs.max(0))]
    anc = _structures()
    nodes.insert(3, "大脑区", nodes["脑区缩写"].map(lambda a: MAJOR_ZH.get(major_of(a, anc), "其他")))
    out["global"], out["nodes"] = gl, nodes

    # 组间：每条边 Fisher z；全局指标和节点度：置换检验
    edges, perm = [], []
    if len(groups) >= 2:
        ctrl = groups[0]
        iu = np.triu_indices(len(regions), 1)
        for other in groups[1:]:
            z, p = fisher_z_compare(out["R"][other], out["N"][other], out["R"][ctrl], out["N"][ctrl])
            e = pd.DataFrame({"比较": f"{other} vs {ctrl}", "脑区 1": np.array(regions)[iu[0]],
                              "脑区 2": np.array(regions)[iu[1]],
                              f"r（{ctrl}）": out["R"][ctrl][iu].round(3), f"r（{other}）": out["R"][other][iu].round(3),
                              "Fisher z": z[iu].round(3), "p 值": p[iu]})
            e["q 值 (FDR 校正)"] = bh_fdr(e["p 值"].to_numpy())
            edges.append(e.sort_values("p 值"))
            tick(f"置换检验：{other} vs {ctrl}…")
            Xa, Xb = X[ctrl], X[other]
            allX = np.vstack([Xa, Xb]); na = len(Xa)
            obs_g = {gname: fast_metrics(adjacency(corr(Xx, params.method), t, params.positive_only))
                     for gname, Xx in ((ctrl, Xa), (other, Xb)) for t in [params.main_threshold]}
            obs = {m: obs_g[other][m] - obs_g[ctrl][m] for m in obs_g[ctrl]}
            Aa = adjacency(corr(Xa, params.method), params.main_threshold, params.positive_only)
            Ab = adjacency(corr(Xb, params.method), params.main_threshold, params.positive_only)
            obs_deg = Ab.sum(1) - Aa.sum(1)
            null, null_deg = {m: [] for m in obs}, []
            for _ in range(params.n_perm):
                idx = rng.permutation(len(allX))
                A1 = adjacency(corr(allX[idx[:na]], params.method), params.main_threshold, params.positive_only)
                A2 = adjacency(corr(allX[idx[na:]], params.method), params.main_threshold, params.positive_only)
                m1, m2 = fast_metrics(A1), fast_metrics(A2)
                for m in obs:
                    null[m].append(m2[m] - m1[m])
                null_deg.append(A2.sum(1) - A1.sum(1))
            for m in obs:
                nv = np.asarray(null[m])
                p_ = (1 + np.sum(np.abs(nv) >= abs(obs[m]) - 1e-12)) / (1 + len(nv))
                perm.append({"比较": f"{other} vs {ctrl}", "指标": m, "阈值": params.main_threshold,
                             ctrl: round(obs_g[ctrl][m], 4), other: round(obs_g[other][m], 4),
                             "差值": round(obs[m], 4), "置换检验 p 值": p_})
            nd = np.asarray(null_deg)
            p_deg = (1 + (np.abs(nd) >= np.abs(obs_deg) - 1e-12).sum(0)) / (1 + len(nd))
            out.setdefault("deg_diff", []).append(pd.DataFrame({
                "比较": f"{other} vs {ctrl}", "脑区缩写": regions, f"度（{ctrl}）": Aa.sum(1), f"度（{other}）": Ab.sum(1),
                "差值": obs_deg, "置换检验 p 值": p_deg, "q 值 (FDR 校正)": bh_fdr(p_deg)}).sort_values("置换检验 p 值"))
    out["edges_diff"] = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame()
    out["perm"] = pd.DataFrame(perm)
    out["deg_diff"] = pd.concat(out.get("deg_diff", []), ignore_index=True) if out.get("deg_diff") else pd.DataFrame()
    tick("完成")
    return out


# ---------------------------------------------------------------- 图
def plot_matrices(res, path, metric):
    from .stats import _plt
    plt = _plt()
    groups, regions = res["groups"], res["regions"]
    anc = _structures(); majors = [major_of(a, anc) for a in regions]
    k = len(groups) + (1 if len(groups) >= 2 else 0)
    fig, axes = plt.subplots(1, k, figsize=(5.2 * k + 1.5, 5.6), squeeze=False, layout="constrained")
    n = len(regions)
    bounds = [i for i in range(1, n) if majors[i] != majors[i - 1]]
    for i, g in enumerate(groups):
        ax = axes[0, i]
        imh = ax.imshow(res["R"][g], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{g}（{res['n_animals'][g]} 只）", fontsize=11)
    if len(groups) >= 2:
        ax = axes[0, -1]
        d = res["R"][groups[1]] - res["R"][groups[0]]
        imd = ax.imshow(d, cmap="PuOr_r", vmin=-2, vmax=2, interpolation="nearest")
        ax.set_title(f"{groups[1]} − {groups[0]}（r 之差）", fontsize=11)
        fig.colorbar(imd, ax=ax, shrink=0.75).set_label("r 之差")
    for ax in axes[0]:
        for b in bounds:
            ax.axhline(b - 0.5, color="k", lw=0.6); ax.axvline(b - 0.5, color="k", lw=0.6)
        starts = [0] + bounds; ends = bounds + [n]
        ax.set_xticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
        ax.set_xticklabels([MAJOR_ZH.get(majors[s], majors[s]) for s in starts], rotation=90, fontsize=8)
        ax.set_yticks(ax.get_xticks()); ax.set_yticklabels(ax.get_xticklabels(), fontsize=8)
    fig.colorbar(imh, ax=axes[0, :len(groups)].tolist(), shrink=0.75).set_label("相关系数 r")
    fig.suptitle(f"脑区间相关矩阵（{metric}，{n} 个脑区）", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_networks(res, params, path):
    from .stats import _plt
    plt = _plt()
    groups, regions = res["groups"], res["regions"]
    anc = _structures(); majors = [major_of(a, anc) for a in regions]
    n = len(regions)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = np.c_[np.cos(ang), np.sin(ang)]
    uniq = [m for m in MAJOR if m in majors]
    cmap = plt.get_cmap("tab20")
    col = {m: cmap(i) for i, m in enumerate(uniq)}
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 6.6), squeeze=False)
    nodes = res["nodes"]
    for gi, g in enumerate(groups):
        ax = axes[0, gi]
        A = adjacency(res["R"][g], params.main_threshold, params.positive_only)
        for i, j in zip(*np.nonzero(np.triu(A))):
            ax.plot(*pos[[i, j]].T, color="#888", lw=0.5, alpha=0.5, zorder=1)
        nd = nodes[(nodes["组别"] == g) & (nodes["阈值"] == params.main_threshold)].set_index("脑区缩写")
        deg = nd.loc[regions, "度"].to_numpy(); hub = nd.loc[regions, "枢纽脑区"].to_numpy() == "是"
        ax.scatter(pos[:, 0], pos[:, 1], s=12 + 9 * deg, c=[col[m] for m in majors], edgecolors=np.where(hub, "k", "none"),
                   linewidths=1.5, zorder=2)
        for i in np.nonzero(hub)[0]:
            ax.text(*(pos[i] * 1.1), regions[i], fontsize=8, ha="center", va="center", fontweight="bold")
        g_ = res["global"]; row = g_[(g_["组别"] == g) & (g_["阈值"] == params.main_threshold)].iloc[0]
        ax.set_title(f"{g}：{int(A.sum() / 2)} 条边，密度 {row['网络密度']:.2f}\n（r ≥ {params.main_threshold}，"
                     "点越大连接越多，黑圈 = 枢纽脑区）", fontsize=10)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[m], label=MAJOR_ZH.get(m, m)) for m in uniq]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(uniq), 6), fontsize=9, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_thresholds(res, params, path):
    from .stats import _plt
    plt = _plt()
    gl = res["global"]
    ms = ["网络密度", "全局效率", "平均聚类系数", "模块度"]
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.4))
    for ax, m in zip(axes, ms):
        for g in res["groups"]:
            s = gl[gl["组别"] == g].sort_values("阈值")
            ax.plot(s["阈值"], s[m], "o-", label=g)
        ax.set_title(m, fontsize=10); ax.set_xlabel("相关阈值 r")
    axes[0].legend(fontsize=9)
    fig.suptitle("不同阈值下的全局网络指标（各阈值趋势一致时结论更可信）", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- 入口
def run_network_analysis(project, out_dir, params: NetParams, region_sec=None, progress=None):
    """返回 (输出文件夹, 摘要文字, 图片路径列表)。"""
    from pathlib import Path
    from .export import build_tables, _write_xlsx
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if region_sec is None:
        if progress:
            progress(0, 1, "正在统计每张脑片的脑区结果…")
        skip, depth = project.export_skip_layers, project.export_max_depth
        project.export_skip_layers, project.export_max_depth = False, 0
        try:
            region_sec = build_tables(project, intensity=False).get("按脑区×脑片")
        finally:
            project.export_skip_layers, project.export_max_depth = skip, depth
    if region_sec is None or region_sec.empty:
        raise ValueError("还没有按脑区的结果。请先完成细胞识别和“⑥ 脑图谱配准”。")
    from .stats import check_samples
    used = set(region_sec["图片"].astype(str) + "/" + region_sec["脑片"].astype(str))
    errs, warns = check_samples({k: v for k, v in project.samples.items() if k in used})
    if errs:
        raise ValueError("\n".join(errs))
    animal = per_animal(region_sec, project.samples)
    if animal.empty:
        raise ValueError("还没有填写分组。请在表格中为每张脑片填写“动物编号”和“组别”。")
    params.metric = params.metric or metric_options(project.channel_names, grading=project.grading)[0]
    res = run_network(animal, params, progress)
    groups, regions = res["groups"], res["regions"]
    anc = _structures()
    st_names = animal.drop_duplicates("脑区缩写").set_index("脑区缩写")["脑区名称"]
    reg_tab = pd.DataFrame({"脑区缩写": regions, "脑区名称": [st_names.get(r, "") for r in regions],
                            "大脑区": [MAJOR_ZH.get(major_of(r, anc), "其他") for r in regions]})
    nmin = min(res["n_animals"].values())
    method = pd.DataFrame([
        ("指标", params.metric), ("半脑", params.hemisphere),
        ("相关方法", "Pearson" if params.method == "pearson" else "Spearman（秩相关）"),
        ("样本", "动物（同一动物的多张脑片先合并）；" + "，".join(f"{g} {n} 只" for g, n in res["n_animals"].items())),
        ("脑区", f"图谱截到第 {params.depth} 级后互不包含的最细脑区，面积 ≥ {params.min_area_mm2} mm²，"
                 f"每组至少 {params.min_coverage:.0%} 的动物有数据：共 {len(regions)} 个"),
        ("建网", f"{'r' if params.positive_only else '|r|'} ≥ 阈值的脑区对之间连边；阈值 "
                 + "、".join(str(t) for t in sorted(set(params.thresholds) | {params.main_threshold}))
                 + f"；主阈值 {params.main_threshold}"),
        ("枢纽脑区", "主阈值下度和介数中心性都在前 20% 的脑区；“留一法中为枢纽的比例”越接近 1 越稳定"),
        ("模块", "Louvain 社区划分（固定随机种子）"),
        ("置信区间", f"对动物有放回重抽样 {params.n_boot} 次（主阈值）"),
        ("组间比较", f"每条边：Fisher z 检验 + FDR；全局指标和每个脑区的度：打乱组别的置换检验 {params.n_perm} 次"),
        ("小样本提示", f"每组只有 {nmin} 只动物时，单个相关系数非常不稳定（n=8 时 r=0.7 的 95% CI 约 0.03–0.94），"
                       "边水平的组间差异通常不能通过 FDR。请重点报告：跨阈值一致的全局指标差异、留一法稳定的枢纽脑区；"
                       "单条边的结果只作为探索性描述。"),
    ], columns=["项目", "内容"])
    sheets = {"说明与方法": method, "全局指标": res["global"].round(4), "节点指标": res["nodes"],
              "组间_全局置换检验": res["perm"], "组间_脑区度差异": res["deg_diff"],
              "组间_相关差异(边)": res["edges_diff"].head(5000), "纳入的脑区": reg_tab}
    for g in groups:
        sheets[f"相关矩阵_{g}"[:31]] = pd.DataFrame(res["R"][g], index=regions, columns=regions).round(3).reset_index().rename(columns={"index": "脑区"})
    _write_xlsx({k: v for k, v in sheets.items() if v is not None and not v.empty}, out_dir / "网络分析.xlsx")
    figs = [out_dir / "网络图.png", out_dir / "相关矩阵.png", out_dir / "阈值曲线.png"]
    plot_networks(res, params, figs[0]); plot_matrices(res, figs[1], params.metric); plot_thresholds(res, params, figs[2])
    # 摘要
    lines = [f"{len(regions)} 个脑区，" + "，".join(f"{g} {n} 只" for g, n in res["n_animals"].items()) + "。"]
    gl = res["global"]; mt = params.main_threshold
    for g in groups:
        r = gl[(gl["组别"] == g) & (gl["阈值"] == mt)].iloc[0]
        nd = res["nodes"]; hubs = nd[(nd["组别"] == g) & (nd["阈值"] == mt) & (nd["枢纽脑区"] == "是")]
        stable = hubs[hubs["留一法中为枢纽的比例"] >= 0.8]["脑区缩写"].tolist()
        lines.append(f"{g}：r≥{mt} 时网络密度 {r['网络密度']:.2f}；稳定的枢纽脑区：{'、'.join(stable) if stable else '无'}")
    if not res["perm"].empty:
        sig = res["perm"][res["perm"]["置换检验 p 值"] < 0.05]
        lines.append("全局指标组间差异（置换检验 p<0.05）：" + ("、".join(sig["指标"]) if len(sig) else "无"))
        ne = int((res["edges_diff"]["q 值 (FDR 校正)"] < 0.05).sum())
        lines.append(f"相关系数有显著差异的脑区对（FDR q<0.05）：{ne} 对")
    lines += ["注意：" + w for w in warns]
    if nmin < 8:
        lines.append(f"注意：每组最少只有 {nmin} 只动物，网络结果不稳定，请谨慎解读。")
    return out_dir, "\n".join(lines), figs
