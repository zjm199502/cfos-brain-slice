"""通道间共标判定：把不同通道中位置重合的细胞合并为同一个细胞，得到每个细胞的阳性通道组合。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree


@dataclass
class ColocParams:
    max_dist_um: float = 6.0     # 两个细胞中心距离不超过此值即视为同一细胞
    use_contour: bool = True     # 或者：一个细胞的中心落在另一个细胞的轮廓内
    nucleus_by_intensity: bool = True  # 标记物细胞轮廓内 DAPI 信号足够强，也算有核（DAPI+）
    require_nucleus: bool = True       # 只统计有细胞核（DAPI+）的细胞；无核的信号碎片不计入

    def to_dict(self):
        return asdict(self)


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def nucleus_channel(channel_names) -> int | None:
    for i, n in enumerate(channel_names):
        if str(n).strip().upper() in ("DAPI", "HOECHST"):
            return i
    return None


def colocalize(df: pd.DataFrame, pixel_um0: float, params: ColocParams,
               nucleus_ch: int | None = None) -> pd.DataFrame:
    """输入：detect_section 的表格（可含多张脑片）。
    输出：在原表上增加 merged_id（合并后的细胞编号）和 combo（该细胞的阳性通道组合，如 "0+1"）。"""
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        df["merged_id"] = []
        df["combo"] = []
        return df
    uf = _UF(n)
    r0 = params.max_dist_um / pixel_um0
    search = max(r0, 20.0 / pixel_um0)
    conts = df["contour"].to_numpy() if params.use_contour and "contour" in df.columns else None
    paths = {}

    def inside(i, pt):
        """细胞 i 的轮廓是否包含点 pt。每个轮廓只建一次 Path（以前每次比较都重建）。"""
        pth = paths.get(i, False)
        if pth is False:
            c = conts[i]
            pth = paths[i] = MplPath(np.asarray(c, float)) if c is not None and len(c) > 2 else None
        return pth is not None and pth.contains_point(pt)

    for sec, g in df.groupby("section"):
        chans = sorted(g["channel"].unique())
        idx_by_ch = {c: g.index[g["channel"] == c].to_numpy() for c in chans}
        trees = {c: cKDTree(df.loc[idx_by_ch[c], ["x", "y"]].to_numpy()) for c in chans}
        for a, b in combinations(chans, 2):
            ia, ib = idx_by_ch[a], idx_by_ch[b]
            if len(ia) == 0 or len(ib) == 0:
                continue
            pa = df.loc[ia, ["x", "y"]].to_numpy()
            pb = df.loc[ib, ["x", "y"]].to_numpy()
            cands = trees[b].query_ball_point(pa, search)
            pairs = []  # (距离, a 下标, b 下标)
            for k, js in enumerate(cands):
                if not js:
                    continue
                d = np.hypot(*(pb[js] - pa[k]).T)
                for o in np.argsort(d):
                    j = js[o]
                    ok = d[o] <= r0
                    if not ok and conts is not None:
                        ok = inside(ib[j], pa[k]) or inside(ia[k], pb[j])
                    if ok:
                        pairs.append((d[o], k, j))
            # 一对一匹配：距离最近的优先，每个细胞最多与另一通道的一个细胞配对
            pairs.sort()
            used_a, used_b = set(), set()
            for _, k, j in pairs:
                if k in used_a or j in used_b:
                    continue
                used_a.add(k); used_b.add(j)
                uf.union(ia[k], ib[j])
    roots = np.array([uf.find(i) for i in range(n)])
    _, merged = np.unique(roots, return_inverse=True)
    df["merged_id"] = merged
    df["nuc_by_intensity"] = False
    col = f"mean_ch{nucleus_ch}" if nucleus_ch is not None else None
    if params.nucleus_by_intensity and col in df.columns:
        has_nuc = df.groupby("merged_id")["channel"].apply(lambda s: nucleus_ch in set(s))
        df["_has_nuc"] = df["merged_id"].map(has_nuc)
        for sec, g in df.groupby("section"):
            marker = g[g["channel"] != nucleus_ch]
            ref = marker.loc[marker["_has_nuc"], col]
            if len(ref) < 20:
                ref = g.loc[g["channel"] == nucleus_ch, col]
            if len(ref) < 20:
                continue
            thr = 0.5 * float(np.percentile(ref, 25))
            cand = marker[(~marker["_has_nuc"]) & (marker[col] >= thr)]
            df.loc[df["merged_id"].isin(cand["merged_id"]), "nuc_by_intensity"] = True
        df = df.drop(columns="_has_nuc")
    def _combo(g):
        chs = set(g["channel"])
        if nucleus_ch is not None and g["nuc_by_intensity"].any():
            chs.add(nucleus_ch)
        return "+".join(str(c) for c in sorted(chs))
    combo = df.groupby("merged_id")[["channel", "nuc_by_intensity"]].apply(_combo)
    df["combo"] = df["merged_id"].map(combo)
    return df


def merged_cells(df: pd.DataFrame, n_channels: int) -> pd.DataFrame:
    """每个合并后的细胞一行：位置、各通道是否阳性。"""
    if df.empty:
        dtypes = {"section": "object", "merged_id": "int64", "x": "float64", "y": "float64"}
        dtypes.update({f"pos_ch{c}": "bool" for c in range(n_channels)})
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()})
    g = df.groupby("merged_id")
    out = g.agg(section=("section", "first"), x=("x", "mean"), y=("y", "mean"))
    if "combo" in df.columns:
        chs = g["combo"].first().apply(lambda s: {int(x) for x in str(s).split("+") if x != ""})
    else:
        chs = g["channel"].apply(set)
    for c in range(n_channels):
        out[f"pos_ch{c}"] = chs.apply(lambda s, c=c: c in s)
    for col in ("hemisphere", "region_id"):
        if col in df.columns:
            out[col] = g[col].first()
    # 每个通道：该细胞在这个通道里被识别出的那一个对象的平均亮度和组织背景（用于强度分级）
    for c in range(n_channels):
        sub = df[df["channel"] == c]
        if f"mean_ch{c}" in sub.columns and len(sub):
            first = sub.groupby("merged_id").first()
            out[f"int_ch{c}"] = first[f"mean_ch{c}"].reindex(out.index)
            if "background" in sub.columns:
                bg = first["background"]
                # 手动补加的细胞没有背景值：用同一脑片其他细胞的背景
                sec_bg = sub.groupby("section")["background"].median()
                fill = first["section"].map(sec_bg)
                out[f"bg_ch{c}"] = bg.fillna(fill).reindex(out.index)
    return out.reset_index()


MIN_NUCLEI = 200   # 一个区域里 DAPI+ 少于这个数时，“X+ 占 DAPI+ 的比例”不稳定
DAPI_NOTE = "DAPI 可能偏少，比例偏高"


def ratio_column(marker: str, nucleus: str = "DAPI") -> str:
    """v0.8.3 推荐主指标的列名：X+ 占 DAPI+ 的比例 %。"""
    return f"{marker}+ 占 {nucleus}+ 的比例 %"


def nucleus_ratio_pairs(columns, channel_names) -> list[tuple[str, str, str]]:
    """[(比例列名, 分子列 = 有核的 X+（DAPI+X 共标）, 分母列 = DAPI+)]。没有 DAPI 通道时为空。"""
    nuc = nucleus_channel(channel_names)
    if nuc is None:
        return []
    nn = channel_names[nuc]
    cols = set(columns)
    out = []
    for c, nm in enumerate(channel_names):
        if c == nuc:
            continue
        a, b = sorted((c, nuc))
        num = f"{channel_names[a]}+{channel_names[b]} 共标"
        if num in cols and f"{nn}+" in cols:
            out.append((ratio_column(nm, nn), num, f"{nn}+"))
    return out


def add_nucleus_ratios(t: pd.DataFrame, channel_names, dense: set | None = None) -> pd.DataFrame:
    """在计数表上加“X+ 占 DAPI+ 的比例 %”（分子只算有核的 X+，分母是同一区域的 DAPI+），
    放在细胞数后面；再加一列提示：DAPI+ 不足 MIN_NUCLEI 个，或是细胞极密的脑区（dense，按脑区缩写）。"""
    pairs = nucleus_ratio_pairs(t.columns, channel_names)
    if not pairs or t.empty:
        return t
    t = t.copy()
    den_col = pairs[0][2]
    den = pd.to_numeric(t[den_col], errors="coerce")
    cols = list(t.columns)
    last = max(i for i, c in enumerate(cols) if c.endswith(" 共标") or c.endswith("+"))
    for k, (rc, num, _) in enumerate(pairs):
        v = (100.0 * pd.to_numeric(t[num], errors="coerce") / den.replace(0, np.nan)).round(2)
        t.insert(last + 1 + k, rc, v)
    note = np.where(den.fillna(0).to_numpy() < MIN_NUCLEI, f"{DAPI_NOTE}（{den_col} 不足 {MIN_NUCLEI} 个）", "")
    if dense and "脑区缩写" in t.columns:
        d = t["脑区缩写"].isin(dense).to_numpy()
        note = np.where(d & (note == ""), f"{DAPI_NOTE}（细胞密集区，DAPI 容易漏识别）", note)
    t.insert(last + 1 + len(pairs), "比例提示", note)
    return t


def summarize(cells: pd.DataFrame, channel_names: list[str], group_cols=("section",),
              require_nucleus: bool = False) -> pd.DataFrame:
    """按分组统计：每个通道的阳性细胞数，以及每对、每三个……通道的共标细胞数。
    require_nucleus=True 时只统计 DAPI+ 的细胞，并另列被排除的无核信号数量。"""
    n = len(channel_names)
    nuc = nucleus_channel(channel_names)
    group_cols = [c for c in group_cols if c in cells.columns]
    rows = []
    groups = cells.groupby(group_cols) if group_cols else [((), cells)]
    empty_groups = cells.empty and bool(group_cols)
    if empty_groups:
        groups = [(tuple(None for _ in group_cols), cells)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        excluded = None
        if require_nucleus and nuc is not None:
            keep = g[f"pos_ch{nuc}"].to_numpy(bool)
            excluded, g = g[~keep], g[keep]
        row = dict(zip(group_cols, key))
        for c in range(n):
            row[f"{channel_names[c]}+"] = int(g[f"pos_ch{c}"].sum())
        for c in range(n):
            gc = f"grade_ch{c}"
            if gc in g.columns:
                gv = g[gc].to_numpy()
                for lv, lab in ((1, "弱"), (2, "中"), (3, "强")):
                    row[f"{channel_names[c]}+ {lab}"] = int((gv == lv).sum())
        for k in range(2, n + 1):
            for combo in combinations(range(n), k):
                m = np.ones(len(g), bool)
                for c in combo:
                    m &= g[f"pos_ch{c}"].to_numpy(dtype=bool)
                row["+".join(channel_names[c] for c in combo) + " 共标"] = int(m.sum())
        if excluded is not None:
            ex = excluded
            for c in range(n):
                if c != nuc:
                    ne = int(ex[f"pos_ch{c}"].sum())
                    kept = row[f"{channel_names[c]}+"]
                    cn, nn = channel_names[c], channel_names[nuc]
                    row[f"{cn} 信号中无 {nn} 核的（未计入 {cn}+）"] = ne
                    row[f"{cn} 全部信号（含无 {nn} 核的）"] = kept + ne
                    row[f"{cn} 信号中有 {nn} 核的比例 %"] = round(100.0 * kept / (kept + ne), 1) if kept + ne else None
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.iloc[:0] if empty_groups else result
