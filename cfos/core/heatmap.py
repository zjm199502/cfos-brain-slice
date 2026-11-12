"""脑区热图：把每个脑区的数值（各组平均密度、组间倍数变化）画回 CCFv3 冠状切面上。

每个像素属于图谱中最细的脑区；如果这个脑区没有统计结果（例如太小被排除，或导出时只到某一级），
就用它最近的、有结果的上级脑区的数值，所以图上不会出现空洞。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .atlas import load_atlas, region_boundaries


def ap_to_index(ap_mm: float) -> int:
    """相对 Bregma 的前后位置（mm，前为正）→ 图谱 AP 下标（与 atlas.Plane.ap_mm 的换算一致）。"""
    return int(round(527 - 332 - ap_mm / 0.025))


def value_lut(values: dict, keep_nan: bool = False) -> np.ndarray:
    """values: {脑区缩写: 数值}。返回长度 n+1 的查找表（下标 = 结构 index），没有数值的继承上级。
    keep_nan=True 时，数值为 NaN 的脑区也算“有结果”（显示为灰色），不继承上级。"""
    st = load_atlas().structures
    id2idx = dict(zip(st["id"], st["index"]))
    acr2idx = dict(zip(st["acronym"], st["index"]))
    own = {acr2idx[a]: v for a, v in values.items()
           if a in acr2idx and v is not None and (keep_nan or np.isfinite(v))}
    lut = np.full(int(st["index"].max()) + 1, np.nan)
    for _, r in st.iterrows():
        path = [id2idx[int(x)] for x in str(r["path"]).split("/") if x and int(x) in id2idx]
        for idx in reversed(path):  # 从自己往上找
            if idx in own:
                lut[int(r["index"])] = own[idx]
                break
    return lut


def coronal_slice(ap_mm: float):
    at = load_atlas()
    a = int(np.clip(ap_to_index(ap_mm), 0, at.annotation.shape[0] - 1))
    return at.annotation[a], at.hemispheres[a]


def default_ap_levels(project, n=6) -> list[float]:
    """优先用实际配准过的脑片所在的前后位置（四舍五入到 0.25 mm 后去重），否则在 +2.5 ~ -3.5 mm 之间均匀取。"""
    from . import registration as R
    aps = []
    for im in project.images:
        for s in im.sections:
            if s.enabled and s.registration.get("plane"):
                try:
                    aps.append(round(R.adjusted_plane(R.Registration.from_dict(s.registration)).ap_mm() * 4) / 4)
                except Exception:  # noqa: BLE001
                    pass
    aps = sorted(set(aps), reverse=True)
    if len(aps) > n:
        aps = [aps[int(i)] for i in np.linspace(0, len(aps) - 1, n)]
    return aps or list(np.round(np.linspace(2.5, -3.5, n), 2))


def _rgba(img, inside, cmap, vmin, vmax):
    """脑外透明（白底），脑内无数据为浅灰。"""
    x = np.clip((np.nan_to_num(img, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = cmap(x)
    rgba[inside & ~np.isfinite(img)] = (0.85, 0.85, 0.85, 1)
    rgba[~inside] = (1, 1, 1, 1)
    return rgba


def heatmap_figure(gsum: pd.DataFrame, comp: pd.DataFrame, metric: str, control: str, ap_levels, path,
                   alpha=0.05):
    """每行一个前后位置；每列一个组的平均值（同一色标），最后一列为与对照组相比的 log2 倍数变化（只给显著的脑区上色）。"""
    from .stats import _plt
    plt = _plt()
    groups = [control] + sorted(g for g in gsum["组别"].unique() if g != control)
    luts = [value_lut(dict(zip(gsum[gsum["组别"] == g]["脑区缩写"], gsum[gsum["组别"] == g]["平均值"])))
            for g in groups]
    vals = np.concatenate([l[np.isfinite(l)] for l in luts]) if luts else np.array([0.0])
    vmax = float(np.nanpercentile(vals, 98)) if len(vals) else 1.0
    comps = []
    for g in groups[1:] if comp is not None and not comp.empty else []:
        c = comp[comp["比较"] == f"{g} vs {control}"]
        fc = np.where(c["q 值 (FDR 校正)"] < alpha, c["log2 倍数变化"], np.nan)
        comps.append((g, value_lut(dict(zip(c["脑区缩写"], fc)), keep_nan=True)))
    ncol = len(groups) + len(comps)
    fig, axes = plt.subplots(len(ap_levels), ncol, figsize=(2.6 * ncol + 0.8, 2.1 * len(ap_levels) + 0.6),
                             squeeze=False)
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#e8e8e8")
    dcmap = plt.get_cmap("RdBu_r").copy(); dcmap.set_bad("#e8e8e8")
    im1 = im2 = None
    for ri, ap in enumerate(ap_levels):
        lab, hemi = coronal_slice(ap)
        inside = lab > 0
        edge = region_boundaries(lab) & inside
        ys, xs = np.nonzero(inside)
        crop = (slice(max(ys.min() - 3, 0), ys.max() + 4), slice(max(xs.min() - 3, 0), xs.max() + 4)) if len(ys) else (slice(None), slice(None))
        for ci in range(ncol):
            ax = axes[ri, ci]; ax.axis("off")
            if ci < len(groups):
                img = luts[ci][lab].astype(float)
                img[~inside] = np.nan
                im1 = ax.imshow(np.ma.masked_invalid(img[crop]), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
                ax.imshow(_rgba(img, inside, cmap, 0, vmax)[crop], interpolation="nearest")
                title = groups[ci]
            else:
                g, lut = comps[ci - len(groups)]
                img = lut[lab].astype(float)
                img[~inside] = np.nan
                im2 = ax.imshow(np.ma.masked_invalid(img[crop]), cmap=dcmap, vmin=-2, vmax=2, interpolation="nearest")
                ax.imshow(_rgba(img, inside, dcmap, -2, 2)[crop], interpolation="nearest")
                title = f"{g} / {control}\n(log2 倍数，仅显著)"
            e = np.zeros(lab.shape + (4,)); e[edge] = (1, 1, 1, 0.5)
            ax.imshow(e[crop], interpolation="nearest")
            if ri == 0:
                ax.set_title(title, fontsize=10)
        axes[ri, 0].text(-0.08, 0.5, f"Bregma\n{ap:+.2f} mm", transform=axes[ri, 0].transAxes,
                         ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.86, top=1 - 0.9 / fig.get_figheight(), bottom=0.03, wspace=0.05, hspace=0.12)
    if im1 is not None:
        cax = fig.add_axes([0.88, 0.55, 0.015, 0.35]); fig.colorbar(im1, cax=cax).set_label(metric, fontsize=8)
    if im2 is not None:
        cax = fig.add_axes([0.88, 0.1, 0.015, 0.35]); fig.colorbar(im2, cax=cax).set_label("log2 倍数变化", fontsize=8)
    fig.suptitle(f"{metric}（灰色 = 无数据或不显著）", fontsize=11, y=0.995)
    fig.savefig(path, dpi=130); plt.close(fig)
