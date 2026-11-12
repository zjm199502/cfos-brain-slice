"""把细胞分配到脑区，并按脑区（含上级脑区）汇总。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import registration as R
from .atlas import load_atlas
from .sections import Section
from .slide_io import Slide


def _thumb_geometry(slide: Slide, section: Section):
    lv = slide.level_for_pixel_size(R.THUMB_UM)
    ds = slide.downsample(lv)
    W = int(np.ceil(section.bbox0[2] / ds))
    H = int(np.ceil(section.bbox0[3] / ds))
    # 与 slide.read 的取整方式保持一致
    a = slide.read(lv, section.bbox0, [0])
    return ds, a.shape[1:], lv


def _excluded_mask(section: Section, shape, ds: float, exclusions, orientation: str) -> np.ndarray:
    """排除区域 + 组织轮廓以外，栅格化到“方向已校正”的缩略图坐标。True = 不计面积。"""
    from skimage.draw import polygon as skpoly
    from .detection import section_mask
    H, W = shape
    bad = ~section_mask(section, (H, W), ds) if section.contour0 else np.zeros((H, W), bool)
    x0, y0 = section.bbox0[0], section.bbox0[1]
    for poly in exclusions or []:
        a = np.asarray(poly, float)
        if len(a) < 3:
            continue
        rr, cc = skpoly((a[:, 1] - y0) / ds, (a[:, 0] - x0) / ds, (H, W))
        bad[rr, cc] = True
    return R.orient(bad, orientation)


def assign_regions(cells: pd.DataFrame, slide: Slide, sections: list[Section], results_dir, image_key,
                   exclusions=None):
    """cells: merged_cells 的输出（x, y 为第 0 层像素）。增加 region_index, hemisphere, 以及每张脑片的脑区面积表。"""
    cells = cells.copy()
    cells["region_index"] = 0
    cells["hemisphere"] = 0
    areas = []
    px0 = slide.pixel_size_um
    for s in sections:
        reg = R.Registration.from_dict(s.registration)
        if not reg.done or not s.enabled:
            continue
        ds, (H, W), _ = _thumb_geometry(slide, s)
        warp = None
        wp = R.warp_path(results_dir, image_key, s.name)
        if reg.nonlinear and wp.exists():
            w = np.load(wp)
            if w.shape[1:] == (H, W):
                warp = (w[0], w[1])
        lab, hemi = R.label_maps(reg, W, H, warp)
        lab = np.where(_excluded_mask(s, (H, W), ds, exclusions, reg.orientation), 0, lab)
        # 只统计组织内的面积
        sel = cells["section"] == s.name
        if sel.any():
            bx, by = s.bbox0[0], s.bbox0[1]
            px = (cells.loc[sel, "x"].to_numpy() - bx) / ds
            py = (cells.loc[sel, "y"].to_numpy() - by) / ds
            if reg.orientation == "r180":
                px, py = W - 1 - px, H - 1 - py
            ix = np.clip(np.round(px).astype(int), 0, W - 1)
            iy = np.clip(np.round(py).astype(int), 0, H - 1)
            cells.loc[sel, "region_index"] = lab[iy, ix]
            cells.loc[sel, "hemisphere"] = hemi[iy, ix]
        pix_mm2 = (ds * px0 / 1000.0) ** 2
        idx, cnt = np.unique(np.stack([lab.ravel(), hemi.ravel()], 1), axis=0, return_counts=True)
        for (ri, hm), n in zip(idx, cnt):
            if ri:
                areas.append({"section": s.name, "region_index": int(ri), "hemisphere": int(hm), "area_mm2": n * pix_mm2})
    return cells, pd.DataFrame(areas, columns=["section", "region_index", "hemisphere", "area_mm2"])


def _ancestors():
    st = load_atlas().structures
    id2idx = dict(zip(st["id"], st["index"]))
    anc = {}
    for _, r in st.iterrows():
        path = [int(x) for x in str(r["path"]).split("/") if x]
        anc[int(r["index"])] = [id2idx[i] for i in path if i in id2idx]
    return anc


def region_summary(cells: pd.DataFrame, areas: pd.DataFrame, channel_names, group_cols, require_nucleus=True):
    """返回按 (分组, 脑区, 半脑) 汇总的表，脑区包括所有上级脑区（每个细胞同时计入它的所有上级）。"""
    from .coloc import summarize
    st = load_atlas().structures.set_index("index")
    anc = _ancestors()
    c = cells[cells["region_index"] > 0].copy()
    c["anc"] = c["region_index"].map(anc)
    c = c.explode("anc").rename(columns={"anc": "region"})
    c["region"] = c["region"].astype(int)
    hemi_name = {1: "左", 2: "右", 0: "?"}
    out = []
    for both in (False, True):
        cc = c.copy()
        cc["半脑"] = "左右合计" if both else cc["hemisphere"].map(hemi_name)
        t = summarize(cc, channel_names, tuple(group_cols) + ("region", "半脑"), require_nucleus=require_nucleus)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    # 面积
    a = areas.copy()
    a["anc"] = a["region_index"].map(anc)
    a = a.explode("anc").rename(columns={"anc": "region"})
    a["region"] = a["region"].astype(int)
    a_h = a.assign(半脑=a["hemisphere"].map(hemi_name))
    a_b = a.assign(半脑="左右合计")
    gcols = [g for g in group_cols if g in a.columns]
    aa = pd.concat([a_h, a_b]).groupby(gcols + ["region", "半脑"], as_index=False)["area_mm2"].sum()
    t = t.merge(aa, on=gcols + ["region", "半脑"], how="left")
    t.insert(len(group_cols), "脑区缩写", t["region"].map(st["acronym"]))
    t.insert(len(group_cols) + 1, "脑区名称", t["region"].map(st["name"]))
    t.insert(len(group_cols) + 2, "层级", t["region"].map(lambda i: len(anc.get(i, [])) - 1))
    t["面积 (mm²)"] = t.pop("area_mm2").round(4)
    for nm in channel_names:
        col = f"{nm}+"
        if col in t.columns:
            t[f"{nm}+ 密度 (个/mm²)"] = (t[col] / t["面积 (mm²)"]).round(1)
    t = t.sort_values(list(group_cols) + ["region", "半脑"]).drop(columns=["region"])
    return t
