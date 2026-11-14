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


INTENSITY_UM = 0.0     # 测量脑区平均荧光强度时读图的分辨率（µm/像素）；0 = 原始分辨率。
# 金字塔低分辨率层不是严格的平均值（示例图片上 2 µm 层的均值比原图差 2–4%），所以用原图。


def region_intensity_sums(slide: Slide, section: Section, lab_u: np.ndarray, hemi_u: np.ndarray, ds_lab: float,
                          target_um: float = INTENSITY_UM) -> dict:
    """按 (脑区, 半脑) 累加每个通道的原始灰度值和像素数（不是识别用的去卷积 / 减背景后的图）。
    lab_u / hemi_u：未旋转方向（与原图一致）的标签图，每个像素对应第 0 层 ds_lab 个像素；
    排除区域和组织外已经是 0。按行分块读图，大图也不会占太多内存。
    返回 {(region, hemi): (像素数, [各通道灰度和])}。"""
    lv = slide.level_for_pixel_size(target_um)
    ds = slide.downsample(lv)
    x0, y0, w, h = section.bbox0
    n_ch = len(slide.channels)
    Hl, Wl = lab_u.shape
    key = lab_u.astype(np.int64) * 3 + hemi_u.astype(np.int64)
    K = int(key.max()) + 1 if key.size else 1
    tot = np.zeros(K, np.float64)
    sums = np.zeros((n_ch, K), np.float64)
    # 按标签图的行分块：每块对应原图的若干行
    rows_per = max(1, int(np.ceil(1024 * ds / ds_lab)))
    for r0 in range(0, Hl, rows_per):
        r1 = min(Hl, r0 + rows_per)
        bb = (x0, y0 + r0 * ds_lab, w, (r1 - r0) * ds_lab)
        a = slide.read(lv, bb)
        if a.size == 0:
            continue
        _, Ha, Wa = a.shape
        # 读出的每个像素 → 标签图像素（读取的起点按 int(y/ds) 取整，误差 < 1 个像素）
        oy = int((y0 + r0 * ds_lab) / ds) * ds
        ox = int(x0 / ds) * ds
        iy = np.clip(((oy + (np.arange(Ha) + 0.5) * ds) - y0) / ds_lab, 0, Hl - 1).astype(int)
        ix = np.clip(((ox + (np.arange(Wa) + 0.5) * ds) - x0) / ds_lab, 0, Wl - 1).astype(int)
        k = key[iy][:, ix].ravel()
        m = k >= 3                      # 0 = 不在任何脑区
        k = k[m]
        tot += np.bincount(k, minlength=K)
        for c in range(n_ch):
            sums[c] += np.bincount(k, weights=a[c].ravel()[m].astype(np.float64), minlength=K)
    out = {}
    for kk in np.nonzero(tot)[0]:
        out[(int(kk // 3), int(kk % 3))] = (float(tot[kk]), sums[:, kk].tolist())
    return out


def _intensity_signature(slide: Slide, section: Section, reg, warp_file, exclusions) -> str:
    """决定脑区平均荧光强度结果的所有输入：原图文件、脑片方框、配准（含非线性贴合文件）、排除区域、z-stack 合成方式。
    任何一项改了，缓存就作废、重新计算。"""
    import hashlib, json, os
    from . import slide_io
    def stat(p):
        try:
            st = os.stat(p)
            return [st.st_size, st.st_mtime_ns]
        except OSError:
            return None
    d = {"v": 1, "file": stat(getattr(slide, "path", "")), "path": str(getattr(slide, "path", "")),
         "bbox0": list(section.bbox0), "contour": np.asarray(section.contour0 or [], float).round(1).tolist(),
         "reg": reg.to_dict(), "warp": stat(warp_file) if reg.nonlinear else None,
         "exclusions": [np.asarray(e, float).round(1).tolist() for e in (exclusions or [])],
         "z": getattr(slide_io, "Z_MODE", ""), "um": INTENSITY_UM, "n_ch": len(slide.channels)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def cached_intensity_sums(slide: Slide, section: Section, lab, hemi, ds, reg, results_dir, image_key,
                          exclusions, warp_file) -> dict:
    """region_intensity_sums 的结果按脑片缓存在 results/intensity/<图片>/<脑片>.pkl（v0.8.2）。
    配准、排除区域、方框都没改时直接用上次的结果，不再读原图；结果与重新计算完全相同。"""
    import pickle
    from pathlib import Path
    sig = _intensity_signature(slide, section, reg, warp_file, exclusions)
    f = Path(results_dir) / "intensity" / str(image_key) / f"{section.name}.pkl"
    try:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if d.get("signature") == sig:
            return d["sums"]
    except Exception:  # noqa: BLE001  没有缓存或缓存损坏时重新计算
        pass
    out = region_intensity_sums(slide, section, R.orient(lab, reg.orientation), R.orient(hemi, reg.orientation), ds)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"signature": sig, "sums": out}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(f)
    except OSError:
        pass
    return out


def assign_regions(cells: pd.DataFrame, slide: Slide, sections: list[Section], results_dir, image_key,
                   exclusions=None, intensity: bool = False):
    """cells: merged_cells 的输出（x, y 为第 0 层像素）。增加 region_index, hemisphere, 以及每张脑片的脑区面积表。
    intensity=True 时面积表还带每个通道的灰度和（sum_ch0…）与像素数（n_px），用于算脑区平均荧光强度。"""
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
        inten = cached_intensity_sums(slide, s, lab, hemi, ds, reg, results_dir, image_key, exclusions, wp) \
            if intensity else {}
        idx, cnt = np.unique(np.stack([lab.ravel(), hemi.ravel()], 1), axis=0, return_counts=True)
        for (ri, hm), n in zip(idx, cnt):
            if ri:
                row = {"section": s.name, "region_index": int(ri), "hemisphere": int(hm), "area_mm2": n * pix_mm2}
                if intensity:
                    npx, sm = inten.get((int(ri), int(hm)), (0.0, [0.0] * len(slide.channels)))
                    row["n_px"] = npx
                    for c, v in enumerate(sm):
                        row[f"sum_ch{c}"] = v
                areas.append(row)
    cols = ["section", "region_index", "hemisphere", "area_mm2"]
    if intensity:
        cols += ["n_px"] + [f"sum_ch{c}" for c in range(len(slide.channels))]
    return cells, pd.DataFrame(areas, columns=cols)


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
    icols = [c for c in a.columns if c == "n_px" or c.startswith("sum_ch")]
    aa = pd.concat([a_h, a_b]).groupby(gcols + ["region", "半脑"], as_index=False)[["area_mm2"] + icols].sum()
    cnt_cols = [c for c in t.columns if c not in tuple(group_cols) + ("region", "半脑")]
    t = t.merge(aa, on=gcols + ["region", "半脑"], how="outer" if icols else "left")
    if icols:   # 没有细胞、但有面积的脑区也列出来（平均荧光强度仍有意义），细胞数为 0
        for col in cnt_cols:
            v = t[col].fillna(0)
            t[col] = v.astype(int) if (v % 1 == 0).all() else v
        for g in group_cols:
            if g not in gcols and g in t.columns:
                t[g] = t[g].ffill().bfill()
    t.insert(len(group_cols), "脑区缩写", t["region"].map(st["acronym"]))
    t.insert(len(group_cols) + 1, "脑区名称", t["region"].map(st["name"]))
    t.insert(len(group_cols) + 2, "层级", t["region"].map(lambda i: len(anc.get(i, [])) - 1))
    t["面积 (mm²)"] = t.pop("area_mm2").round(4)
    for nm in channel_names:
        col = f"{nm}+"
        if col in t.columns:
            t[f"{nm}+ 密度 (个/mm²)"] = (t[col] / t["面积 (mm²)"]).round(1)
    if icols:
        # 平均荧光强度（原始灰度值）；相对值 = 除以同一张脑片整个组织（root，左右合计）的平均值，便于不同脑片比较
        npx = t.pop("n_px").replace(0, np.nan)
        root = st.index[st["acronym"] == "root"]
        root = int(root[0]) if len(root) else None
        for c, nm in enumerate(channel_names):
            sc = f"sum_ch{c}"
            if sc not in t.columns:
                continue
            mean = t.pop(sc) / npx
            t[f"{nm} 平均荧光强度"] = mean.round(1)
            if root is not None:
                rm = t.assign(_m=mean)[(t["region"] == root) & (t["半脑"] == "左右合计")]
                ref = rm.set_index(gcols)["_m"] if gcols else rm["_m"]
                if gcols:
                    base = t.set_index(gcols).index.map(ref)
                    t[f"{nm} 平均荧光强度（相对整张脑片）"] = (mean.to_numpy() / np.asarray(base, float)).round(3)
    t = t.sort_values(list(group_cols) + ["region", "半脑"]).drop(columns=["region"])
    return t
