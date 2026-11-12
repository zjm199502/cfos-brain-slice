"""自动质量检查：给每张脑片打分，把可能有问题的脑片标出来，方便学生优先人工复查。

检查四项（都在低分辨率上算，每张脑片几秒钟）：
1. 配准重合度：最终脑区图（含非线性贴合和修正点）的脑轮廓与组织轮廓的 Dice。
2. 模糊：把组织分成约 100 µm 的小块，算每块的清晰度（高频细节 / 亮度）；
   比同一张脑片最清晰的那些块低很多的，算作模糊块。整张脑片都偏糊时，与项目中其他脑片比较。
3. 过曝：组织内各通道达到最大灰度（饱和）的像素比例。饱和的地方亮度已经失真，强度分级和亮度比较都不可靠。
4. 组织缺损：图谱认为有脑组织、实际却没有组织的面积比例（脑室除外；已画排除区域的部分不算）。
   未配准时，改为组织轮廓内的空洞比例。
另外，DAPI 细胞密度明显偏离项目中其他脑片时（常见于对焦失败、染色失败）也会提示。

结果缓存在 results/quality/<图片>.json，配准或方框改动后自动重算。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import ndimage as ndi

from . import registration as R

FOCUS_UM = 1.0        # 清晰度在约 1 µm/像素 上计算
TILE_UM = 100.0       # 小块边长
# 阈值：(注意, 可疑)
DICE_WARN, DICE_BAD = 0.85, 0.75
BLUR_WARN, BLUR_BAD = 10.0, 25.0          # 模糊面积 %
SHARP_WARN, SHARP_BAD = 0.6, 0.4          # 整张脑片清晰度 / 项目中位数
SAT_WARN, SAT_BAD = 0.5, 2.0              # 过曝像素 %
DAMAGE_WARN, DAMAGE_BAD = 8.0, 20.0       # 未处理的组织缺损 %
DENS_LOW, DENS_HIGH = 0.5, 2.0            # DAPI 密度 / 项目中位数
LEVELS = ("正常", "注意", "可疑")
VERSION = 2


def _signature(project, im, s) -> str:
    d = {"v": VERSION, "bbox": list(s.bbox0), "reg": s.registration, "exc": im.exclusions,
         "z": project.z_mode}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _path(project, im):
    d = project.results_dir / "quality"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{im.key}.json"


def load_cache(project, im) -> dict:
    p = _path(project, im)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        return {}


def _poly_mask(polys, x0, y0, ds, shape):
    from skimage.draw import polygon as _poly
    m = np.zeros(shape, bool)
    for pts in polys:
        pts = np.asarray(pts, float)
        if len(pts) < 3:
            continue
        rr, cc = _poly((pts[:, 1] - y0) / ds, (pts[:, 0] - x0) / ds, shape)
        m[rr, cc] = True
    return m


def _ventricle_indices():
    from .atlas import load_atlas
    st = load_atlas().structures
    vs = st[st["path"].astype(str).str.startswith("997/73")]
    return vs["index"].to_numpy(int)


# ------------------------------------------------------------------ 各项指标
def registration_metrics(project, im, sl, s, nuc, thumb=None):
    """返回 (dice 或 None, 组织缺损 %)。在配准缩略图（8 µm）上算。"""
    img, mask, ds = thumb or R.section_thumbnail(sl, s, nuc)
    reg = R.Registration.from_dict(s.registration)
    exc = _poly_mask(im.exclusions, s.bbox0[0], s.bbox0[1], ds, mask.shape)
    if not reg.done:
        filled = ndi.binary_fill_holes(mask)
        holes = filled & ~mask & ~exc
        return None, round(100.0 * holes.sum() / max(filled.sum(), 1), 2)
    mo = R.orient(mask, reg.orientation)
    H, W = mo.shape
    warp = None
    wp = R.warp_path(project.results_dir, im.key, s.name)
    if reg.nonlinear and wp.exists():
        w = np.load(wp)
        if w.shape[1:] == (H, W):
            warp = (w[0], w[1])
    lab, _ = R.label_maps(reg, W, H, warp)
    brain = lab > 0
    d = R.dice(brain, mo)
    vent = np.isin(lab, _ventricle_indices())
    # 脑室旁一圈也常常是空的（切片时脑室会张开），稍微放宽
    vent = ndi.binary_dilation(vent, iterations=2)
    missing = brain & ~vent & ~ndi.binary_dilation(mo, iterations=1) & ~R.orient(exc, reg.orientation)
    dmg = 100.0 * missing.sum() / max((brain & ~vent).sum(), 1)
    return round(float(d), 3), round(float(dmg), 2)


def _tile_sharpness(x, tissue, t):
    """x: 单通道图（浮点）；返回每个组织小块的清晰度（高频能量 / 亮度）。向量化计算，整张脑片约 1 秒。"""
    hp = x - ndi.gaussian_filter(x, 2.0, truncate=2.5)
    H, W = (x.shape[0] // t) * t, (x.shape[1] // t) * t
    if H == 0 or W == 0:
        return np.zeros(0)

    def blocks(a):
        return a[:H, :W].reshape(H // t, t, W // t, t).swapaxes(1, 2).reshape(H // t, W // t, t * t)
    m = blocks(tissue).astype(np.float32)
    n = m.sum(-1)
    ok = n >= 0.7 * t * t
    h = blocks(hp) * m
    mean_h = h.sum(-1) / np.maximum(n, 1)
    var_h = (h * h).sum(-1) / np.maximum(n, 1) - mean_h ** 2
    b = (blocks(x) * m).sum(-1) / np.maximum(n, 1)
    ok &= b > 0
    return (np.sqrt(np.maximum(var_h[ok], 0)) / b[ok]).astype(float)


def image_metrics(project, sl, s, nuc, thumb=None):
    """返回 (模糊面积 %, 整体清晰度, {通道: 过曝 %})。在约 1 µm/像素 上算。"""
    lv = sl.level_for_pixel_size(FOCUS_UM)
    ds = sl.downsample(lv)
    pix = sl.pixel_size_um * ds
    a = sl.read(lv, s.bbox0)
    # 组织掩膜：在低分辨率上算再放大，快且稳定
    _, mask, tds = thumb or R.section_thumbnail(sl, s, nuc)
    from PIL import Image
    tissue = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(a.shape[:0:-1], Image.NEAREST)) > 0
    tissue = ndi.binary_erosion(tissue, iterations=int(20 / pix))  # 去掉组织边缘（边缘天然锐利/发亮）
    sat = {}
    names = project.channel_names
    for c in range(a.shape[0]):
        x = a[c]
        vmax = np.iinfo(x.dtype).max if np.issubdtype(x.dtype, np.integer) else None
        top = x.max()
        # 12 / 14 位相机存在 16 位文件里时，饱和值是 4095 / 16383
        cands = [v for v in (255, 4095, 16383, 65535) if vmax is None or v <= vmax]
        full = next((v for v in cands if top == v), None)
        frac = float((x[tissue] >= full).mean() * 100) if full is not None and tissue.any() else 0.0
        sat[names[c] if c < len(names) else f"通道{c + 1}"] = round(frac, 3)
    c = nuc if nuc is not None else 0
    t = max(16, int(TILE_UM / pix))
    sh = _tile_sharpness(a[c].astype(np.float32), tissue, t)
    if len(sh) < 5:
        return None, None, sat
    ref = np.percentile(sh, 90)
    blur = float((sh < 0.5 * ref).mean() * 100)
    return round(blur, 1), float(np.median(sh)), sat


def section_quality(project, im, sl, s) -> dict:
    from .coloc import nucleus_channel
    nuc = nucleus_channel(project.channel_names)
    thumb = R.section_thumbnail(sl, s, nuc)
    d, dmg = registration_metrics(project, im, sl, s, nuc, thumb)
    blur, sharp, sat = image_metrics(project, sl, s, nuc, thumb)
    return {"signature": _signature(project, im, s), "dice": d, "damage": dmg, "blur": blur,
            "sharpness": sharp, "saturation": sat}


def check_image(project, im, sl=None, redo=False, progress=None) -> dict:
    """计算（或从缓存读取）一张图片所有脑片的指标。返回 {脑片名: 指标}。"""
    from .slide_io import open_slide
    cache = load_cache(project, im)
    secs = [s for s in im.sections if s.enabled]
    todo = [s for s in secs if redo or cache.get(s.name, {}).get("signature") != _signature(project, im, s)]
    if todo:
        own = sl is None
        sl = sl or open_slide(im.path)
        try:
            for k, s in enumerate(todo):
                if progress:
                    progress(k, len(todo), f"质量检查：{im.key} {s.name}")
                try:
                    cache[s.name] = section_quality(project, im, sl, s)
                except Exception as e:  # noqa: BLE001
                    from .log import friendly
                    cache[s.name] = {"signature": _signature(project, im, s), "error": friendly(e)}
        finally:
            if own:
                sl.close()
        _path(project, im).write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {s.name: cache.get(s.name, {}) for s in secs}


# ------------------------------------------------------------------ 评分
def _dapi_density(project):
    """{(图片, 脑片): DAPI 细胞数 / 组织面积 mm²}，没有识别结果时为空。"""
    from .coloc import nucleus_channel
    nuc = nucleus_channel(project.channel_names)
    out = {}
    if nuc is None:
        return out
    from .pipeline import drop_excluded
    for im in project.images:
        df = project.load_cells(im)
        if df is None or df.empty:
            continue
        df = drop_excluded(df, im)
        n = df[df["channel"] == nuc].groupby("section").size()
        for s in im.sections:
            if s.enabled and s.name in n.index and len(s.contour0) >= 3:
                pts = np.asarray(s.contour0, float)
                area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
                area_mm2 = area * (project_pixel_um(project, im) / 1000) ** 2
                if area_mm2 > 0:
                    out[(im.key, s.name)] = n[s.name] / area_mm2
    return out


def project_pixel_um(project, im) -> float:
    if getattr(im, "_px", None) is None:
        from .slide_io import open_slide
        sl = open_slide(im.path); im._px = sl.pixel_size_um; sl.close()
    return im._px


def grade(m: dict, sharp_rel=None, dens_rel=None) -> tuple[int, str, list[str]]:
    """返回 (分数 0–100, 等级, 问题列表)。"""
    if m.get("error"):
        return 0, "可疑", [f"无法检查：{m['error']}"]
    score, lvl, why = 100.0, 0, []

    def hit(bad, warn, txt_bad, txt_warn, pen):
        nonlocal score, lvl
        if bad:
            score -= pen; lvl = max(lvl, 2); why.append(txt_bad)
        elif warn:
            score -= pen / 3; lvl = max(lvl, 1); why.append(txt_warn)

    d = m.get("dice")
    if d is not None:
        hit(d < DICE_BAD, d < DICE_WARN, f"配准不准（重合度 {d:.2f}）", f"配准重合度偏低（{d:.2f}）", 40)
    b = m.get("blur")
    if b is not None:
        hit(b >= BLUR_BAD, b >= BLUR_WARN, f"大片模糊（{b:.0f}% 面积）", f"局部模糊（{b:.0f}% 面积）", 30)
    if sharp_rel is not None:
        hit(sharp_rel < SHARP_BAD, sharp_rel < SHARP_WARN, "整张脑片明显比其他脑片模糊",
            "整张脑片比其他脑片稍模糊", 30)
    for nm, v in (m.get("saturation") or {}).items():
        hit(v >= SAT_BAD, v >= SAT_WARN, f"{nm} 过曝（{v:.1f}% 像素饱和）", f"{nm} 少量过曝（{v:.2f}%）", 20)
    g = m.get("damage")
    if g is not None:
        hit(g >= DAMAGE_BAD, g >= DAMAGE_WARN, f"组织缺损/撕裂多（{g:.0f}%），建议画排除区域或不分析",
            f"有组织缺损（{g:.0f}%）", 25)
    if dens_rel is not None:
        hit(dens_rel < DENS_LOW or dens_rel > DENS_HIGH, False,
            f"DAPI 细胞密度异常（为其他脑片的 {dens_rel:.1f} 倍），可能识别失败", "", 25)
    return int(max(0, round(score))), LEVELS[lvl], why


def quality_table(project, progress=None, redo=False):
    """返回 DataFrame：每张启用的脑片一行。需要时会计算（较慢的部分有缓存）。"""
    import pandas as pd
    jobs = [im for im in project.images if any(s.enabled for s in im.sections)]
    allm = {}
    for k, im in enumerate(jobs):
        rep = (lambda i, n, t, k=k: progress(k * 100 + int(100 * i / max(n, 1)), len(jobs) * 100, t)) if progress else None
        for name, m in check_image(project, im, redo=redo, progress=rep).items():
            allm[(im.key, name)] = m
    sharp = [m["sharpness"] for m in allm.values() if m.get("sharpness")]
    med_sharp = float(np.median(sharp)) if len(sharp) >= 3 else None
    dens = _dapi_density(project)
    med_dens = float(np.median(list(dens.values()))) if len(dens) >= 3 else None
    rows = []
    for (key, name), m in allm.items():
        sr = m["sharpness"] / med_sharp if med_sharp and m.get("sharpness") else None
        dr = dens[(key, name)] / med_dens if med_dens and (key, name) in dens else None
        score, lvl, why = grade(m, sr, dr)
        sat = m.get("saturation") or {}
        rows.append({"图片": key, "脑片": name, "质量": lvl, "质量分": score,
                     "需要检查的问题": "；".join(why),
                     "配准重合度": m.get("dice"), "模糊面积 %": m.get("blur"),
                     "清晰度（相对中位数）": round(sr, 2) if sr is not None else None,
                     "最大过曝 %": max(sat.values()) if sat else None,
                     "组织缺损 %": m.get("damage"),
                     "DAPI 密度（相对中位数）": round(dr, 2) if dr is not None else None})
    df = pd.DataFrame(rows)
    summ = {f"{r['图片']}/{r['脑片']}": {"level": r["质量"], "why": r["需要检查的问题"],
                                       "signature": allm[(r["图片"], r["脑片"])].get("signature")} for r in rows}
    _summary_path(project).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    return df


def _summary_path(project):
    return project.results_dir / "quality" / "_汇总.json"


def cached_levels(project) -> dict:
    """只读上次质量检查的结果、不计算：{(图片, 脑片): (等级, 问题)}，给界面表格快速显示。
    脑片在检查之后改过（配准、方框、排除区域）的不返回。"""
    try:
        summ = json.loads(_summary_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for im in project.images:
        for s in im.sections:
            r = summ.get(f"{im.key}/{s.name}")
            if s.enabled and r and r.get("signature") == _signature(project, im, s):
                out[(im.key, s.name)] = (r["level"], r["why"])
    return out
