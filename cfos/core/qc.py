"""质控图：每张脑片一张 PNG，叠加识别到的细胞、排除区域和配准后的脑区边界，方便批量处理后快速翻看检查。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .project import Project
from .slide_io import open_slide

QC_UM = 4.0  # 质控图分辨率（µm/像素）
COLORS = {"DAPI": (80, 140, 255), "Green": (60, 220, 75), "Red": (255, 64, 64), "Far red": (255, 64, 255)}
FALLBACK = [(80, 140, 255), (60, 220, 75), (255, 64, 64), (255, 64, 255), (255, 200, 0)]


def _font(size):
    """返回 (字体, 是否支持中文)。"""
    for f in ("msyh.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc", "wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
              "/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(f, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:
        return ImageFont.load_default(), False


def section_qc_image(project: Project, im, sl, s, df) -> Image.Image:
    from . import registration as R
    from .atlas import region_boundaries
    names = project.channel_names
    lv = sl.level_for_pixel_size(QC_UM)
    ds = sl.downsample(lv)
    a = sl.read(lv, s.bbox0).astype(np.float32)
    H, W = a.shape[1:]
    rgb = np.zeros((H, W, 3), np.float32)
    for c in range(a.shape[0]):
        d = project.display.get(c)
        lo, hi = (d.low, d.high) if d is not None and d.high > d.low else np.percentile(a[c], [1, 99.7])
        x = np.clip((a[c] - lo) / max(hi - lo, 1e-6), 0, 1)
        col = np.array(COLORS.get(names[c] if c < len(names) else "", FALLBACK[c % 5]), np.float32) / 255
        rgb += x[..., None] * col * 0.8
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    x0, y0 = s.bbox0[0], s.bbox0[1]
    # 脑区边界
    reg = R.Registration.from_dict(s.registration)
    if reg.done:
        try:
            tlv = sl.level_for_pixel_size(R.THUMB_UM)
            tds = sl.downsample(tlv)
            th = sl.read(tlv, s.bbox0, [0])[0].shape
            warp = None
            wp = R.warp_path(project.results_dir, im.key, s.name)
            if reg.nonlinear and wp.exists():
                w = np.load(wp)
                if w.shape[1:] == th:
                    warp = (w[0], w[1])
            lab, _ = R.label_maps(reg, th[1], th[0], warp)
            b = R.orient(region_boundaries(lab), reg.orientation)  # orient 两次 = 回到原图方向
            b = np.asarray(Image.fromarray(b.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
            arr = np.asarray(img).copy()
            arr[b] = (arr[b] * 0.5 + 0.5 * 220).astype(np.uint8)
            img = Image.fromarray(arr); dr = ImageDraw.Draw(img)
        except Exception:  # noqa: BLE001
            pass
    # 细胞
    from .coloc import nucleus_channel
    nuc = nucleus_channel(names)
    if df is not None and not df.empty:
        g = df[df["section"] == s.name]
        r = max(1.5, 4.0 / (sl.pixel_size_um * ds))
        for c, nm in enumerate(names):
            if c == nuc and len(names) > 1:
                continue  # DAPI 细胞太密，圈出来会盖住图像；DAPI 本身在图上可见
            col = COLORS.get(nm, FALLBACK[c % 5])
            gc = g[g["channel"] == c]
            for x, y in zip((gc["x"].to_numpy() - x0) / ds, (gc["y"].to_numpy() - y0) / ds):
                dr.ellipse([x - r, y - r, x + r, y + r], outline=col)
    # 排除区域
    for poly in im.exclusions:
        pts = [((px - x0) / ds, (py - y0) / ds) for px, py in poly]
        if len(pts) >= 3:
            dr.line(pts + [pts[0]], fill=(255, 220, 0), width=3)
    for roi in im.rois:
        pts = [((px - x0) / ds, (py - y0) / ds) for px, py in roi["polygon"]]
        if len(pts) >= 3:
            dr.line(pts + [pts[0]], fill=(0, 255, 255), width=3)
            f, zh = _font(28)
            dr.text(pts[0], roi["name"] if zh else "ROI", fill=(0, 255, 255), font=f)
    # 标题
    cnt = "  ".join(f"{nm} {int((df[(df['section'] == s.name)]['channel'] == c).sum())}"
                    for c, nm in enumerate(names)) if df is not None and len(df) else "未识别"
    f, zh = _font(26)
    st = (reg.status if reg.done else "未配准") if zh else ("registered" if reg.done else "not registered")
    dr.rectangle([0, 0, W, 40], fill=(0, 0, 0))
    dr.text((8, 6), f"{im.key}  {s.name}   {cnt}   {st}", fill=(255, 255, 255), font=f)
    if nuc is not None and len(names) > 1:
        dr.text((8, H - 34), "圆圈 = 识别到的标记物阳性细胞（DAPI 未圈出）；白线 = 脑区边界；黄框 = 排除区域"
                if zh else "circles: marker+ cells; white: regions; yellow: excluded", fill=(255, 255, 255), font=f)
    return img


def export_qc(project: Project, out_dir, progress=None) -> Path:
    from .pipeline import drop_excluded
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(im, s) for im in project.images for s in im.sections if s.enabled]
    for k, (im, s) in enumerate(jobs):
        if progress:
            progress(k, len(jobs), f"正在生成质控图：{im.key} {s.name}")
        df = project.load_cells(im)
        df = drop_excluded(df, im) if df is not None else None
        sl = open_slide(im.path)
        try:
            section_qc_image(project, im, sl, s, df).save(out_dir / f"{im.key}_{s.name}.png", optimize=True)
        finally:
            sl.close()
    return out_dir
