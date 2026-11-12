"""⑨ 截图制图：把选定的矩形区域（截图框）按原图分辨率截取、放大，套用 ③ 中的亮度 / 颜色，
叠加细胞、脑区边界、排除区域，画上可编辑的比例尺，导出 PNG / TIFF / SVG。

坐标约定：截图框 {"name", "x", "y", "w", "h"} 为第 0 层像素；比例尺长度按 Slide.pixel_size_um 换算。
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .project import Project, ImageEntry
from .slide_io import Slide, open_slide
from .qc import COLORS, FALLBACK, _font

NICE_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
POSITIONS = ["右下", "左下", "右上", "左上"]
RES_CHOICES = [(0.0, "原始分辨率（最清楚，文件大）"), (0.5, "0.5 µm/像素"), (1.0, "1 µm/像素"),
               (2.0, "2 µm/像素"), (4.0, "4 µm/像素（缩略）")]
OUT_DIR = "截图"


@dataclass
class ScaleBar:
    length_um: float = 0.0        # 0 = 自动（约为图宽的 1/5，取整到 50/100/200/500 µm）
    thickness_px: int = 0         # 0 = 自动（图高的 1.2%，至少 3 像素）
    color: str = "白"             # 白 / 黑
    position: str = "右下"
    show_text: bool = True
    font_px: int = 0              # 0 = 自动
    margin_frac: float = 0.04     # 距边缘的比例


@dataclass
class FigureSettings:
    channels: list = field(default_factory=list)   # 显示的通道序号；空 = 全部
    separate: bool = False        # 合并图之外，每个通道再单独导出一张
    overlay_cells: bool = False
    overlay_regions: bool = False
    overlay_region_names: bool = False
    overlay_exclusions: bool = False
    inset: bool = False           # 左上角加定位小图（整张脑片 + 截图框位置）
    pixel_um: float = 0.0         # 导出分辨率，0 = 原始
    max_side: int = 8000          # 输出最长边上限（防止误选超大区域撑爆内存）
    formats: list = field(default_factory=lambda: ["png"])   # png / tif / svg
    scale: ScaleBar = field(default_factory=ScaleBar)
    fixed_size: bool = False      # 截图框固定为 fixed_w_um × fixed_h_um
    fixed_w_um: float = 500.0
    fixed_h_um: float = 500.0
    region_color: str = "白"       # 脑区轮廓线颜色：白 / 黑 / 黄 / 图谱颜色
    region_width_px: int = 0      # 轮廓线粗细，0 = 自动
    region_name_px: int = 0       # 脑区缩写字号，0 = 自动
    outline_only: bool = False    # 只画脑区轮廓（白底黑线），不显示荧光
    hemisphere_line: bool = False # 画左右半脑分界线

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "FigureSettings":
        d = dict(d or {})
        sc = d.pop("scale", {}) or {}
        st = FigureSettings(**{k: v for k, v in d.items() if k in FigureSettings.__dataclass_fields__})
        st.scale = ScaleBar(**{k: v for k, v in sc.items() if k in ScaleBar.__dataclass_fields__})
        return st


# ------------------------------------------------------------------ 小工具
def safe_name(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(s)).strip("_")
    return s or "ROI"


def auto_length_um(width_um: float) -> float:
    target = width_um / 5
    return float(min(NICE_UM, key=lambda v: abs(np.log(v) - np.log(max(target, 1e-6)))))


def channel_color(name: str, idx: int) -> tuple[int, int, int]:
    return tuple(COLORS.get(name, FALLBACK[idx % len(FALLBACK)]))


def clamp_roi(roi: dict, shape0_hw) -> tuple[int, int, int, int]:
    H, W = shape0_hw
    x, y = int(round(roi["x"])), int(round(roi["y"]))
    w, h = int(round(roi["w"])), int(round(roi["h"]))
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
    return x, y, w, h


def choose_level(sl: Slide, w: int, h: int, st: FigureSettings) -> int:
    lv = 0 if st.pixel_um <= 0 else sl.level_for_pixel_size(st.pixel_um)
    while lv < sl.n_levels - 1 and max(w, h) / sl.downsample(lv) > st.max_side:
        lv += 1
    return lv


def section_for(im: ImageEntry, x: float, y: float):
    """截图框中心所在的脑片。"""
    for s in im.sections:
        bx, by, bw, bh = s.bbox0
        if bx <= x < bx + bw and by <= y < by + bh:
            return s
    return None


# ------------------------------------------------------------------ 渲染
def render_rgb(project: Project, sl: Slide, bbox0, lv: int, channels: list[int]) -> np.ndarray:
    """按 ③ 的亮度/伽马和通道颜色合成 RGB（float 0–1）。"""
    names = project.channel_names
    a = sl.read(lv, bbox0, channels).astype(np.float32)
    H, W = a.shape[1:]
    rgb = np.zeros((H, W, 3), np.float32)
    for k, c in enumerate(channels):
        d = project.display.get(c)
        if d is not None and d.high > d.low:
            lo, hi, gm = d.low, d.high, d.gamma
        else:
            lo, hi = np.percentile(a[k], [0.5, 99.7]); gm = 1.0
        x = np.clip((a[k] - lo) / max(hi - lo, 1e-6), 0, 1)
        if gm != 1.0:
            x = x ** gm
        col = np.array(channel_color(names[c] if c < len(names) else "", c), np.float32) / 255
        rgb += x[..., None] * col
    return np.clip(rgb, 0, 1)


def _smooth_upscale(lab: np.ndarray, out_hw, cap: int = 2000) -> np.ndarray:
    """把低分辨率标签图放大到输出大小。先按“每个脑区的掩膜双线性插值后取最大”放大到中等尺寸，
    让脑区边界变成平滑的斜线而不是锯齿；再最近邻放大到最终大小。
    每个脑区只在自己的外接框里插值，脑区再多也很快。"""
    from scipy import ndimage as ndi
    H, W = out_hw
    h0, w0 = lab.shape
    sc = min(1.0, cap / max(H, W))
    mh, mw = max(1, int(round(H * sc))), max(1, int(round(W * sc)))
    if lab.max() == 0 or (mh <= h0 and mw <= w0):
        return np.asarray(Image.fromarray(lab.astype(np.int32)).resize((W, H), Image.NEAREST))
    fy, fx = mh / h0, mw / w0
    lab32 = lab.astype(np.int32)
    best = np.full((mh, mw), 0.5, np.float32)          # 低于 0.5 的位置保持背景（0）
    mid = np.zeros((mh, mw), np.int32)
    for v, sl_ in enumerate(ndi.find_objects(lab32), start=1):
        if sl_ is None:
            continue
        r0, r1 = max(0, sl_[0].start - 1), min(h0, sl_[0].stop + 1)
        c0, c1 = max(0, sl_[1].start - 1), min(w0, sl_[1].stop + 1)
        R0, R1 = int(round(r0 * fy)), int(round(r1 * fy))
        C0, C1 = int(round(c0 * fx)), int(round(c1 * fx))
        if R1 <= R0 or C1 <= C0:
            continue
        m = (lab32[r0:r1, c0:c1] == v).astype(np.float32)
        up = np.asarray(Image.fromarray(m).resize((C1 - C0, R1 - R0), Image.BILINEAR))
        b = best[R0:R1, C0:C1]
        sel = up > b
        b[sel] = up[sel]
        mid[R0:R1, C0:C1][sel] = v
    if (mh, mw) == (H, W):
        return mid
    return np.asarray(Image.fromarray(mid).resize((W, H), Image.NEAREST))


def region_overlay(project: Project, im: ImageEntry, sl: Slide, s, bbox0, out_hw):
    """截图范围内的脑区标签图（输出分辨率）。没有配准时返回 None。"""
    from . import registration as R
    from .atlas import load_atlas
    reg = R.Registration.from_dict(s.registration)
    if not reg.done:
        return None, {}
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
    lab = R.orient(lab, reg.orientation)           # 回到原图方向
    x, y, w, h = bbox0
    bx, by = s.bbox0[0], s.bbox0[1]
    c0, r0 = int((x - bx) / tds), int((y - by) / tds)
    c1, r1 = int(np.ceil((x + w - bx) / tds)), int(np.ceil((y + h - by) / tds))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(lab.shape[1], max(c1, c0 + 1)), min(lab.shape[0], max(r1, r0 + 1))
    crop = lab[r0:r1, c0:c1]
    # 截图框可能超出脑片范围：补 0
    full = np.zeros((max(1, int(np.ceil(h / tds))), max(1, int(np.ceil(w / tds)))), lab.dtype)
    oy, ox = max(0, int((by - y) / tds)), max(0, int((bx - x) / tds))
    hh, ww = min(crop.shape[0], full.shape[0] - oy), min(crop.shape[1], full.shape[1] - ox)
    if hh > 0 and ww > 0:
        full[oy:oy + hh, ox:ox + ww] = crop[:hh, :ww]
    up = _smooth_upscale(full, out_hw)
    st = load_atlas().structures
    names = dict(zip(st["index"].to_numpy(), st["acronym"].to_numpy()))
    return up, names


def draw_scale_bar(img: Image.Image, pix_um: float, sb: ScaleBar) -> dict:
    """在图上画比例尺，返回其几何信息（给 SVG 用）。"""
    W, H = img.size
    length_um = sb.length_um if sb.length_um > 0 else auto_length_um(W * pix_um)
    length_px = max(2, int(round(length_um / pix_um)))
    thick = sb.thickness_px if sb.thickness_px > 0 else max(3, int(round(H * 0.012)))
    margin = max(10, int(round(min(W, H) * sb.margin_frac)))
    font_px = sb.font_px if sb.font_px > 0 else max(14, int(round(H * 0.045)))
    color = (0, 0, 0) if sb.color == "黑" else (255, 255, 255)
    text = f"{length_um:g} µm" if length_um < 1000 else f"{length_um / 1000:g} mm"
    f, _ = _font(font_px)
    dr = ImageDraw.Draw(img)
    tw, th_ = (dr.textbbox((0, 0), text, font=f)[2:] if sb.show_text else (0, 0))
    right = sb.position.startswith("右")
    bottom = sb.position.endswith("下")
    x0 = W - margin - length_px if right else margin
    if bottom:
        y_bar = H - margin - thick
        y_text = y_bar - th_ - max(2, thick // 2)
    else:
        y_text = margin
        y_bar = (y_text + th_ + max(2, thick // 2)) if sb.show_text else margin
    dr.rectangle([x0, y_bar, x0 + length_px, y_bar + thick], fill=color)
    tx = x0 + (length_px - tw) / 2
    if sb.show_text:
        dr.text((tx, y_text), text, fill=color, font=f)
    return {"length_um": length_um, "length_px": length_px, "thickness_px": thick, "x": x0, "y": y_bar,
            "text": text, "text_x": tx, "text_y": y_text, "font_px": font_px, "color": sb.color}


def make_inset(project: Project, sl: Slide, s, bbox0, channels, width_px: int) -> Image.Image:
    """定位小图：整张脑片缩略图 + 红框标出截图位置。"""
    lv = sl.level_for_pixel_size(8.0)
    ds = sl.downsample(lv)
    rgb = render_rgb(project, sl, s.bbox0, lv, channels)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    x, y, w, h = bbox0
    bx, by = s.bbox0[0], s.bbox0[1]
    dr = ImageDraw.Draw(im)
    rect = [(x - bx) / ds, (y - by) / ds, (x + w - bx) / ds, (y + h - by) / ds]
    scale = width_px / im.width
    dr.rectangle(rect, outline=(255, 60, 60), width=max(2, int(round(3 / scale))))
    im = im.resize((width_px, max(1, int(round(im.height * scale)))), Image.BILINEAR)
    bordered = Image.new("RGB", (im.width + 4, im.height + 4), (255, 255, 255))
    bordered.paste(im, (2, 2))
    return bordered


REGION_COLORS = {"白": (235, 235, 235), "黑": (0, 0, 0), "黄": (255, 220, 0)}


def region_contours(lab: np.ndarray, cap: int = 1600, min_frac: float = 0.0) -> dict:
    """标签图 → 每个脑区的平滑轮廓线（输出像素坐标，(x, y)）。返回 {index: [N×2 数组, ...]}。
    在缩小到 cap 的标签图上找轮廓（快），先轻微平滑去掉锯齿，再按比例放大回输出大小。"""
    from scipy import ndimage as ndi
    from skimage import measure
    H, W = lab.shape
    sc = min(1.0, cap / max(H, W))
    if sc < 1.0:
        small = np.asarray(Image.fromarray(lab.astype(np.int32)).resize(
            (max(1, int(round(W * sc))), max(1, int(round(H * sc)))), Image.NEAREST))
    else:
        small = lab
    fy, fx = H / small.shape[0], W / small.shape[1]
    out = {}
    objs = ndi.find_objects(small)
    for v, sl_ in enumerate(objs, start=1):
        if sl_ is None:
            continue
        r0, r1 = max(0, sl_[0].start - 2), min(small.shape[0], sl_[0].stop + 2)
        c0, c1 = max(0, sl_[1].start - 2), min(small.shape[1], sl_[1].stop + 2)
        m = (small[r0:r1, c0:c1] == v)
        if m.sum() < max(4, min_frac * small.size):
            continue
        mf = ndi.gaussian_filter(np.pad(m.astype(np.float32), 1), 0.8)
        lines = []
        for c in measure.find_contours(mf, 0.5):
            if len(c) < 4:
                continue
            c = measure.approximate_polygon(c, 0.35)
            xy = np.stack([(c[:, 1] - 1 + c0 + 0.5) * fx, (c[:, 0] - 1 + r0 + 0.5) * fy], 1)
            lines.append(xy.astype(np.float32))
        if lines:
            out[v] = lines
    return out


def _region_layer(lab, acr, W, H, st: FigureSettings):
    """脑区轮廓 + 缩写的绘制数据（像素和 SVG 共用）。"""
    from scipy import ndimage as ndi
    from .atlas import load_atlas
    contours = region_contours(lab) if st.overlay_regions else {}
    if st.outline_only:
        color = (0, 0, 0)
    else:
        color = REGION_COLORS.get(st.region_color)
    lw = st.region_width_px if st.region_width_px > 0 else max(1, int(round(min(W, H) / 700)))
    rgb_of = {}
    if color is None:          # 图谱颜色
        stt = load_atlas().structures
        rgb_of = {int(i): (int(r), int(g), int(b)) for i, r, g, b in stt[["index", "r", "g", "b"]].to_numpy()}
    labels = []
    if st.overlay_region_names:
        fpx = st.region_name_px if st.region_name_px > 0 else max(12, int(round(min(W, H) * 0.022)))
        ids = [int(v) for v in np.unique(lab) if v > 0]
        if ids:
            ones = np.ones_like(lab, bool)
            # 名字放在脑区“最里面”的点（距离边界最远），比重心更不容易落到区外
            cents = ndi.center_of_mass(ones, lab, ids)
            sizes = ndi.sum(ones, lab, ids)
            placed = []        # 已放的文字框，重叠的小脑区名字跳过（大脑区优先）
            for v, (cy, cx), n in sorted(zip(ids, cents, sizes), key=lambda z: -z[2]):
                if n < (W * H) * 0.003:
                    continue
                t = str(acr.get(v, ""))
                if not t or t == "root":
                    continue
                iy, ix = int(min(max(cy, 0), H - 1)), int(min(max(cx, 0), W - 1))
                if lab[iy, ix] != v:
                    yy, xx = np.nonzero(lab == v)
                    k = np.argmin((yy - cy) ** 2 + (xx - cx) ** 2)
                    cy, cx = float(yy[k]), float(xx[k])
                hw, hh = 0.3 * fpx * len(t) + 2, 0.55 * fpx + 1
                if any(abs(cx - px_) < hw + qw and abs(cy - py_) < hh + qh for px_, py_, qw, qh in placed):
                    continue
                placed.append((cx, cy, hw, hh))
                labels.append((t, float(cx), float(cy), fpx))
    return {"contours": contours, "color": color, "rgb_of": rgb_of, "width": lw, "labels": labels,
            "text_color": (0, 0, 0) if st.outline_only else (240, 240, 240)}


def _draw_regions(img: Image.Image, layer: dict):
    dr = ImageDraw.Draw(img)
    for v, lines in layer["contours"].items():
        col = layer["color"] or layer["rgb_of"].get(int(v), (230, 230, 230))
        for xy in lines:
            pts = [tuple(p) for p in xy.tolist()]
            dr.line(pts + [pts[0]], fill=col, width=layer["width"], joint="curve")
    for t, cx, cy, fpx in layer["labels"]:
        f, _ = _font(fpx)
        tw, th_ = dr.textbbox((0, 0), t, font=f)[2:]
        px, py = cx - tw / 2, cy - th_ / 2
        if layer["text_color"] != (0, 0, 0):
            dr.text((px + 1, py + 1), t, fill=(0, 0, 0), font=f)
        dr.text((px, py), t, fill=layer["text_color"], font=f)


def render_roi(project: Project, im: ImageEntry, sl: Slide, roi: dict, st: FigureSettings,
               df=None, channels: list[int] | None = None, max_side: int | None = None,
               with_scale: bool = True, layers: bool = False) -> tuple[Image.Image, dict]:
    """渲染一个截图框。返回 (PIL 图, 元信息)。channels 为 None 时用设置里的通道。
    layers=True 时，meta["_layers"] 里另外给出不含脑区轮廓 / 定位小图的底图和各层数据（SVG 用）。"""
    names = project.channel_names
    n_ch = len(sl.channels)
    chans = channels if channels is not None else (st.channels or list(range(n_ch)))
    chans = [c for c in chans if c < n_ch] or list(range(n_ch))
    bbox0 = clamp_roi(roi, sl.level_shapes[0])
    x, y, w, h = bbox0
    st2 = st if max_side is None else FigureSettings.from_dict({**st.to_dict(), "max_side": max_side})
    lv = choose_level(sl, w, h, st2)
    ds = sl.downsample(lv)
    pix = sl.pixel_size_um * ds
    if st.outline_only:
        W0, H0 = int(np.ceil(w / ds)), int(np.ceil(h / ds))
        img = Image.new("RGB", (W0, H0), (255, 255, 255))
    else:
        rgb = render_rgb(project, sl, bbox0, lv, chans)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
    W, H = img.size
    dr = ImageDraw.Draw(img)
    meta = {"image": im.key, "roi": roi.get("name", ""), "x": x, "y": y, "w": w, "h": h,
            "pixel_um": pix, "width_um": W * pix, "height_um": H * pix, "level": lv, "out_w": W, "out_h": H,
            "channels": [] if st.outline_only else [names[c] if c < len(names) else str(c) for c in chans]}
    sec = section_for(im, x + w / 2, y + h / 2)
    # 细胞
    if st.overlay_cells and df is not None and not df.empty and not st.outline_only:
        from .coloc import nucleus_channel
        nuc = nucleus_channel(names)
        g = df[(df["x"] >= x) & (df["x"] < x + w) & (df["y"] >= y) & (df["y"] < y + h)]
        r = max(2.0, 5.0 / pix)
        lw = max(1, int(round(r / 4)))
        for c in chans:
            if c == nuc and len(chans) > 1:
                continue
            col = channel_color(names[c] if c < len(names) else "", c)
            col = tuple(min(255, int(v * 0.5 + 128)) for v in col)
            gc = g[g["channel"] == c]
            for cx, cy in zip((gc["x"].to_numpy() - x) / ds, (gc["y"].to_numpy() - y) / ds):
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=lw)
        meta["cells_in_roi"] = {names[c] if c < len(names) else str(c): int((g["channel"] == c).sum()) for c in chans}
    # 排除区域
    if st.overlay_exclusions:
        for poly in im.exclusions:
            pts = [((px - x) / ds, (py - y) / ds) for px, py in poly]
            if len(pts) >= 3:
                dr.line(pts + [pts[0]], fill=(255, 220, 0), width=max(2, int(round(min(W, H) / 400))))
    base = img.copy() if layers else None
    # 脑区边界 / 名称（画在细胞之上）
    reg_layer = None
    if (st.overlay_regions or st.overlay_region_names) and sec is not None:
        try:
            lab, acr = region_overlay(project, im, sl, sec, bbox0, (H, W))
        except Exception:  # noqa: BLE001
            lab = None
        if lab is not None:
            reg_layer = _region_layer(lab, acr, W, H, st)
            if st.hemisphere_line:
                reg_layer["midline"] = _midline(project, im, sl, sec, bbox0, (H, W))
            _draw_regions(img, reg_layer)
            if reg_layer.get("midline") is not None:
                pts = [tuple(p) for p in reg_layer["midline"].tolist()]
                ImageDraw.Draw(img).line(pts, fill=reg_layer["color"] or (230, 230, 230),
                                         width=reg_layer["width"] * 2)
            meta["regions_drawn"] = len(reg_layer["contours"])
        else:
            meta["regions_drawn"] = 0
            meta["regions_note"] = "这张脑片还没有配准，没有画脑区边界"
    # 定位小图
    inset = None
    if st.inset and sec is not None:
        try:
            ins = make_inset(project, sl, sec, bbox0, chans, max(80, int(W * 0.25)))
            m = max(6, int(round(min(W, H) * 0.02)))
            img.paste(ins, (m, m))
            inset = (ins, m, m)
            meta["inset"] = True
        except Exception:  # noqa: BLE001
            pass
    if layers:
        meta["_layers"] = {"base": base, "regions": reg_layer, "inset": inset}
    if with_scale:
        meta["scale_bar"] = draw_scale_bar(img, pix, st.scale)
    return img, meta


def _midline(project, im, sl, s, bbox0, out_hw):
    """左右半脑分界线（输出像素坐标）；取不到时为 None。"""
    from . import registration as R
    from .regions import _thumb_geometry
    reg = R.Registration.from_dict(s.registration)
    if not reg.done:
        return None
    tds, (th, tw), _ = _thumb_geometry(sl, s)
    warp = None
    wp = R.warp_path(project.results_dir, im.key, s.name)
    if reg.nonlinear and wp.exists():
        w = np.load(wp)
        if w.shape[1:] == (th, tw):
            warp = (w[0], w[1])
    lab, hemi = R.label_maps(reg, tw, th, warp)
    hemi = R.orient(np.where(lab > 0, hemi, 0), reg.orientation)
    x, y, w, h = bbox0
    H, W = out_hw
    pts = []
    for r in range(hemi.shape[0]):
        row = hemi[r]
        idx = np.nonzero((row[:-1] > 0) & (row[1:] > 0) & (row[:-1] != row[1:]))[0]
        if len(idx):
            c = idx[len(idx) // 2] + 1
            X = (s.bbox0[0] + c * tds - x) / w * W
            Y = (s.bbox0[1] + (r + 0.5) * tds - y) / h * H
            pts.append((X, Y))
    if len(pts) < 2:
        return None
    a = np.array(pts, float)
    from scipy import ndimage as ndi
    a[:, 0] = ndi.uniform_filter1d(a[:, 0], 5, mode="nearest")
    return a


# ------------------------------------------------------------------ 保存
def _hex(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


def _svg(img_no_bar: Image.Image, sb: dict, font_family="Arial, 'Microsoft YaHei', 'PingFang SC', sans-serif",
         layers: dict | None = None) -> str:
    """SVG：荧光图是嵌入的位图；脑区轮廓、脑区缩写、比例尺都是矢量（可在 Illustrator / Inkscape 里改颜色、粗细、位置）。"""
    bg = img_no_bar
    if layers is not None and layers.get("base") is not None:
        bg = layers["base"]
    buf = io.BytesIO(); bg.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    W, H = bg.size
    col = "#000000" if sb["color"] == "黑" else "#ffffff"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<image id="fluorescence" width="{W}" height="{H}" xlink:href="data:image/png;base64,{b64}"/>']
    reg = (layers or {}).get("regions")
    if reg:
        from .atlas import load_atlas
        acr = {}
        try:
            stt = load_atlas().structures
            acr = dict(zip(stt["index"].astype(int), stt["acronym"].astype(str)))
        except Exception:  # noqa: BLE001
            pass
        parts.append(f'<g id="region-outlines" fill="none" stroke-width="{reg["width"]}" stroke-linejoin="round">')
        for v, lines in reg["contours"].items():
            c = reg["color"] or reg["rgb_of"].get(int(v), (230, 230, 230))
            d = " ".join("M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in xy.tolist()) + " Z" for xy in lines)
            parts.append(f'<path id="r{int(v)}-{_xml(acr.get(int(v), ""))}" stroke="{_hex(c)}" d="{d}"/>')
        if reg.get("midline") is not None:
            d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in reg["midline"].tolist())
            parts.append(f'<path id="midline" stroke="{_hex(reg["color"] or (230, 230, 230))}" '
                         f'stroke-width="{reg["width"] * 2}" d="{d}"/>')
        parts.append("</g>")
        if reg["labels"]:
            tc = _hex(reg["text_color"])
            parts.append(f'<g id="region-names" font-family="{font_family}" text-anchor="middle" fill="{tc}">')
            for t, cx, cy, fpx in reg["labels"]:
                parts.append(f'<text x="{cx:.1f}" y="{cy + fpx * 0.35:.1f}" font-size="{fpx}">{_xml(t)}</text>')
            parts.append("</g>")
    ins = (layers or {}).get("inset")
    if ins is not None:
        ib = io.BytesIO(); ins[0].save(ib, format="PNG")
        parts.append(f'<image id="inset" x="{ins[1]}" y="{ins[2]}" width="{ins[0].width}" height="{ins[0].height}" '
                     f'xlink:href="data:image/png;base64,{base64.b64encode(ib.getvalue()).decode("ascii")}"/>')
    parts.append(f'<g id="scale-bar"><rect x="{sb["x"]}" y="{sb["y"]}" width="{sb["length_px"]}" '
                 f'height="{sb["thickness_px"]}" fill="{col}"/>')
    if sb.get("text"):
        parts.append(f'<text x="{sb["x"] + sb["length_px"] / 2:.1f}" y="{sb["text_y"] + sb["font_px"] * 0.85:.1f}" '
                     f'text-anchor="middle" font-family="{font_family}" font-size="{sb["font_px"]}" '
                     f'fill="{col}">{sb["text"]}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def _xml(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def save_outputs(img: Image.Image, img_no_bar: Image.Image | None, meta: dict, stem: Path, formats: list[str]) -> list[Path]:
    out = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    layers = meta.pop("_layers", None)

    def ext(e):  # 不用 with_suffix：图片名里可能带点（如 示例脑片.ome）
        return stem.with_name(stem.name + e)
    if "png" in formats:
        p = ext(".png"); img.save(p, dpi=(300, 300)); out.append(p)
    if "tif" in formats:
        p = ext(".tif")
        um = meta.get("pixel_um", 1.0)
        img.save(p, compression="tiff_lzw", resolution=10000.0 / um, resolution_unit=3)  # 像素/厘米
        out.append(p)
    if "svg" in formats and img_no_bar is not None and meta.get("scale_bar"):
        p = ext(".svg"); p.write_text(_svg(img_no_bar, meta["scale_bar"], layers=layers), encoding="utf-8"); out.append(p)
    stem.with_name(stem.name + "_参数.json").write_text(
        json.dumps({k: v for k, v in meta.items() if not k.startswith("_")}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    return out


def export_roi(project: Project, im: ImageEntry, sl: Slide, roi: dict, st: FigureSettings, df=None,
               out_dir: Path | None = None, base_name: str | None = None) -> list[Path]:
    """导出一个截图框的所有文件（合并图 + 可选各通道单独图）。"""
    out_dir = Path(out_dir or project.folder / OUT_DIR)
    names = project.channel_names
    n_ch = len(sl.channels)
    chans = [c for c in (st.channels or list(range(n_ch))) if c < n_ch] or list(range(n_ch))
    base = base_name or f"{im.key}_{safe_name(roi.get('name', 'ROI'))}"
    files = []
    if st.outline_only:
        jobs = [(chans, "轮廓")]
    else:
        jobs = [(chans, "合并" if len(chans) > 1 else safe_name(names[chans[0]] if chans[0] < len(names) else str(chans[0])))]
        if st.separate and len(chans) > 1:
            jobs += [([c], safe_name(names[c] if c < len(names) else str(c))) for c in chans]
    for cs, tag in jobs:
        need_svg = "svg" in st.formats
        img_nb, meta = render_roi(project, im, sl, roi, st, df=df, channels=cs, with_scale=False, layers=need_svg)
        img = img_nb.copy() if need_svg else img_nb
        meta["scale_bar"] = draw_scale_bar(img, meta["pixel_um"], st.scale)
        files += save_outputs(img, img_nb if need_svg else None, meta, out_dir / f"{base}_{tag}", st.formats)
    return files


def section_roi(s, margin_frac: float = 0.03) -> dict:
    """整张脑片的截图框（四周留一点边）。"""
    x, y, w, h = s.bbox0
    m = margin_frac * max(w, h)
    return {"name": s.name, "x": x - m, "y": y - m, "w": w + 2 * m, "h": h + 2 * m}


def export_sections(project: Project, images: list[ImageEntry], st: FigureSettings, sections: list[str] | None = None,
                    progress=None, cancel=None, out_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    """导出整张脑片的配准图（荧光 + 脑区轮廓 + 缩写 + 比例尺）。sections 为 None 时导出所有已配准的脑片。"""
    from . import registration as R
    st = FigureSettings.from_dict({**st.to_dict(), "overlay_regions": True, "inset": False})
    out_dir = Path(out_dir or project.folder / OUT_DIR / "配准图")
    todo = [(im, s) for im in images for s in im.sections
            if s.enabled and R.Registration.from_dict(s.registration).done and (sections is None or s.name in sections)]
    files, errors = [], []
    if not todo:
        return files, ["没有已配准的脑片。请先在“⑥ 脑图谱配准”中配准。"]
    for i, (im, s) in enumerate(todo):
        if cancel is not None and cancel():
            raise InterruptedError("已取消")
        if progress:
            progress(i, len(todo), f"{im.key}：{s.name} 配准图")
        try:
            sl = open_slide(im.path)
            try:
                df = project.load_cells(im) if st.overlay_cells else None
                files += export_roi(project, im, sl, section_roi(s), st, df=df, out_dir=out_dir,
                                    base_name=f"{im.key}_{safe_name(s.name)}_配准图")
            finally:
                sl.close()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{im.key} / {s.name}：{e}")
    if progress:
        progress(len(todo), len(todo), "完成")
    return files, errors


def export_images(project: Project, images: list[ImageEntry], st: FigureSettings, progress=None, cancel=None,
                  out_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    """批量导出：给定图片的全部截图框。返回 (文件列表, 出错说明列表)。"""
    files, errors = [], []
    todo = [(im, r) for im in images for r in im.figure_rois]
    for i, (im, roi) in enumerate(todo):
        if cancel is not None and cancel():
            raise InterruptedError("已取消")
        if progress:
            progress(i, len(todo), f"{im.key}：{roi.get('name', '')}")
        try:
            sl = open_slide(im.path)
            try:
                df = project.load_cells(im) if st.overlay_cells else None
                files += export_roi(project, im, sl, roi, st, df=df, out_dir=out_dir)
            finally:
                sl.close()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{im.key} / {roi.get('name', '')}：{e}")
    if progress:
        progress(len(todo), len(todo), "完成")
    return files, errors
