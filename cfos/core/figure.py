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
    让脑区边界变成平滑的斜线而不是锯齿；再最近邻放大到最终大小。"""
    H, W = out_hw
    ids = np.unique(lab)
    sc = min(1.0, cap / max(H, W))
    mh, mw = max(1, int(round(H * sc))), max(1, int(round(W * sc)))
    if len(ids) <= 1 or len(ids) > 96 or (mh <= lab.shape[0] and mw <= lab.shape[1]):
        return np.asarray(Image.fromarray(lab.astype(np.int32)).resize((W, H), Image.NEAREST))
    best = np.full((mh, mw), -1.0, np.float32)
    mid = np.zeros((mh, mw), np.int32)
    for v in ids:
        m = np.asarray(Image.fromarray((lab == v).astype(np.float32)).resize((mw, mh), Image.BILINEAR))
        sel = m > best
        mid[sel] = int(v); best[sel] = m[sel]
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


def render_roi(project: Project, im: ImageEntry, sl: Slide, roi: dict, st: FigureSettings,
               df=None, channels: list[int] | None = None, max_side: int | None = None,
               with_scale: bool = True) -> tuple[Image.Image, dict]:
    """渲染一个截图框。返回 (PIL 图, 元信息)。channels 为 None 时用设置里的通道。"""
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
    rgb = render_rgb(project, sl, bbox0, lv, chans)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    W, H = img.size
    dr = ImageDraw.Draw(img)
    meta = {"image": im.key, "roi": roi.get("name", ""), "x": x, "y": y, "w": w, "h": h,
            "pixel_um": pix, "width_um": W * pix, "height_um": H * pix, "level": lv, "out_w": W, "out_h": H,
            "channels": [names[c] if c < len(names) else str(c) for c in chans]}
    sec = section_for(im, x + w / 2, y + h / 2)
    # 脑区边界 / 名称
    if (st.overlay_regions or st.overlay_region_names) and sec is not None:
        try:
            lab, acr = region_overlay(project, im, sl, sec, bbox0, (H, W))
        except Exception:  # noqa: BLE001
            lab = None
        if lab is not None:
            if st.overlay_regions:
                from .atlas import region_boundaries
                b = region_boundaries(lab)
                lw = max(1, int(round(min(W, H) / 600)))
                if lw > 1:
                    from scipy import ndimage as ndi
                    b = ndi.binary_dilation(b, iterations=lw - 1)
                arr = np.asarray(img).copy()
                arr[b] = (arr[b] * 0.4 + 0.6 * 230).astype(np.uint8)
                img = Image.fromarray(arr); dr = ImageDraw.Draw(img)
            if st.overlay_region_names:
                f, _ = _font(max(12, int(round(H * 0.03))))
                from scipy import ndimage as ndi
                ids = [v for v in np.unique(lab) if v > 0]
                if ids:
                    cents = ndi.center_of_mass(np.ones_like(lab, bool), lab, ids)
                    sizes = ndi.sum(np.ones_like(lab, bool), lab, ids)
                    for v, (cy, cx), n in zip(ids, cents, sizes):
                        if n < (W * H) * 0.004:
                            continue
                        t = str(acr.get(int(v), ""))
                        if not t:
                            continue
                        tw, th_ = dr.textbbox((0, 0), t, font=f)[2:]
                        px, py = cx - tw / 2, cy - th_ / 2
                        dr.text((px + 1, py + 1), t, fill=(0, 0, 0), font=f)
                        dr.text((px, py), t, fill=(240, 240, 240), font=f)
    # 细胞
    if st.overlay_cells and df is not None and not df.empty:
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
    # 定位小图
    if st.inset and sec is not None:
        try:
            ins = make_inset(project, sl, sec, bbox0, chans, max(80, int(W * 0.25)))
            m = max(6, int(round(min(W, H) * 0.02)))
            img.paste(ins, (m, m))
            meta["inset"] = True
        except Exception:  # noqa: BLE001
            pass
    if with_scale:
        meta["scale_bar"] = draw_scale_bar(img, pix, st.scale)
    return img, meta


# ------------------------------------------------------------------ 保存
def _svg(img_no_bar: Image.Image, sb: dict, font_family="Arial, 'Microsoft YaHei', 'PingFang SC', sans-serif") -> str:
    buf = io.BytesIO(); img_no_bar.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    W, H = img_no_bar.size
    col = "#000000" if sb["color"] == "黑" else "#ffffff"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<image width="{W}" height="{H}" xlink:href="data:image/png;base64,{b64}"/>',
             f'<g id="scale-bar"><rect x="{sb["x"]}" y="{sb["y"]}" width="{sb["length_px"]}" '
             f'height="{sb["thickness_px"]}" fill="{col}"/>']
    if sb.get("text"):
        parts.append(f'<text x="{sb["x"] + sb["length_px"] / 2:.1f}" y="{sb["text_y"] + sb["font_px"] * 0.85:.1f}" '
                     f'text-anchor="middle" font-family="{font_family}" font-size="{sb["font_px"]}" '
                     f'fill="{col}">{sb["text"]}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def save_outputs(img: Image.Image, img_no_bar: Image.Image | None, meta: dict, stem: Path, formats: list[str]) -> list[Path]:
    out = []
    stem.parent.mkdir(parents=True, exist_ok=True)

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
        p = ext(".svg"); p.write_text(_svg(img_no_bar, meta["scale_bar"]), encoding="utf-8"); out.append(p)
    stem.with_name(stem.name + "_参数.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float),
                                                      encoding="utf-8")
    return out


def export_roi(project: Project, im: ImageEntry, sl: Slide, roi: dict, st: FigureSettings, df=None,
               out_dir: Path | None = None) -> list[Path]:
    """导出一个截图框的所有文件（合并图 + 可选各通道单独图）。"""
    out_dir = Path(out_dir or project.folder / OUT_DIR)
    names = project.channel_names
    n_ch = len(sl.channels)
    chans = [c for c in (st.channels or list(range(n_ch))) if c < n_ch] or list(range(n_ch))
    base = f"{im.key}_{safe_name(roi.get('name', 'ROI'))}"
    files = []
    jobs = [(chans, "合并" if len(chans) > 1 else safe_name(names[chans[0]] if chans[0] < len(names) else str(chans[0])))]
    if st.separate and len(chans) > 1:
        jobs += [([c], safe_name(names[c] if c < len(names) else str(c))) for c in chans]
    for cs, tag in jobs:
        need_svg = "svg" in st.formats
        img_nb, meta = render_roi(project, im, sl, roi, st, df=df, channels=cs, with_scale=False)
        img = img_nb.copy() if need_svg else img_nb
        meta["scale_bar"] = draw_scale_bar(img, meta["pixel_um"], st.scale)
        files += save_outputs(img, img_nb if need_svg else None, meta, out_dir / f"{base}_{tag}", st.formats)
    return files


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
