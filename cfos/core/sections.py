"""在低分辨率图上自动找出一张玻片上的各个脑片。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .slide_io import Slide


@dataclass
class Section:
    id: int
    bbox0: tuple[int, int, int, int]  # 第 0 层坐标 (x, y, w, h)
    contour0: list[tuple[float, float]]  # 组织轮廓 (x, y)，第 0 层坐标
    name: str = ""
    enabled: bool = True
    flip_h: bool = False
    registration: dict = field(default_factory=dict)  # 见 registration.Registration

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["bbox0"] = tuple(d["bbox0"])
        d["contour0"] = [tuple(p) for p in d.get("contour0", [])]
        return Section(**d)


def tissue_mask(img: np.ndarray, pixel_um: float, sensitivity: float = 1.0) -> np.ndarray:
    """img: (C, H, W) 低分辨率图。返回组织掩膜。"""
    k = int(round(8.0 / pixel_um)) if 0 < pixel_um < 6.0 else 1
    if k >= 2 and min(img.shape[1:]) >= 8 * k:
        # 金字塔里没有约 8 µm 的层（如江丰 .kfbf 是 5.2 µm）时，先缩小再算：形态学运算的耗时随半径平方增长，
        # 原来要几十秒，缩小后不到 1 秒。结果放大回原尺寸。
        C, H, W = img.shape
        h, w = H // k, W // k
        small = img[:, :h * k, :w * k].astype(np.float32).reshape(C, h, k, w, k).mean(axis=(2, 4))
        m = tissue_mask(small, pixel_um * k, sensitivity)
        out = np.zeros((H, W), bool)
        big = np.repeat(np.repeat(m, k, 0), k, 1)
        out[:h * k, :w * k] = big
        if H > h * k:
            out[h * k:, :w * k] = big[-1:, :]
        if W > w * k:
            out[:, w * k:] = out[:, w * k - 1:w * k]
        return out
    chans = []
    for c in img.astype(np.float32):
        lo, hi = np.percentile(c, [1, 99.5])
        chans.append(np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1))
    m = np.max(chans, axis=0)
    m = filters.gaussian(m, sigma=max(1.0, 20 / pixel_um))
    t = filters.threshold_otsu(m) * 0.6 / sensitivity
    mask = m > t
    r = max(1, int(round(150 / pixel_um)))
    mask = ndi.binary_closing(mask, structure=morphology.disk(r))
    mask = ndi.binary_fill_holes(mask)
    return mask


def find_sections(slide: Slide, min_area_mm2: float = 3.0, pad_um: float = 200.0,
                  sensitivity: float = 1.0, channels: list[int] | None = None) -> list[Section]:
    """channels：用哪些通道找组织轮廓（None = 全部通道）。某个通道背景太亮时可以不用它（v0.8.2）。"""
    level = slide.level_for_pixel_size(8.0)
    ds = slide.downsample(level)
    pix = slide.pixel_size_um * ds
    img = slide.read(level, None, _valid(slide, channels))
    mask = tissue_mask(img, pix, sensitivity)
    lab = measure.label(mask)
    min_px = min_area_mm2 * 1e6 / pix ** 2
    H0, W0 = slide.level_shapes[0]
    secs = []
    for rp in measure.regionprops(lab):
        if rp.area < min_px:
            continue
        y0, x0, y1, x1 = rp.bbox
        pad = pad_um / pix
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(mask.shape[1], x1 + pad); y1 = min(mask.shape[0], y1 + pad)
        bbox0 = (int(x0 * ds), int(y0 * ds), int(min(W0, (x1 - x0) * ds)), int(min(H0, (y1 - y0) * ds)))
        cs = measure.find_contours(np.pad(lab == rp.label, 1), 0.5)
        c = max(cs, key=len) - 1
        c = measure.approximate_polygon(c, tolerance=1.0)
        contour0 = [(float(x * ds), float(y * ds)) for y, x in c]
        secs.append((bbox0, contour0))
    # 按从上到下、从左到右排序（同一行的容差为脑片高度的一半）
    if secs:
        hmed = np.median([b[3] for b, _ in secs])
        secs.sort(key=lambda s: (round(s[0][1] / max(hmed, 1)), s[0][0]))
    return [Section(id=i + 1, bbox0=b, contour0=c, name=f"S{i + 1:02d}") for i, (b, c) in enumerate(secs)]


def _valid(slide: Slide, channels):
    chs = [c for c in (channels or []) if 0 <= c < len(slide.channels)]
    return chs or None


def contour_in_bbox(slide: Slide, bbox0: tuple[int, int, int, int],
                    channels: list[int] | None = None) -> list[tuple[float, float]]:
    """用户手动调整矩形后，在矩形内重新找组织轮廓（取最大的一块组织）。"""
    level = slide.level_for_pixel_size(8.0)
    ds = slide.downsample(level)
    pix = slide.pixel_size_um * ds
    img = slide.read(level, bbox0, _valid(slide, channels))
    x0, y0 = bbox0[0], bbox0[1]
    rect = [(x0, y0), (x0 + bbox0[2], y0), (x0 + bbox0[2], y0 + bbox0[3]), (x0, y0 + bbox0[3])]
    if img.shape[1] < 4 or img.shape[2] < 4:
        return rect
    try:
        mask = tissue_mask(img, pix)
    except Exception:
        return rect
    lab = measure.label(mask)
    if lab.max() == 0:
        return rect
    rp = max(measure.regionprops(lab), key=lambda r: r.area)
    cs = measure.find_contours(np.pad(lab == rp.label, 1), 0.5)
    c = measure.approximate_polygon(max(cs, key=len) - 1, tolerance=1.0)
    x0l, y0l = int(x0 / ds), int(y0 / ds)
    return [(float((x + x0l) * ds), float((y + y0l) * ds)) for y, x in c]
