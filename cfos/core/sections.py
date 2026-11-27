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

    other_registrations: dict = field(default_factory=dict)

    def vertices(self):
        """Editable scanner coordinates; support old projects with only a crop rectangle."""
        if len(self.contour0) >= 3:
            points = np.asarray(self.contour0, dtype=float)
            if len(points) > 3 and np.array_equal(points[0], points[-1]):
                points = points[:-1]
            return points
        x, y, w, h = self.bbox0
        return np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=float)

    def geometry_key(self):
        """Identify the same closed boundary across vertex rotations, windings and straight-edge insertions."""
        import hashlib
        pts = self.vertices().astype(np.float32).astype(np.float64)
        pts = pts[np.any(pts != np.roll(pts, 1, axis=0), axis=1)]
        if len(pts) >= 3:
            prev_edge = pts - np.roll(pts, 1, axis=0)
            next_edge = np.roll(pts, -1, axis=0) - pts
            cross = prev_edge[:,0]*next_edge[:,1] - prev_edge[:,1]*next_edge[:,0]
            keep = (cross != 0) | ((prev_edge * next_edge).sum(axis=1) < 0)
            pts = pts[keep]
        def ring_bytes(points):
            if not len(points):
                return b""
            start = np.lexsort((points[:,1], points[:,0]))[0]
            return np.roll(points, -int(start), axis=0).astype("<f8").tobytes()
        return hashlib.sha256(np.asarray(self.bbox0, dtype="<i8").tobytes() +
                              min(ring_bytes(pts), ring_bytes(pts[::-1]))).hexdigest()

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


def _tissue_image(slide, channels, bbox0=None):
    """Read the available pyramid level in bands; return thumbnail and level-zero geometry."""
    lv = slide.level_for_pixel_size(8.0)
    ds = slide.downsample(lv)
    h, w = slide.level_shapes[lv]
    if bbox0 is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x, y, width, height = bbox0
        x0, y0 = max(0, int(x / ds)), max(0, int(y / ds))
        x1, y1 = min(w, int(np.ceil((x+width)/ds))), min(h, int(np.ceil((y+height)/ds)))
    channels = _valid(slide, channels)
    channels = list(range(len(slide.channels))) if channels is None else channels
    step = max(1, round(8.0 / (slide.pixel_size_um * ds)))
    if step == 1 or min(y1-y0, x1-x0) < step:
        return slide.read(lv, bbox0, channels), ds, x0*ds, y0*ds
    rows, cols = (y1-y0+step-1)//step, (x1-x0+step-1)//step
    out = np.empty((len(channels), rows, cols), np.float32)
    band = max(1, (4 * 1024**2) // (len(channels) * (x1-x0) * step))
    for r in range(0, rows, band):
        n = min(band, rows-r)
        sy = y0+r*step
        height = min(n*step, y1-sy)
        # Move read boundaries slightly inside level pixels to avoid rounding down/up
        # when a scanner level has a fractional downsample ratio.
        box = ((x0+.25)*ds, (sy+.25)*ds, (x1-x0-.5)*ds, (height-.5)*ds)
        a = slide.read(lv, box, channels).astype(np.float32)
        a = np.pad(a, ((0,0), (0,n*step-height), (0,cols*step-(x1-x0))), mode="edge")
        out[:,r:r+n] = a.reshape(len(channels), n, step, cols, step).mean(axis=(2,4))
    return out, ds*step, x0*ds, y0*ds


def _fit_components(img, pixel_um, sensitivity, min_pixels):
    """Group fragments at section scale, then follow the local fluorescence edge."""
    if min(img.shape[1:]) < 2:
        return []
    signal = np.zeros(img.shape[1:], np.float32)
    for channel in img:
        lo, hi = np.percentile(channel, [1, 99.5])
        if hi <= lo:
            hi = channel.max()
        if hi > lo:
            np.maximum(signal, np.clip((channel.astype(np.float32)-lo)/(hi-lo), 0, 1), out=signal)
    if signal.max() == signal.min():
        return []
    coarse = filters.gaussian(signal, sigma=max(1., 20/pixel_um))
    threshold = filters.threshold_otsu(coarse) * 0.6 / sensitivity
    edge = filters.gaussian(signal, sigma=max(.7, 10/pixel_um)) > threshold
    # Pad before closing so sections touching the scan boundary keep their tissue.
    radius = max(1, round(150/pixel_um))
    support = np.pad(coarse > threshold, radius, mode="edge")
    support = morphology.isotropic_closing(support, radius)[radius:-radius, radius:-radius]
    grouped = measure.label(ndi.binary_fill_holes(support))
    margin = max(2, int(np.ceil(100/pixel_um)))
    contours = []
    for region in measure.regionprops(grouped):
        if region.area < min_pixels:
            continue
        y0, x0, y1, x1 = region.bbox
        y0, x0 = max(0,y0-margin), max(0,x0-margin)
        y1, x1 = min(edge.shape[0],y1+margin), min(edge.shape[1],x1+margin)
        labels = grouped[y0:y1,x0:x1]
        component = labels == region.label
        # Only refit a narrow outer band. Keep dim interiors and exclude neighbours.
        pad = margin + 2
        core = np.pad(component, pad)
        nearby = ndi.distance_transform_edt(~core) <= margin
        interior = ndi.distance_transform_edt(core) > margin
        foreground = np.pad(edge[y0:y1,x0:x1], pad)
        fit = morphology.isotropic_closing(foreground & nearby, max(1., 32/pixel_um))
        fit = ndi.binary_fill_holes(interior | fit)
        fit = morphology.isotropic_dilation(fit, max(1., 12/pixel_um))[pad:-pad,pad:-pad]
        fit &= (labels == 0) | component
        candidates = measure.label(fit)
        overlap = np.bincount(candidates[component], minlength=2)
        overlap[0] = 0
        mask = candidates == overlap.argmax() if overlap.max() else component
        loops = measure.find_contours(np.pad(mask, 1), .5)
        loop = max(loops, key=len) - 1
        # Thumbnail pixel centres are offset half a pixel from scanner box corners.
        loop = measure.approximate_polygon(loop, tolerance=.8) + [y0+.5,x0+.5]
        contours.append((region.area, loop[:, ::-1]))
    return contours


def find_sections(slide: Slide, min_area_mm2: float = 3.0, pad_um: float = 16.0,
                  sensitivity: float = 1.0, channels: list[int] | None = None) -> list[Section]:
    """Fit tissue contours and bound each section by its contour plus a small crop margin."""
    img, ds, ox, oy = _tissue_image(slide, channels)
    pix = slide.pixel_size_um * ds
    h, w = slide.level_shapes[0]
    sections = []
    for _, path in _fit_components(img, pix, sensitivity, min_area_mm2 * 1e6 / pix**2):
        path = np.clip(path * ds + [ox, oy], [0, 0], [w, h])
        pad = max(0, pad_um) / slide.pixel_size_um
        x, y = np.maximum(0, np.floor(path.min(axis=0)-pad)).astype(int)
        x1, y1 = np.minimum([w,h], np.ceil(path.max(axis=0)+pad)).astype(int)
        sections.append(((int(x),int(y),int(x1-x),int(y1-y)), path.tolist()))
    if sections:
        height = np.median([b[3] for b, _ in sections])
        sections.sort(key=lambda s: (round(s[0][1] / max(height, 1)), s[0][0]))
    return [Section(id=i+1, bbox0=b, contour0=c, name=f"S{i+1:02d}") for i,(b,c) in enumerate(sections)]


def _valid(slide: Slide, channels):
    chs = [c for c in (channels or []) if 0 <= c < len(slide.channels)]
    return chs or None


def contour_in_bbox(slide: Slide, bbox0: tuple[int,int,int,int],
                    channels: list[int] | None = None) -> list[tuple[float,float]]:
    """Fit the largest component inside the selected section; no evidence leaves it untouched."""
    img, ds, ox, oy = _tissue_image(slide, channels, bbox0)
    fits = _fit_components(img, slide.pixel_size_um * ds, 1., 1)
    if not fits:
        return []
    path = max(fits, key=lambda item:item[0])[1] * ds + [ox,oy]
    x, y, w, h = bbox0
    return np.clip(path, [x,y], [x+w,y+h]).tolist()
