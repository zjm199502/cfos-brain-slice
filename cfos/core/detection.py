"""逐通道细胞识别：分块运行分割模型，测量细胞并保存坐标与轮廓。"""
from __future__ import annotations

import os
from time import monotonic
from dataclasses import dataclass, asdict, field, replace

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from skimage import measure

from .slide_io import Slide
from .sections import Section
from . import cellpose4, segmentation

TILE_SIZES = (2048, 1024, 512)


@dataclass
class ChannelParams:
    """每个通道的识别参数（界面上可调）。"""
    detect: bool = True
    min_area_um2: float = 20.0        # 细胞最小面积
    min_intensity: float = 0.0        # 平均荧光强度下限（原始灰度值），0 表示不限制
    bg_fold: float = 0.0              # 自动阈值：平均强度至少是组织背景的多少倍才算阳性，0 表示不用
    min_contrast: float = 0.0         # 局部对比度下限：细胞至少比紧邻周围（3 µm 宽的环）亮多少倍，0 表示不限制。
                                      # 厚切片中焦平面外的模糊细胞和周围差别小，用它可以去掉
    deconv: bool = False              # 识别前去卷积
    deconv_blur_um: float = 0.4       # 去卷积：模糊程度（µm）
    deconv_iter: int = 10             # 去卷积：迭代次数
    bg_radius_um: float = 0.0         # 识别前减背景的半径（µm），0 表示不减
    norm_low: float = 1.0             # 归一化百分位（低）
    norm_high: float = 99.8           # 归一化百分位（高）
    gamma: float = 1.0                # 识别前的伽马增强，<1 会把暗的细胞提亮（DAPI 默认 0.5）
    model: str = cellpose4.MODEL_KEY

    skip_dark_tiles: bool = True
    cellprob_threshold: float = 0.0   # Cellpose cell probability threshold (logit)

    model_override: bool = False
    resolution_percent: int | None = None  # None inherits the saved project grid.

    def __post_init__(self):
        model = segmentation.normalize_model_id(self.model)
        if model != self.model or model not in segmentation.NAMES:
            # An unavailable engine has no active override or reusable Cellpose threshold.
            self.model = cellpose4.MODEL_KEY
            self.model_override = False
            self.cellprob_threshold = 0.0

    @classmethod
    def from_dict(cls, stored, channel_name=""):
        values = {k: v for k, v in stored.items() if k in cls.__dataclass_fields__}
        if stored.get("model") != cellpose4.MODEL_KEY:
            values.pop("cellprob_threshold", None)
        if not values.get("model_override"):
            values["model"] = segmentation.default_model(channel_name)
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionSettings:
    pixel_um: float = 0.5             # 识别时使用的分辨率
    tile: int = TILE_SIZES[0]
    overlap: int = 96
    channels: dict = field(default_factory=dict)  # 通道序号 -> ChannelParams
    # 识别前去除通道串色（v0.8.2）：unmix.prepare() 在识别前设置，不单独保存（设置在 Project.unmix 里）
    unmix: object = field(default=None, repr=False, compare=False)
    unmix_sig: object = field(default=None, repr=False, compare=False)
    resolution_percent: int | None = 100
    display_windows: dict = field(default_factory=dict)  # resolved per section, never the project defaults
    tile_in_source_pixels: bool = True   # False preserves the legacy pixel_um grid until a size is chosen.
    # Downsampled channels: True makes the tile edge the model input (core + halo) in inference-grid
    # pixels, so a 2048 tile stays 2048 px at 20 % instead of shrinking to ~450 px. Projects saved
    # before this keep source pixels.
    tile_in_model_pixels: bool = True
    # Split each section into n = ceil(extent / tile) equal tiles per axis instead of full tiles plus a
    # thin remainder sliver, and crop each tile's model input to the tissue inside it (plus halo).
    # Projects saved before this keep their grid (and raw results) until a tile size is picked again.
    equal_tiles: bool = True


def channel_percent(settings, channel):
    value = settings.channels[channel].resolution_percent
    return settings.resolution_percent if value is None else value


def detection_grid(slide, settings, channel=None):
    """Select one pyramid source and the spacing of the inference grid."""
    percent = settings.resolution_percent if channel is None else channel_percent(settings, channel)
    if percent is None:  # preserve the original grid of an existing project
        level = slide.level_for_pixel_size(settings.pixel_um)
        return level, slide.downsample(level)
    if not 1 <= percent <= 100:
        raise ValueError("识别分辨率必须在 1% 到 100% 之间。")
    step = 100.0 / percent
    level = slide.level_for_pixel_size(step * slide.pixel_size_um)
    return level, step


def tile_dimensions(settings, step, preview=False):
    """Return the core edge and one-sided halo in pixels of the group's inference grid."""
    if settings.resolution_percent is None and not settings.tile_in_source_pixels:
        return settings.tile, settings.overlap
    halo = int(np.ceil(settings.overlap/step))  # the halo stays physical: cells keep their size in µm
    if tiles_in_model_pixels(settings):
        # Core + halo on both sides = tile, e.g. 2048 px, which Cellpose covers with 9 × 9 network tiles.
        return max(256, settings.tile - 2*halo), halo
    return max(1, round(settings.tile/step)), halo


def tiles_in_model_pixels(settings):
    """Whether this (single-percent) group sizes its tile by the downsampled model input."""
    return settings.tile_in_model_pixels and settings.resolution_percent not in (None, 100)


def read_detection_tile(slide, level, bbox, width, height, step):
    """Area-average on a section-anchored grid. Bound source reads even on flat TIFFs."""
    x, y, w, h = bbox
    source_step = slide.downsample(level)
    if step == source_step == 1:
        return slide.read(level, bbox)[:, :height, :width]
    # Integrate source pixels over each destination pixel's actual footprint.
    # At a clipped right/bottom edge only the last pixel has a shorter footprint.
    def average_axis(values, edges, axis):
        a = np.moveaxis(values, axis, -1)
        edges = np.clip(edges, 0, a.shape[-1])
        idx = np.floor(edges).astype(int)
        integral = np.zeros((*a.shape[:-1], a.shape[-1]+1), np.float64)
        np.cumsum(a, axis=-1, dtype=np.float64, out=integral[...,1:])
        samples = integral[...,idx] + (edges-idx)*a[...,np.minimum(idx,a.shape[-1]-1)]
        avg = np.diff(samples,axis=-1)/np.maximum(np.diff(edges),1e-12)
        return np.moveaxis(avg, -1, axis)

    chunk = max(1,min(1024,int(1024*source_step/step)))
    result = None
    for j in range(0,height,chunk):
        for i in range(0,width,chunk):
            n,m = min(chunk,width-i),min(chunk,height-j)
            left, top = x+i*step,y+j*step
            right,bottom = min(x+w,left+n*step),min(y+h,top+m*step)
            block = slide.read(level,(left,top,right-left,bottom-top))
            if result is None:
                result = np.zeros((block.shape[0],height,width),block.dtype)
            if not block.size:
                continue
            ox,oy = int(left/source_step),int(top/source_step)
            ex = np.minimum(left+np.arange(n+1)*step,right)/source_step-ox
            ey = np.minimum(top+np.arange(m+1)*step,bottom)/source_step-oy
            if block.shape[1:]==(m,n) and np.array_equal(ex,np.arange(n+1)) and np.array_equal(ey,np.arange(m+1)):
                data = block
            else:
                data = average_axis(average_axis(block,ex,2),ey,1)
                if np.issubdtype(block.dtype,np.integer):
                    data = np.rint(data)
            result[:,j:j+m,i:i+n] = data
    return result


def section_norms(slide, section, channels, model=None, cancel=None):
    """Read the normalization thumbnail and rasterize the tissue mask once per section."""
    if not channels:
        return {}
    lv = slide.level_for_pixel_size(2.0)
    indices = list(channels)
    def read_planes(box):
        if model is not None and np.asarray(model.matrix).shape == (len(slide.channels),) * 2:
            from .unmix import apply
            return apply(slide.read(lv, box), model, indices)
        return slide.read(lv, box, indices)

    x, y, w, h = section.bbox0
    tif = getattr(slide, "_tif", None)
    if (tif is not None and slide.downsample(lv) == 1 and w*h > 8_000_000
            and tif.pages[0].dtype.kind == "u" and tif.pages[0].dtype.itemsize <= 2):
        return _band_norms(section, channels, read_planes, cancel)
    planes = read_planes(section.bbox0)
    mask = section_mask(section, planes.shape[1:], slide.downsample(lv))
    use_mask = mask.any()
    norms = {}
    for c, plane in zip(indices, planes):
        params = channels[c]
        lo, hi, bg = np.percentile(plane[mask] if use_mask else plane.ravel(),
                                   [params.norm_low, params.norm_high, 50])
        norms[c] = lo, hi, max(float(bg), 1.0)
    return norms


def preview_norms(slide, section, channels, model, cache, cancel=None):
    """Reuse scalar normalization values between fields in the current single-file image."""
    if cache is None or slide.path.suffix.lower() not in (".tif", ".tiff", ".qptiff"):
        return section_norms(slide, section, channels, model, cancel)
    stat = slide.path.stat()
    mixing = None if model is None else (np.asarray(model.matrix).tobytes(), np.asarray(model.offsets).tobytes())
    key = (str(slide.path), stat.st_size, stat.st_mtime_ns, slide.z_mode, slide.pixel_size_um,
           tuple(section.bbox0), np.asarray(section.contour0, dtype=float).tobytes(), mixing,
           tuple((c, p.norm_low, p.norm_high) for c, p in channels.items()))
    if key not in cache:
        values = section_norms(slide, section, channels, model, cancel)
        if len(cache) >= 24:
            cache.pop(next(iter(cache)))
        cache[key] = values
    return cache[key]


def _band_norms(section, channels, read_planes, cancel=None):
    """Exact linear percentiles without allocating a full-resolution scan and its mask."""
    from dataclasses import replace
    x, y, w, h = section.bbox0
    hist = np.zeros((len(channels), 65536), np.int64)
    unmasked = np.zeros_like(hist)
    tissue_seen = False
    band = max(1, min(256, 500_000 // max(1, w)))
    for offset in range(0, h, band):
        if cancel is not None and cancel():
            raise InterruptedError("已取消")
        box = (x, y+offset, w, min(h-offset, band))
        planes = read_planes(box)
        if not planes.size:
            continue
        mask = section_mask(replace(section, bbox0=box), planes.shape[1:], 1)
        if mask.any():
            tissue_seen = True
            for i, plane in enumerate(planes):
                hist[i] += np.bincount(plane[mask], minlength=65536)
        elif not tissue_seen:
            for i, plane in enumerate(planes):
                unmasked[i] += np.bincount(plane.ravel(), minlength=65536)
    norms = {}
    for (c, p), counts in zip(channels.items(), hist if tissue_seen else unmasked):
        cumulative = counts.cumsum()
        ranks = (cumulative[-1]-1)*np.array([p.norm_low, p.norm_high, 50])/100
        left = np.floor(ranks)
        lo = np.searchsorted(cumulative, left, side="right")
        hi = np.searchsorted(cumulative, np.ceil(ranks), side="right")
        lower, upper, bg = lo + (ranks-left)*(hi-lo)
        norms[c] = lower, upper, max(float(bg), 1.0)
    return norms


def read_unmixed(slide: Slide, lv: int, bbox0, ch: int, model=None) -> np.ndarray:
    """读一个通道；给了解混模型时先读全部通道、去除串色，再取这个通道（v0.8.2）。"""
    if model is None or len(slide.channels) < 2 or np.asarray(model.matrix).shape[0] != len(slide.channels):
        return slide.read(lv, bbox0, [ch])[0]
    from .unmix import apply
    return apply(slide.read(lv, bbox0), model, [ch])[0]


def section_mask(section: Section, shape, ds: float) -> np.ndarray:
    """把脑片轮廓栅格化成掩膜（坐标相对于 bbox 左上角）。"""
    from skimage.draw import polygon as skpoly
    x0, y0 = section.bbox0[0], section.bbox0[1]
    m = np.zeros(shape, bool)
    if not section.contour0:
        m[:] = True
        return m
    xs = [(x - x0) / ds for x, _ in section.contour0]
    ys = [(y - y0) / ds for _, y in section.contour0]
    rr, cc = skpoly(ys, xs, shape)
    m[rr, cc] = True
    return m


def prepare_input(raw: np.ndarray, p: "ChannelParams", lo: float, hi: float, pix: float, display_window=None) -> np.ndarray:
    """识别前的预处理（减背景 / 去卷积 → 归一化 → 伽马），识别和训练用同一个函数，保证一致。"""
    raw = raw.astype(np.float32)
    # A manual display window replaces automatic background subtraction and normalization.
    # Optional deconvolution is applied in the original intensity units for both paths.
    radius = p.bg_radius_um if display_window is None else 0.0
    if p.deconv or radius > 0:
        from .deconv import enhance
        raw = enhance(raw, pix, p.deconv, p.deconv_blur_um, p.deconv_iter, radius)
        if radius > 0:
            lo = 0.0
    if display_window is not None:
        black, white, gamma = display_window
        return np.power(np.clip((raw - black) / (white - black), 0., 1.), gamma)
    x = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1.5)
    if p.gamma != 1.0:
        x = np.clip(x, 0, 1) ** p.gamma
    return x.astype(np.float32)


RING_UM = 3.0  # 计算局部对比度时，细胞外围环的宽度


def local_contrast(chan: np.ndarray, lab: np.ndarray, rp, ring_px: int) -> float:
    """细胞内平均亮度 / 细胞外围一圈（不含其他细胞）的平均亮度。"""
    from scipy import ndimage as ndi
    r0, c0, r1, c1 = rp.bbox
    R0, C0 = max(0, r0 - ring_px - 1), max(0, c0 - ring_px - 1)
    R1, C1 = min(lab.shape[0], r1 + ring_px + 1), min(lab.shape[1], c1 + ring_px + 1)
    win = lab[R0:R1, C0:C1]
    me = win == rp.label
    big = ndi.binary_dilation(me, iterations=ring_px)
    ring = big & ~ndi.binary_dilation(me, iterations=1) & (win == 0)
    vals = chan[R0:R1, C0:C1]
    inner = float(vals[me].mean())
    if ring.sum() < 5:
        return float("nan")
    return inner / max(float(vals[ring].mean()), 1.0)


# ------------------------------------------------------------------ 并行测量
# 识别后逐个细胞测量（轮廓、亮度、局部对比度）是纯 CPU 的 Python 计算，比神经网络本身还慢。
# 这部分交给几个后台进程并行做；神经网络（GPU 或 CPU）在主进程里继续算下一块。结果与单进程完全相同。
_POOL = None
_POOL_N = 0


def n_workers() -> int:
    from .cpu import n_workers as _n
    return _n()


def _worker_init():
    """后台进程里每个进程只用一个线程（否则几个进程各开满所有核，互相抢反而更慢）。"""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def _pool():
    global _POOL, _POOL_N
    n = n_workers()
    if n < 1:
        return None
    if _POOL is None or _POOL_N != n:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_worker_init)
        _POOL_N = n
        import atexit
        atexit.register(_POOL.shutdown, wait=False, cancel_futures=True)
    return _POOL


def _reset_pool():
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
    _POOL = None


def _native_position(indices, step, extent):
    """Map coordinates at pixel centres or fractional contour vertices to level zero.

    A downsampled pixel covers [i * step, (i + 1) * step) of the native pixel
    cells. Native array indices denote centres, so the first cell starts at -0.5.
    The final output bin may span fewer source pixels.
    """
    edge_pos = indices + 0.5
    cell = np.floor(edge_pos)
    left = np.minimum(cell * step, extent)
    right = np.minimum((cell + 1) * step, extent)
    return left + (edge_pos - cell) * (right - left) - 0.5


def _measure_tile(lab, img, c, tile, geo, crit):
    """测量一块图像上一个通道识别出的细胞。tile=(tx, ty, tw, th, rx0, ry0)；geo=(x0, y0, w, h, ds, pix, contour0)；
    crit=(prefilter, min_area, thr, min_contrast, bg)。返回每个细胞一行的列表。"""
    tx, ty, tw, th, rx0, ry0 = tile
    x0, y0, width, height, ds, pix, contour0 = geo
    prefilter, min_area, thr, min_contrast, bg = crit
    poly = MplPath(np.asarray(contour0)) if contour0 is not None and len(contour0) > 2 else None
    n_ch = img.shape[0]
    ring_px = max(2, int(round(RING_UM / pix)))
    rows = []
    for rp in measure.regionprops(lab):
        sl = rp.slice
        m = rp.image
        edge_weights = None
        if (rx0 + sl[1].stop) * ds > width + 1e-7 or (ry0 + sl[0].stop) * ds > height + 1e-7:
            # Weight the last partial row / column by the native area it represents.
            iy, ix = rp.coords.T
            iy, ix = ry0 + iy, rx0 + ix
            wx = np.minimum(ds, width - ix * ds)
            wy = np.minimum(ds, height - iy * ds)
            edge_weights = wx * wy
            gx = np.average(ix * ds + (wx - 1) / 2, weights=edge_weights)
            gy = np.average(iy * ds + (wy - 1) / 2, weights=edge_weights)
            area = edge_weights.sum() * (pix / ds) ** 2
        else:
            cyl, cxl = rp.centroid
            gx = (rx0 + cxl) * ds + (ds - 1) / 2
            gy = (ry0 + cyl) * ds + (ds - 1) / 2
            area = rp.area * pix * pix
        # Ownership is defined in native coordinates, independent of tile overlap.
        if not (tx * ds <= gx < (tx + tw) * ds and ty * ds <= gy < (ty + th) * ds):
            continue
        if area < 2.0 or (prefilter and area < min_area):
            continue
        X0, Y0 = x0 + gx, y0 + gy
        if poly is not None and not poly.contains_point((X0, Y0)):
            continue
        if edge_weights is None:
            means = [float(img[k][sl][m].mean()) for k in range(n_ch)]
        else:
            means = [float(np.average(img[k][sl][m], weights=edge_weights)) for k in range(n_ch)]
        if prefilter and means[c] < thr:
            continue
        contrast = local_contrast(img[c], lab, rp, ring_px)
        if prefilter and min_contrast > 0 and contrast < min_contrast:
            continue
        cont = measure.find_contours(np.pad(m, 1), 0.5)
        cont = measure.approximate_polygon(max(cont, key=len) - 1, 0.7) if cont else np.zeros((0, 2))
        ys_ = _native_position(cont[:, 0] + sl[0].start + ry0, ds, height) + y0
        xs_ = _native_position(cont[:, 1] + sl[1].start + rx0, ds, width) + x0
        row = {"channel": c, "x": X0, "y": Y0, "area_um2": area, "background": bg, "contrast": contrast,
               "contour": np.round(np.stack([xs_, ys_], 1), 1).astype(np.float32)}
        for k in range(n_ch):
            row[f"mean_ch{k}"] = means[k]
        rows.append(row)
    return rows


def _equal_edges(extent: int, tile: int):
    """Edges of n = ceil(extent / tile) cores whose sizes differ by at most one pixel."""
    n = max(1, int(np.ceil(extent/tile)))
    return [(i*extent)//n for i in range(n+1)]


def _section_path(section: Section):
    if len(section.contour0) < 3:
        return None
    points = np.asarray(section.contour0, dtype=float)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    return MplPath(points)


def section_tiles(section: Section, ds: float, tile: int, field_box=None, equal=False):
    """Core tiles (tx, ty, tw, th) intersecting tissue, including partially covered edge tiles.

    equal=True splits the section into equal cores per axis (no remainder sliver). With a field box
    only the field is split, so a small preview runs no tiles beyond it; cells near tile seams may then
    differ slightly from a full run. The legacy grid keeps field tiles on the section-anchored grid.
    """
    x, y, w, h = section.bbox0
    width, height = int(np.ceil(w/ds)), int(np.ceil(h/ds))
    poly = _section_path(section)
    if equal:
        x0, y0, x1, y1 = 0, 0, width, height
        if field_box is not None:
            # Previews favour speed: split only the field itself into equal tiles (with the usual halo).
            fx, fy, fw, fh = field_box
            x0, y0 = max(0, int(np.floor((fx-x)/ds))), max(0, int(np.floor((fy-y)/ds)))
            x1, y1 = min(width, int(np.ceil((fx+fw-x)/ds))), min(height, int(np.ceil((fy+fh-y)/ds)))
            if x1 <= x0 or y1 <= y0:
                return []
        xs = [x0 + e for e in _equal_edges(x1-x0, tile)]
        ys = [y0 + e for e in _equal_edges(y1-y0, tile)]
        result = []
        for ty, ty1 in zip(ys, ys[1:]):
            for tx, tx1 in zip(xs, xs[1:]):
                l, t, r, b = x+tx*ds, y+ty*ds, x+min(w, tx1*ds), y+min(h, ty1*ds)
                # One spare pixel keeps a core whose edge lies exactly on the contour.
                if poly is None or poly.intersects_bbox(Bbox.from_extents(l-ds, t-ds, r+ds, b+ds), filled=True):
                    result.append((tx, ty, tx1-tx, ty1-ty))
        return result
    left, top, right, bottom = 0, 0, width, height
    if field_box is not None:
        fx, fy, fw, fh = field_box
        left = max(0, int(np.floor((fx-x)/(ds*tile)))*tile)
        top = max(0, int(np.floor((fy-y)/(ds*tile)))*tile)
        right = min(width, int(np.ceil((fx+fw-x)/ds)))
        bottom = min(height, int(np.ceil((fy+fh-y)/ds)))
    result = []
    for ty in range(top,bottom,tile):
        for tx in range(left,right,tile):
            bounds = Bbox.from_extents(x+tx*ds, y+ty*ds, x+min(w,(tx+tile)*ds), y+min(h,(ty+tile)*ds))
            if poly is None or poly.intersects_bbox(bounds, filled=True):
                result.append((tx, ty, tile, tile))
    return result


def _clip_polygon(points, left, top, right, bottom):
    """Sutherland-Hodgman clip of a (possibly concave) polygon to an axis-aligned rectangle."""
    def clip(pts, inside, cross):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i-1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out
    def at_x(xv):
        return lambda p, q: (xv, p[1] + (q[1]-p[1])*(xv-p[0])/(q[0]-p[0]))
    def at_y(yv):
        return lambda p, q: (p[0] + (q[0]-p[0])*(yv-p[1])/(q[1]-p[1]), yv)
    pts = [tuple(p) for p in points]
    for inside, cross in ((lambda p: p[0] >= left, at_x(left)), (lambda p: p[0] <= right, at_x(right)),
                          (lambda p: p[1] >= top, at_y(top)), (lambda p: p[1] <= bottom, at_y(bottom))):
        if not pts:
            break
        pts = clip(pts, inside, cross)
    return pts


def tissue_crop(section: Section, ds: float, core):
    """Bounding box (x0, y0, x1, y1), in inference pixels, of the tissue inside a core tile.

    Cells are kept only if their centroid lies inside the section contour, so the model input
    can shrink to this box plus the halo without changing which tissue cells a tile owns.
    """
    tx, ty, tw, th = core
    if len(section.contour0) < 3:
        return tx, ty, tx+tw, ty+th
    x0, y0 = section.bbox0[:2]
    points = (np.asarray(section.contour0, dtype=float) - (x0, y0)) / ds
    clipped = _clip_polygon(points, tx-1, ty-1, tx+tw+1, ty+th+1)
    if len(clipped) < 3:
        return tx, ty, tx+tw, ty+th
    c = np.asarray(clipped)
    # The rasterised contour covers whole inference pixels: round outward and keep one spare pixel.
    left = max(tx, int(np.floor(c[:, 0].min())) - 1)
    top = max(ty, int(np.floor(c[:, 1].min())) - 1)
    right = min(tx+tw, int(np.ceil(c[:, 0].max())) + 1)
    bottom = min(ty+th, int(np.ceil(c[:, 1].max())) + 1)
    if right <= left or bottom <= top:
        return tx, ty, tx+tw, ty+th
    return left, top, right, bottom


def detect_section(slide: Slide, section: Section, settings: DetectionSettings,
                   progress=None, cancel=None, prefilter: bool = True, norms=None, field_box=None, detail=None, channel_names=(), image_name="", preview=False, selected_boxes=None) -> pd.DataFrame:
    """识别一张脑片上所有通道的细胞。返回表格：每行一个细胞，坐标为第 0 层像素。

    prefilter=False 时只做最基本的筛选（面积 ≥ 2 µm²），面积 / 强度 / 对比度阈值留给 apply_filters，
    这样以后改这些阈值不需要重新识别。"""
    groups = {}
    active = {c: p for c, p in settings.channels.items() if 0 <= c < len(slide.channels) and p.detect}
    if not active:
        raise ValueError("请在「细胞识别」中勾选至少一个需要识别的通道。")
    for c, params in active.items():
        groups.setdefault(channel_percent(settings, c), {})[c] = params

    plans, totals, grids = [], {}, []
    names = {c: (channel_names[c] if c < len(channel_names) else str(c+1)) + " · " +
             segmentation.NAMES[p.model].split("（")[0] for c, p in active.items()}
    _, _, w, h = section.bbox0
    for percent, channels in groups.items():
        local = replace(settings, resolution_percent=percent, channels=channels)
        lv, ds = detection_grid(slide, local)
        tile, overlap = tile_dimensions(local, ds, preview)
        tiles = section_tiles(section, ds, tile, field_box, equal=local.equal_tiles)
        if selected_boxes is not None:
            sx, sy = section.bbox0[:2]
            tiles = [(tx, ty, tw, th) for tx, ty, tw, th in tiles if any(
                sx + tx*ds < right and sx + (tx+tw)*ds > left and
                sy + ty*ds < bottom and sy + (ty+th)*ds > top
                for left, top, right, bottom in selected_boxes)]
        plans.append((local, lv, ds, tile, overlap, tiles))
        totals.update(dict.fromkeys(channels, len(tiles)))
        label = "、".join(channel_names[c] if c < len(channel_names) else str(c+1) for c in channels)
        unit = "原图" if percent is not None or local.tile_in_source_pixels else "识别层"
        grids.append(f"{label} 识别比例 {100/ds:g}% · 金字塔第 {lv} 层 · {ds*slide.pixel_size_um:g} µm/像素"
                     f" · 识别区域 {int(np.ceil(w/ds))} × {int(np.ceil(h/ds))} px"
                     + (f" · 模型输入边长 {local.tile} 识别层 px（核心 {tile} px）" if tiles_in_model_pixels(local) else
                        f" · 核心边长 {local.tile} {unit} px（识别层 {tile} px）")
                     + (" · 等分图块并裁到组织范围" if local.equal_tiles else "")
                     + f" · 单侧重叠 {overlap} px，含重叠的模型输入边长最多 {tile + 2*overlap} px · {len(tiles)} 块")
    if detail:
        detail(dict(kind="section", title=f"{image_name} · {section.name}", channels=names,
                    channel_totals=totals, total=max(totals.values()), grid="；".join(grids)))
    total = sum(totals.values())
    if norms is None and total:
        if detail:
            detail(dict(kind="note", text=f"{section.name} 读取归一化范围"))
        norms = section_norms(slide, section, active, settings.unmix, cancel)
    frames = []
    done = 0
    def report(count, unused):
        progress(done + count, total)
    for local, lv, ds, tile, overlap, tiles in plans:
        frame = _detect_at_scale(slide, section, local, lv, ds, tile, overlap, tiles, names,
                                 report if progress else None, cancel, prefilter, norms, field_box, detail)
        frames.append(frame)
        done += len(tiles) * len(local.channels)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    df["cell_id"] = np.arange(len(df))
    df.attrs["dark_tiles"] = [entry for f in frames for entry in f.attrs["dark_tiles"]]
    for key in ("channel_models", "channel_devices"):
        df.attrs[key] = {c: value for f in frames for c, value in f.attrs[key].items()}
    df.attrs["inference_settings"] = raw_signature(settings)
    df.attrs["z_mode"] = getattr(slide, "z_mode", "max")
    if detail:
        detail(dict(kind="end", text=f"{section.name} 完成，返回 {len(df)} 个对象"))
    return df


def _detect_at_scale(slide, section, settings, lv, ds, _T, ov, tiles, names,
                     progress, cancel, prefilter, norms, field_box, detail):
    pix = slide.pixel_size_um * ds
    x0, y0, w, h = section.bbox0
    W, H = int(np.ceil(w / ds)), int(np.ceil(h / ds))
    n_ch = len(slide.channels)
    contour0 = np.array(section.contour0) if len(section.contour0) > 2 else None
    geo = (x0, y0, w, h, ds, pix, contour0)

    det_chs = [c for c in range(n_ch) if c in settings.channels and settings.channels[c].detect]
    um = getattr(settings, "unmix", None)
    um = um if um is not None and np.asarray(um.matrix).shape == (n_ch, n_ch) else None
    dark_tiles = []
    devices = {c: set() for c in det_chs}

    if field_box is not None:
        fx, fy, fw, fh = field_box
    def emit(kind, **values):
        if detail:
            detail(dict(kind=kind, **values))
    if detail:
        for c in det_chs:
            window = settings.display_windows.get(c)
            if window is None:
                source = "识别归一化 · 识别 Gamma " + str(settings.channels[c].gamma)
            else:
                lo, hi, gamma = window
                source = f"第③步亮度 · 最暗 {lo:g} · 最亮 {hi:g} · Gamma {gamma:g}"
            emit("note", text=f"{section.name} · {names[c]} · 模型输入：{source}")
    total = len(tiles) * max(1, len(det_chs))
    state = {"done": 0}
    results = []          # 按分块、通道的顺序保存（列表或尚未完成的后台任务），保证结果顺序与单进程相同
    pool = _pool() if tiles else None
    max_pending = 2 * max(1, _POOL_N)

    def tick(k=1):
        state["done"] += k
        if progress:
            progress(min(state["done"], total), total)

    def finish(i):
        f, args = results[i]
        try:
            results[i] = f.result()
        except Exception as e:  # noqa: BLE001  后台进程意外退出（内存不足等）：这一块在本进程里重算
            from concurrent.futures.process import BrokenProcessPool
            if not isinstance(e, BrokenProcessPool):
                raise
            import logging
            logging.getLogger("cfos").warning("并行测量进程意外退出，改为单进程：%s", e)
            _reset_pool()
            results[i] = _measure_tile(*args)
        tick()

    def pending():
        return [i for i, r in enumerate(results) if not isinstance(r, list)]

    def collect(block: bool):
        """把已完成的后台任务换成结果；block=True 时等到未完成的任务不超过上限（限制内存）。"""
        for i in pending():
            if results[i][0].done():
                finish(i)
        while block and len(pending()) > max_pending:
            finish(pending()[0])

    def submit(args):
        nonlocal pool
        if pool is not None:
            try:
                results.append((pool.submit(_measure_tile, *args), args))
                return
            except Exception as e:  # noqa: BLE001  后台进程启动失败等：改为在本进程里算
                import logging
                logging.getLogger("cfos").warning("并行测量不可用，改为单进程：%s", e)
                _reset_pool()
                pool = None
        results.append(_measure_tile(*args))
        tick()

    try:
        for tile_index, (tx, ty, tw, th) in enumerate(tiles, 1):
            if cancel is not None and cancel():
                raise InterruptedError("已取消")
            # 带重叠的读取区域（识别层坐标）；等分图块时只读组织所在范围
            cx0, cy0, cx1, cy1 = (tissue_crop(section, ds, (tx, ty, tw, th)) if settings.equal_tiles
                                  else (tx, ty, tx + tw, ty + th))
            rx0, ry0 = max(0, cx0 - ov), max(0, cy0 - ov)
            rx1, ry1 = min(W, cx1 + ov), min(H, cy1 + ov)
            bb = (x0 + rx0 * ds, y0 + ry0 * ds, min((rx1-rx0)*ds, w-rx0*ds), min((ry1-ry0)*ds, h-ry0*ds))
            img = read_detection_tile(slide, lv, bb, rx1-rx0, ry1-ry0, ds)
            if um is not None and img.size:
                from .unmix import apply as _unmix
                img = _unmix(img, um)       # 识别前去除通道串色；之后的识别和亮度测量都用校正后的图
            if img.size == 0:
                for c in det_chs:
                    emit("skip", channel=c, text=f"图块 {tile_index} 没有可读像素")
                tick(len(det_chs))
                continue
            for c in det_chs:
                p = settings.channels[c]
                lo, hi, bg = norms[c]
                thr = max(p.min_intensity, p.bg_fold * bg)
                x = prepare_input(img[c], p, lo, hi, pix, settings.display_windows.get(c))
                if p.skip_dark_tiles and float(x.max()) < 0.2:
                    emit("skip", channel=c, text=f"图块 {tile_index} 跳过暗区")
                    dark_tiles.append((c, x0 + tx * ds, y0 + ty * ds,
                                       x0 + min(tx + tw, W) * ds, y0 + min(ty + th, H) * ds))
                    tick()
                    continue
                crit = (prefilter, p.min_area_um2, thr, p.min_contrast, bg)
                started = monotonic()
                emit("start", channel=c, pixels=x.size, started=started, text=f"图块 {tile_index}/{len(tiles)}，模型输入 {x.shape[1]} × {x.shape[0]} px")
                def report_batch(done, total):
                    emit("batch", channel=c, done=done, total=total)
                labels, device = segmentation.predict(x, p, cancel, progress=report_batch if detail else None)
                duration = monotonic()-started
                emit("done", channel=c, duration=duration, text=f"图块 {tile_index} 完成 · {duration:.1f} 秒 · {device}")
                devices[c].add(device)
                submit((labels, img, c, (tx, ty, tw, th, rx0, ry0), geo, crit))
                collect(block=True)
        emit("note", text=f"{section.name} 正在汇总细胞测量")
        for i in pending():
            if cancel is not None and cancel():
                raise InterruptedError("已取消")
            finish(i)
    except BaseException:
        for i in pending():
            results[i][0].cancel()
        raise
    rows = [row for r in results for row in r]
    df = pd.DataFrame(rows)
    if df.empty:
        dtypes = {"channel": "int64", "x": "float64", "y": "float64",
                  "area_um2": "float64", "contour": "object"}
        dtypes.update({f"mean_ch{k}": "float64" for k in range(n_ch)})
        df = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()})
    df.insert(0, "section", section.name)
    df["cell_id"] = np.arange(len(df))
    if field_box is not None:
        df = df[(df.x >= fx) & (df.x < fx + fw) & (df.y >= fy) & (df.y < fy + fh)].copy()
    df.attrs["dark_tiles"] = dark_tiles
    df.attrs["channel_models"] = {c: settings.channels[c].model for c in det_chs}
    df.attrs["channel_devices"] = {c: sorted(labels) for c, labels in devices.items()}
    return df


RAW_KEYS = ("model", "cellprob_threshold", "norm_low", "norm_high", "gamma", "deconv", "deconv_blur_um",
            "deconv_iter", "bg_radius_um", "skip_dark_tiles")


def raw_signature(settings: DetectionSettings) -> str:
    """Include the segmentation engine and enabled channels' inference settings."""
    import json
    d = {"model": cellpose4.MODEL_KEY, "px": settings.pixel_um, "tile": settings.tile, "ov": settings.overlap,
         "ch": {str(c): {k: getattr(p, k) for k in RAW_KEYS}
                for c, p in settings.channels.items() if p.detect}}
    if settings.display_windows:
        d["display_windows"] = {str(c): v for c, v in settings.display_windows.items()}
        if any(p.detect and p.deconv and c in settings.display_windows for c, p in settings.channels.items()):
            d["deconvolve_before_window"] = True
    if settings.resolution_percent is not None:
        d["resolution_percent"] = settings.resolution_percent
    if settings.tile_in_source_pixels and any(p.detect and channel_percent(settings, c) is None for c, p in settings.channels.items()):
        d["tile_in_source_pixels"] = True
    # Only downsampled channels get a different grid, so full-resolution signatures stay unchanged.
    if settings.tile_in_model_pixels and any(p.detect and channel_percent(settings, c) not in (None, 100)
                                             for c, p in settings.channels.items()):
        d["tile_in_model_pixels"] = True
    if any(p.detect and channel_percent(settings, c) != 100 for c, p in settings.channels.items()):
        d["native_pixel_centres"] = 1
    if settings.equal_tiles:   # older projects load with False, so their signatures stay unchanged
        d["equal_tiles"] = 1
    for c, p in settings.channels.items():
        if p.detect and channel_percent(settings, c) != settings.resolution_percent:
            d["ch"][str(c)]["resolution_percent"] = p.resolution_percent
    if getattr(settings, "unmix_sig", None):     # 只有开启了串色校正才加进签名，旧项目的签名不变
        d["unmix"] = settings.unmix_sig
    return json.dumps(d, sort_keys=True)


def apply_filters(df: pd.DataFrame, settings: DetectionSettings) -> pd.DataFrame:
    """在不重新识别的情况下，按当前参数（面积、强度）重新筛选。"""
    if df.empty:
        return df
    keep = np.ones(len(df), bool)
    for c, p in settings.channels.items():
        sel = df["channel"].values == c
        a = df["area_um2"].values
        it = df[f"mean_ch{c}"].values
        thr = np.maximum(p.min_intensity, p.bg_fold * df["background"].to_numpy()) if "background" in df.columns \
            else p.min_intensity
        bad = sel & ((a < p.min_area_um2) | (it < thr))
        if p.min_contrast > 0 and "contrast" in df.columns:
            ct = df["contrast"].to_numpy(dtype=float)
            bad |= sel & (ct < p.min_contrast)  # NaN（周围全是细胞）不会被去掉
        keep &= ~bad
    return df[keep].reset_index(drop=True)
