"""逐通道细胞识别：分块运行分割模型，测量细胞并保存坐标与轮廓。"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from skimage import measure

from .slide_io import Slide
from .sections import Section
from . import cellpose4


@dataclass
class ChannelParams:
    """每个通道的识别参数（界面上可调）。"""
    detect: bool = True
    min_area_um2: float = 20.0        # 细胞最小面积
    max_area_um2: float = 600.0       # 细胞最大面积
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

    def __post_init__(self):
        # Old projects and templates may still contain a StarDist model directory.
        self.model = cellpose4.MODEL_KEY

    @classmethod
    def from_dict(cls, stored):
        values = {k: v for k, v in stored.items() if k in cls.__dataclass_fields__}
        if stored.get("model") != cellpose4.MODEL_KEY:
            values.pop("cellprob_threshold", None)
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionSettings:
    pixel_um: float = 0.5             # 识别时使用的分辨率
    tile: int = 2048
    overlap: int = 96
    channels: dict = field(default_factory=dict)  # 通道序号 -> ChannelParams
    # 识别前去除通道串色（v0.8.2）：unmix.prepare() 在识别前设置，不单独保存（设置在 Project.unmix 里）
    unmix: object = field(default=None, repr=False, compare=False)
    unmix_sig: object = field(default=None, repr=False, compare=False)


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


def prepare_input(raw: np.ndarray, p: "ChannelParams", lo: float, hi: float, pix: float) -> np.ndarray:
    """识别前的预处理（减背景 / 去卷积 → 归一化 → 伽马），识别和训练用同一个函数，保证一致。"""
    raw = raw.astype(np.float32)
    if p.deconv or p.bg_radius_um > 0:
        from .deconv import enhance
        raw = enhance(raw, pix, p.deconv, p.deconv_blur_um, p.deconv_iter, p.bg_radius_um)
        if p.bg_radius_um > 0:
            lo = 0.0
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


def _measure_tile(lab, img, c, tile, geo, crit):
    """测量一块图像上一个通道识别出的细胞。tile=(tx, ty, T, rx0, ry0)；geo=(x0, y0, ds, pix, contour0)；
    crit=(prefilter, min_area, max_area, thr, min_contrast, bg)。返回每个细胞一行的列表。"""
    tx, ty, T, rx0, ry0 = tile
    x0, y0, ds, pix, contour0 = geo
    prefilter, min_area, max_area, thr, min_contrast, bg = crit
    poly = MplPath(np.asarray(contour0)) if contour0 is not None and len(contour0) > 2 else None
    n_ch = img.shape[0]
    ring_px = max(2, int(round(RING_UM / pix)))
    rows = []
    for rp in measure.regionprops(lab):
        cyl, cxl = rp.centroid
        gx, gy = rx0 + cxl, ry0 + cyl  # 识别层、脑片内坐标
        # 只保留中心在本分块“核心区”内的细胞，避免重叠区重复计数
        if not (tx <= gx < tx + T and ty <= gy < ty + T):
            continue
        area = rp.area * pix * pix
        if area < 2.0 or (prefilter and (area < min_area or area > max_area)):
            continue
        X0, Y0 = x0 + gx * ds, y0 + gy * ds
        if poly is not None and not poly.contains_point((X0, Y0)):
            continue
        sl = rp.slice
        m = rp.image
        means = [float(img[k][sl][m].mean()) for k in range(n_ch)]
        if prefilter and means[c] < thr:
            continue
        contrast = local_contrast(img[c], lab, rp, ring_px)
        if prefilter and min_contrast > 0 and contrast < min_contrast:
            continue
        cont = measure.find_contours(np.pad(m, 1), 0.5)
        cont = measure.approximate_polygon(max(cont, key=len) - 1, 0.7) if cont else np.zeros((0, 2))
        ys_ = (cont[:, 0] + sl[0].start + ry0) * ds + y0
        xs_ = (cont[:, 1] + sl[1].start + rx0) * ds + x0
        row = {"channel": c, "x": X0, "y": Y0, "area_um2": area, "background": bg, "contrast": contrast,
               "contour": np.round(np.stack([xs_, ys_], 1), 1).astype(np.float32)}
        for k in range(n_ch):
            row[f"mean_ch{k}"] = means[k]
        rows.append(row)
    return rows


def detect_section(slide: Slide, section: Section, settings: DetectionSettings,
                   progress=None, cancel=None, prefilter: bool = True, norms=None, field_box=None) -> pd.DataFrame:
    """识别一张脑片上所有通道的细胞。返回表格：每行一个细胞，坐标为第 0 层像素。

    prefilter=False 时只做最基本的筛选（面积 ≥ 2 µm²），面积 / 强度 / 对比度阈值留给 apply_filters，
    这样以后改这些阈值不需要重新识别。"""
    lv = slide.level_for_pixel_size(settings.pixel_um)
    ds = slide.downsample(lv)
    pix = slide.pixel_size_um * ds
    x0, y0, w, h = section.bbox0
    W, H = int(np.ceil(w / ds)), int(np.ceil(h / ds))
    n_ch = len(slide.channels)
    contour0 = np.array(section.contour0) if len(section.contour0) > 2 else None
    poly = MplPath(contour0) if contour0 is not None else None
    geo = (x0, y0, ds, pix, contour0)

    det_chs = [c for c in range(n_ch) if c in settings.channels and settings.channels[c].detect]
    if not det_chs:
        raise ValueError("请在「细胞识别」中勾选至少一个需要识别的通道。")
    um = getattr(settings, "unmix", None)
    um = um if um is not None and np.asarray(um.matrix).shape == (n_ch, n_ch) else None
    if norms is None:
        norms = section_norms(slide, section, {c: settings.channels[c] for c in det_chs}, um, cancel)
    dark_tiles = []
    devices = {c: set() for c in det_chs}

    T, ov = settings.tile, settings.overlap
    x_start, y_start, x_stop, y_stop = 0, 0, W, H
    if field_box is not None:
        # Keep the section's tile origin and full overlap, including at the field boundary.
        fx, fy, fw, fh = field_box
        x_start = max(0, int(np.floor((fx - x0) / (ds*T))) * T)
        y_start = max(0, int(np.floor((fy - y0) / (ds*T))) * T)
        x_stop = min(W, int(np.ceil((fx + fw - x0) / ds)))
        y_stop = min(H, int(np.ceil((fy + fh - y0) / ds)))
    tiles = [(tx, ty) for ty in range(y_start, y_stop, T) for tx in range(x_start, x_stop, T)]
    total = len(tiles) * max(1, len(det_chs))
    state = {"done": 0}
    results = []          # 按分块、通道的顺序保存（列表或尚未完成的后台任务），保证结果顺序与单进程相同
    pool = _pool()
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
        for tx, ty in tiles:
            if cancel is not None and cancel():
                raise InterruptedError("已取消")
            # 带重叠的读取区域（识别层坐标）
            rx0, ry0 = max(0, tx - ov), max(0, ty - ov)
            rx1, ry1 = min(W, tx + T + ov), min(H, ty + T + ov)
            bb = (x0 + rx0 * ds, y0 + ry0 * ds, (rx1 - rx0) * ds, (ry1 - ry0) * ds)
            # 跳过不含组织的分块
            if poly is not None:
                cx = x0 + (tx + min(T, W - tx) / 2) * ds
                cy = y0 + (ty + min(T, H - ty) / 2) * ds
                corners = np.array([[x0 + tx * ds, y0 + ty * ds], [x0 + (tx + T) * ds, y0 + ty * ds],
                                    [x0 + tx * ds, y0 + (ty + T) * ds], [x0 + (tx + T) * ds, y0 + (ty + T) * ds],
                                    [cx, cy]])
                inside_any = poly.contains_points(corners).any() or \
                    poly.intersects_bbox(Bbox([[x0 + tx * ds, y0 + ty * ds], [x0 + (tx + T) * ds, y0 + (ty + T) * ds]]))
                if not inside_any:
                    tick(len(det_chs))
                    continue
            img = slide.read(lv, bb)
            if um is not None and img.size:
                from .unmix import apply as _unmix
                img = _unmix(img, um)       # 识别前去除通道串色；之后的识别和亮度测量都用校正后的图
            if img.size == 0:
                tick(len(det_chs))
                continue
            for c in det_chs:
                p = settings.channels[c]
                lo, hi, bg = norms[c]
                thr = max(p.min_intensity, p.bg_fold * bg)
                x = prepare_input(img[c], p, lo, hi, pix)
                if p.skip_dark_tiles and float(x.max()) < 0.2:
                    dark_tiles.append((c, x0 + tx * ds, y0 + ty * ds,
                                       x0 + min(tx + T, W) * ds, y0 + min(ty + T, H) * ds))
                    tick()
                    continue
                crit = (prefilter, p.min_area_um2, p.max_area_um2, thr, p.min_contrast, bg)
                labels = cellpose4.predict_labels(x, cellprob_threshold=p.cellprob_threshold, cancel=cancel)
                devices[c].add(cellpose4.backend_text())
                submit((labels, img, c, (tx, ty, T, rx0, ry0), geo, crit))
                collect(block=True)
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
        df = pd.DataFrame(columns=["channel", "x", "y", "area_um2", "contour"] +
                          [f"mean_ch{k}" for k in range(n_ch)])
    df.insert(0, "section", section.name)
    df["cell_id"] = np.arange(len(df))
    if field_box is not None:
        df = df[(df.x >= fx) & (df.x < fx + fw) & (df.y >= fy) & (df.y < fy + fh)].copy()
    df.attrs["dark_tiles"] = dark_tiles
    df.attrs["channel_models"] = {c: cellpose4.MODEL_KEY for c in det_chs}
    df.attrs["channel_devices"] = {c: sorted(labels) for c, labels in devices.items()}
    df.attrs["inference_settings"] = raw_signature(settings)
    df.attrs["z_mode"] = getattr(slide, "z_mode", "max")
    return df


RAW_KEYS = ("cellprob_threshold", "norm_low", "norm_high", "gamma", "deconv", "deconv_blur_um",
            "deconv_iter", "bg_radius_um", "skip_dark_tiles")


def raw_signature(settings: DetectionSettings) -> str:
    """Include the segmentation engine and enabled channels' inference settings."""
    import json
    d = {"model": cellpose4.MODEL_KEY, "px": settings.pixel_um, "tile": settings.tile, "ov": settings.overlap,
         "ch": {str(c): {k: getattr(p, k) for k in RAW_KEYS}
                for c, p in settings.channels.items() if p.detect}}
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
        bad = sel & ((a < p.min_area_um2) | (a > p.max_area_um2) | (it < thr))
        if p.min_contrast > 0 and "contrast" in df.columns:
            ct = df["contrast"].to_numpy(dtype=float)
            bad |= sel & (ct < p.min_contrast)  # NaN（周围全是细胞）不会被去掉
        keep &= ~bad
    return df[keep].reset_index(drop=True)
