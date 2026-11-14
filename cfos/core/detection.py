"""逐通道细胞识别（StarDist，CPU 可运行），分块处理整张脑片，只保留细胞表格和轮廓以节省内存。"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from skimage import measure

from .slide_io import Slide
from .sections import Section

_MODELS: dict = {}
MODEL_ROOT = None   # 项目中微调模型所在的文件夹（打开项目时设置）


def _set_tf_threads():
    """TensorFlow 的线程数按 cpu.main_threads()（只在 TensorFlow 第一次运行前有效）。"""
    try:
        import tensorflow as tf
        from .cpu import main_threads
        n = main_threads()
        if tf.config.threading.get_intra_op_parallelism_threads() != n:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(2 if n >= 4 else 1)
    except Exception:  # noqa: BLE001  已经初始化过就保持原样
        pass


def _model_dir(key: str):
    from pathlib import Path
    if key:
        d = Path(MODEL_ROOT or ".") / key
        if not (d / "weights_best.h5").exists() and not (d / "weights_last.h5").exists() \
                and not (d / "model.onnx").exists():
            raise FileNotFoundError(f"找不到微调模型“{key}”（{d}）。请在“④ 细胞识别”中改回通用模型，或重新训练。")
        return d
    from .resources import stardist_model_dir
    return stardist_model_dir()


def get_model(name: str = ""):
    """name 为空 = 软件自带的通用模型；否则为项目 models 文件夹里微调过的模型。

    v0.8.1 起默认直接用模型文件夹里的 ONNX 文件（core/sdnet.py），不加载 TensorFlow；
    只有找不到对应的 ONNX 文件（比如 v0.8 以前训练、还没转换过的微调模型）或设置了 CFOS_NET=tf 时，
    才用 TensorFlow（需要已安装“微调组件”）。"""
    import logging
    from . import accel
    key = name or ""
    if key not in _MODELS:
        d = _model_dir(key)
        if accel.net_backend() == "onnx" and d is not None:
            from .sdnet import OnnxStarDist, onnx_for
            f = onnx_for(d)
            if f is not None:
                _MODELS[key] = OnnxStarDist(d, f, key)
                accel._status.update(device="CPU（ONNX Runtime）", failed="")
                logging.getLogger("cfos").info("细胞识别神经网络：ONNX Runtime（%s，不用 TensorFlow）", f.name)
                return _MODELS[key]
        from .components import require_tensorflow
        require_tensorflow(f"使用微调模型“{key}”" if key else "细胞识别")
        _set_tf_threads()
        from stardist.models import StarDist2D
        _MODELS[key] = StarDist2D(None, name=d.name, basedir=str(d.parent)) if d is not None \
            else StarDist2D.from_pretrained("2D_versatile_fluo")
    m = _MODELS[key]
    if not getattr(m, "_cfos_net_done", False) and hasattr(m, "_predict_setup"):    # TensorFlow 模型
        m._cfos_net_done = True
        if accel.net_backend() == "onnx":
            try:
                ok = accel.use_onnx_cpu(m, getattr(m, "logdir", None), key)   # 顺便转换成 ONNX，以后不再需要 TensorFlow
            except Exception as e:  # noqa: BLE001
                ok = False
                logging.getLogger("cfos").warning("ONNX Runtime 不可用，改用 TensorFlow：%s", e)
            logging.getLogger("cfos").info("细胞识别神经网络：%s", "ONNX Runtime" if ok else "TensorFlow")
    return m


def forget_model(name: str):
    _MODELS.pop(name, None)


@dataclass
class ChannelParams:
    """每个通道的识别参数（界面上可调）。"""
    detect: bool = True
    prob_thresh: float = 0.5          # StarDist 置信度阈值，越高越严格
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
    model: str = ""                   # 识别模型：空 = 通用模型；否则为项目里微调过的模型名称

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionSettings:
    pixel_um: float = 0.5             # 识别时使用的分辨率
    tile: int = 2048
    overlap: int = 96
    channels: dict = field(default_factory=dict)  # 通道序号 -> ChannelParams


def _norm_percentiles(slide: Slide, section: Section, ch: int, lo: float, hi: float):
    """在较低分辨率下估计整张脑片的归一化范围，保证各分块亮度一致。"""
    lv = slide.level_for_pixel_size(2.0)
    a = slide.read(lv, section.bbox0, [ch])[0]
    mask = section_mask(section, a.shape, slide.downsample(lv))
    v = a[mask] if mask.any() else a.ravel()
    lo_v, hi_v, bg = np.percentile(v, [lo, hi, 50])
    return lo_v, hi_v, max(float(bg), 1.0)


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


def _segment_measure(shape, prob, dist, points, nms_thresh, img, c, tile, geo, crit):
    """从 StarDist 网络输出得到细胞（非极大值抑制 + 画多边形，与 predict_instances 相同），再测量。
    在后台进程中运行，不需要加载 TensorFlow。"""
    from stardist.nms import non_maximum_suppression_sparse
    from stardist.geometry.geom2d import polygons_to_label
    pts, probi, disti, _ = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh, verbose=False)
    lab = polygons_to_label(disti, pts, prob=probi, shape=shape, scale_dist=(1, 1))
    return _measure_tile(lab, img, c, tile, geo, crit)


def detect_section(slide: Slide, section: Section, settings: DetectionSettings,
                   progress=None, cancel=None, prefilter: bool = True) -> pd.DataFrame:
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

    det_chs = [c for c in range(n_ch) if settings.channels.get(c, ChannelParams()).detect]
    norms = {c: _norm_percentiles(slide, section, c, settings.channels[c].norm_low,
                                  settings.channels[c].norm_high) for c in det_chs}

    T, ov = settings.tile, settings.overlap
    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
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
            results[i] = _segment_measure(*args)
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

    def submit(net, img, c, tile, crit):
        nonlocal pool
        args = net + (img, c, tile, geo, crit)
        if pool is not None:
            try:
                results.append((pool.submit(_segment_measure, *args), args))
                return
            except Exception as e:  # noqa: BLE001  后台进程启动失败等：改为在本进程里算
                import logging
                logging.getLogger("cfos").warning("并行测量不可用，改为单进程：%s", e)
                _reset_pool()
                pool = None
        results.append(_segment_measure(*args))
        tick()

    try:
        for tx, ty in tiles:
            if cancel is not None and cancel():
                raise InterruptedError("已取消")
            # 带重叠的读取区域（识别层坐标）
            rx0, ry0 = max(0, tx - ov), max(0, ty - ov)
            rx1, ry1 = min(W, tx + T + ov), min(H, ty + T + ov)
            bb = (x0 + rx0 * ds, y0 + ry0 * ds, (rx1 - rx0) * ds, (ry1 - ry0) * ds)
            img = slide.read(lv, bb)
            if img.size == 0:
                tick(len(det_chs))
                continue
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
            for c in det_chs:
                p = settings.channels[c]
                lo, hi, bg = norms[c]
                thr = max(p.min_intensity, p.bg_fold * bg)
                x = prepare_input(img[c], p, lo, hi, pix)
                if float(x.max()) < 0.2:
                    tick()
                    continue
                # 主进程只算神经网络（GPU 或 CPU），得到候选点；去重、画轮廓、测量交给后台进程
                model = get_model(p.model)
                prob, dist, points = model.predict_sparse(x, prob_thresh=p.prob_thresh, show_tile_progress=False)
                crit = (prefilter, p.min_area_um2, p.max_area_um2, thr, p.min_contrast, bg)
                submit((x.shape, prob, dist, points, model.thresholds.nms), img, c, (tx, ty, T, rx0, ry0), crit)
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
    return df


RAW_KEYS = ("detect", "model", "prob_thresh", "norm_low", "norm_high", "gamma", "deconv", "deconv_blur_um",
            "deconv_iter", "bg_radius_um")


def raw_signature(settings: DetectionSettings) -> str:
    """影响“原始识别结果”的参数。参数不变时可以直接复用上次的识别结果（中断后继续）。"""
    import json
    d = {"px": settings.pixel_um, "tile": settings.tile, "ov": settings.overlap,
         "ch": {str(c): {k: getattr(p, k) for k in RAW_KEYS if not (k == "model" and not p.model)}
                for c, p in sorted(settings.channels.items())}}
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
