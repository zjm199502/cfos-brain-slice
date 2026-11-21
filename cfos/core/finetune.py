"""从当前模型预标注中创建可编辑的训练样本。"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from . import detection as det

PATCH_UM = 256.0  # 标注块边长（µm）


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def training_dir(project, ch: int) -> Path:
    d = Path(project.folder) / "training" / _safe(project.channel_names[ch])
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_patches(project, ch: int) -> list[Path]:
    return sorted(training_dir(project, ch).glob("patch_*.npz"))


def make_patch(project, slide, image, center_xy, ch: int, cancel=None):
    """在 center_xy（第 0 层坐标）附近取一块，返回 (x 预处理后的图, labels 自动识别的初始标注, meta)。"""
    settings = project.detection
    from .unmix import prepare as _prepare_unmix
    _prepare_unmix(project, slide, image)      # 与正式识别一致：开启了串色校正时，标注块也用校正后的图
    p = settings.channels[ch]
    lv = slide.level_for_pixel_size(settings.pixel_um)
    ds = slide.downsample(lv)
    pix = slide.pixel_size_um * ds
    n = int(PATCH_UM / pix) // 16 * 16
    size0 = n * ds
    x0 = int(center_xy[0] - size0 / 2); y0 = int(center_xy[1] - size0 / 2)
    H0, W0 = slide.level_shapes[0]
    x0 = int(np.clip(x0, 0, max(W0 - size0, 0))); y0 = int(np.clip(y0, 0, max(H0 - size0, 0)))
    raw = det.read_unmixed(slide, lv, (x0, y0, size0, size0), ch, getattr(project.detection, "unmix", None))[:n, :n]
    # 归一化范围：用这块所在脑片的范围（与正式识别一致）
    sec = next((s for s in image.sections if s.enabled and s.bbox0[0] <= center_xy[0] < s.bbox0[0] + s.bbox0[2]
                and s.bbox0[1] <= center_xy[1] < s.bbox0[1] + s.bbox0[3]), None)
    if sec is not None:
        lo, hi, _ = det.section_norms(slide, sec, {ch: p}, getattr(project.detection, "unmix", None), cancel)[ch]
    else:
        lo, hi = np.percentile(raw, [p.norm_low, p.norm_high])
    x = det.prepare_input(raw, p, lo, hi, pix)
    lab = det.cellpose4.predict_labels(x, cellprob_threshold=p.cellprob_threshold, cancel=cancel)
    meta = {"image": image.key, "x0": x0, "y0": y0, "ds": ds, "n": n, "channel": project.channel_names[ch],
            "model": det.cellpose4.MODEL_KEY, "cellprob_threshold": p.cellprob_threshold,
            "time": f"{_dt.datetime.now():%Y-%m-%d %H:%M}"}
    return x, lab.astype(np.int32), meta


def save_patch(project, ch: int, x, labels, meta) -> Path:
    d = training_dir(project, ch)
    k = 1
    while (d / f"patch_{k:03d}.npz").exists():
        k += 1
    p = d / f"patch_{k:03d}.npz"
    np.savez_compressed(p, x=x.astype(np.float32), y=labels.astype(np.int32), meta=json.dumps(meta, ensure_ascii=False))
    return p
