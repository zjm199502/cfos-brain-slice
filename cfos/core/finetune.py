"""用学生自己标注的少量区域微调 StarDist 识别模型（只用 CPU）。

流程：
1. 在图上选一块区域（约 250 × 250 µm），软件先用当前模型自动圈出细胞作为起点；
2. 学生修改：补画漏掉的细胞、擦掉错的；保存为一个“标注块”；
3. 有 5 块以上后点“开始训练”，在通用模型的基础上继续训练（约 10–30 分钟）；
4. 训练前后在留出的标注块上比较准确率（F1），确实变好才建议使用。
训练好的模型保存在项目文件夹的 models 里，可以在“④ 细胞识别”中按通道选择。
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
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


def make_patch(project, slide, image, center_xy, ch: int):
    """在 center_xy（第 0 层坐标）附近取一块，返回 (x 预处理后的图, labels 自动识别的初始标注, meta)。"""
    settings = project.detection
    p = settings.channels[ch]
    lv = slide.level_for_pixel_size(settings.pixel_um)
    ds = slide.downsample(lv)
    pix = slide.pixel_size_um * ds
    n = int(PATCH_UM / pix) // 16 * 16
    size0 = n * ds
    x0 = int(center_xy[0] - size0 / 2); y0 = int(center_xy[1] - size0 / 2)
    H0, W0 = slide.level_shapes[0]
    x0 = int(np.clip(x0, 0, max(W0 - size0, 0))); y0 = int(np.clip(y0, 0, max(H0 - size0, 0)))
    raw = slide.read(lv, (x0, y0, size0, size0), [ch])[0][:n, :n]
    # 归一化范围：用这块所在脑片的范围（与正式识别一致）
    sec = next((s for s in image.sections if s.enabled and s.bbox0[0] <= center_xy[0] < s.bbox0[0] + s.bbox0[2]
                and s.bbox0[1] <= center_xy[1] < s.bbox0[1] + s.bbox0[3]), None)
    if sec is not None:
        lo, hi, _ = det._norm_percentiles(slide, sec, ch, p.norm_low, p.norm_high)
    else:
        lo, hi = np.percentile(raw, [p.norm_low, p.norm_high])
    x = det.prepare_input(raw, p, lo, hi, pix)
    lab, _ = det.get_model(p.model).predict_instances(x, prob_thresh=p.prob_thresh, verbose=False)
    meta = {"image": image.key, "x0": x0, "y0": y0, "ds": ds, "n": n, "channel": project.channel_names[ch],
            "model": p.model, "time": f"{_dt.datetime.now():%Y-%m-%d %H:%M}"}
    return x, lab.astype(np.int32), meta


def save_patch(project, ch: int, x, labels, meta) -> Path:
    d = training_dir(project, ch)
    k = 1
    while (d / f"patch_{k:03d}.npz").exists():
        k += 1
    p = d / f"patch_{k:03d}.npz"
    np.savez_compressed(p, x=x.astype(np.float32), y=labels.astype(np.int32), meta=json.dumps(meta, ensure_ascii=False))
    return p


def _load(paths):
    from stardist import fill_label_holes
    from skimage.segmentation import relabel_sequential
    X, Y = [], []
    for p in paths:
        z = np.load(p)
        X.append(z["x"].astype(np.float32))
        Y.append(relabel_sequential(fill_label_holes(z["y"].astype(np.int32)))[0])
    return X, Y


def _augment(x, y):
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.05, 0.05)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y)


def f1_score(model, X, Y, prob_thresh=None) -> float:
    from stardist.matching import matching_dataset
    pred = [model.predict_instances(x, prob_thresh=prob_thresh, verbose=False)[0] for x in X]
    return float(matching_dataset(Y, pred, thresh=0.5, show_progress=False).f1)


def finetune(project, ch: int, epochs: int = 40, steps_per_epoch: int = 30, progress=None, cancel=None) -> dict:
    """返回 {"name": 新模型名, "f1_before": , "f1_after": , "n_train": , "n_val": }。"""
    paths = list_patches(project, ch)
    if len(paths) < 3:
        raise ValueError(f"至少需要 3 个标注块（现在有 {len(paths)} 个），建议 5–10 个，覆盖亮的、暗的、密集的区域。")
    # v0.8.1：安装包以后可能不带 TensorFlow，第一次训练时自动下载“微调组件”（core/components.py）
    from .components import ensure_tensorflow
    ensure_tensorflow(progress, cancel)
    import tensorflow as tf
    from stardist.models import StarDist2D
    X, Y = _load(paths)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * 0.2)))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt, Yt = [X[i] for i in ti], [Y[i] for i in ti]
    Xv, Yv = [X[i] for i in vi], [Y[i] for i in vi]

    p = project.detection.channels[ch]
    base = det.get_model(p.model)
    f1_before = f1_score(base, Xv, Yv, p.prob_thresh)

    root = Path(project.folder) / "models"; root.mkdir(parents=True, exist_ok=True)
    name = f"{_safe(project.channel_names[ch])}_ft_{_dt.datetime.now():%Y%m%d_%H%M}"
    dst = root / name
    shutil.copytree(Path(base.logdir), dst, ignore=shutil.ignore_patterns("model.onnx", "model.sig", "model.tmp"))
    model = StarDist2D(None, name=name, basedir=str(root))
    ps = min(256, min(min(x.shape) for x in Xt) // 16 * 16)
    model.config.train_patch_size = (ps, ps)
    model.config.train_batch_size = 4
    model.config.train_learning_rate = 1e-4
    model.config.train_tensorboard = False
    model.config.train_reduce_lr = {"factor": 0.5, "patience": 10, "min_delta": 0}
    model.prepare_for_training()

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress:
                progress(epoch + 1, epochs, f"正在训练：第 {epoch + 1}/{epochs} 轮"
                         + (f"（验证误差 {logs.get('val_loss', 0):.3f}）" if logs else ""))
            if cancel is not None and cancel():
                self.model.stop_training = True

    model.callbacks.append(CB())
    model.train(Xt, Yt, validation_data=(Xv, Yv), augmenter=_augment, epochs=epochs,
                steps_per_epoch=steps_per_epoch)
    if cancel is not None and cancel():
        shutil.rmtree(dst, ignore_errors=True)
        raise InterruptedError
    if progress:
        progress(epochs, epochs, "正在优化阈值并评估…")
    # 阈值在训练块上优化，留出的块只用来评估，避免“用考题调参再用同一份考题打分”导致准确率虚高
    model.optimize_thresholds(Xt, Yt, nms_threshs=[0.3], iou_threshs=[0.5])
    f1_after = f1_score(model, Xv, Yv, None)
    info = {"name": name, "channel": project.channel_names[ch], "f1_before": round(f1_before, 3),
            "f1_after": round(f1_after, 3), "n_train": len(Xt), "n_val": len(Xv), "epochs": epochs,
            "base": p.model or "通用模型 2D_versatile_fluo", "time": f"{_dt.datetime.now():%Y-%m-%d %H:%M}",
            "prob_thresh": float(model.thresholds.prob)}
    # 训练完马上转换成 ONNX：以后识别直接用它，不再需要 TensorFlow
    from . import accel
    onnx = accel.ensure_onnx(model, dst)
    if onnx is not None:
        ref = model.keras_model.predict(Xv[0][np.newaxis, ..., np.newaxis], verbose=0)
        got = accel.OrtNet(onnx, "CPUExecutionProvider").predict(Xv[0][np.newaxis, ..., np.newaxis])
        info["onnx_max_diff"] = round(max(float(np.abs(ref[0] - got[0]).max()), float(np.abs(ref[1] - got[1]).max())), 6)
    (dst / "训练记录.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    det.forget_model(name)
    return info


def list_models(project) -> list[dict]:
    root = Path(project.folder) / "models"
    out = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        f = d / "训练记录.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except ValueError:
                pass
    return out
