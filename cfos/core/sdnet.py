"""不经 TensorFlow 的 StarDist 识别模型（v0.8.1）：直接读模型文件夹里的 config.json、thresholds.json
和 model.onnx，用 ONNX Runtime 计算。predict_sparse / predict_instances 与 stardist 的
StarDist2D（n_tiles=None）逐行对应，结果完全相同（tests/v081_onnx_check.py 核对）。

这样细胞识别不需要加载 TensorFlow：启动更快、内存少几百 MB，安装包以后也可以不带 TensorFlow。
TensorFlow 只在“微调模型”训练时才需要（见 core/components.py）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class OnnxStarDist:
    def __init__(self, model_dir: Path, onnx_file: Path, name: str = ""):
        from .accel import OrtNet
        self.logdir = Path(model_dir)
        self.name = name or self.logdir.name
        cfg = json.loads((self.logdir / "config.json").read_text(encoding="utf-8"))
        self.config = SimpleNamespace(**cfg)
        th = {"prob": 0.5, "nms": 0.4}
        tf = self.logdir / "thresholds.json"
        if tf.exists():
            th.update(json.loads(tf.read_text(encoding="utf-8")))
        self.thresholds = SimpleNamespace(prob=float(th["prob"]), nms=float(th["nms"]))
        self.keras_model = OrtNet(onnx_file, "CPUExecutionProvider", False)
        self.onnx_file = Path(onnx_file)
        (self.config.backbone == "unet" and self.config.axes == "YXC" and self.config.n_channel_in == 1) or \
            _raise(ValueError(f"不支持的模型结构：{self.logdir}"))

    # ---------------------------------------------------------------- 与 StarDist2D 相同的计算
    def _div_by(self):
        return tuple(p ** self.config.unet_n_depth * g for p, g in zip(self.config.unet_pool, self.config.grid))

    def _net(self, img):
        """img (Y, X) → prob (Y/g, X/g), dist (Y/g, X/g, n_rays)，以及补边前的大小。"""
        x = np.asarray(img)
        x = x[..., np.newaxis]                       # YX → YXC
        div = self._div_by()
        pad = [(0, (d - s % d) % d) for d, s in zip(div, x.shape[:2])] + [(0, 0)]
        x = np.pad(x, pad, mode="reflect")
        prob, dist = self.keras_model.predict(x[np.newaxis])[:2]
        prob, dist = prob[0], dist[0]
        prob = np.take(prob, 0, axis=-1)
        dist = np.maximum(1e-3, dist)
        return prob, dist, x.shape[:2], pad

    def predict_sparse(self, img, prob_thresh=None, b=2, show_tile_progress=False, **_):
        """返回 (prob, dist, points)，与 StarDist2D.predict_sparse(img, n_tiles=None) 相同。"""
        from stardist.nms import _ind_prob_thresh
        if prob_thresh is None:
            prob_thresh = self.thresholds.prob
        prob, dist, padded, pad = self._net(img)
        inds = _ind_prob_thresh(prob, prob_thresh, b=b)
        proba = prob[inds].copy()
        dista = dist[inds].copy()
        points = np.stack(np.where(inds), axis=1) * np.array(self.config.grid).reshape((1, 2))
        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1, self.config.n_rays))
        points = np.asarray(points).reshape((-1, 2))
        bounds = np.array([padded[0] - pad[0][1], padded[1] - pad[1][1]])
        idx = np.where(np.all(points < bounds, 1))
        return proba[idx], dista[idx], points[idx]

    def predict_instances(self, img, prob_thresh=None, nms_thresh=None, verbose=False, **_):
        """返回 (标签图, {"points", "prob", "coord"…})，与 StarDist2D.predict_instances（sparse）相同。"""
        from stardist.nms import non_maximum_suppression_sparse
        from stardist.geometry.geom2d import polygons_to_label
        from stardist.geometry import dist_to_coord
        if nms_thresh is None:
            nms_thresh = self.thresholds.nms
        prob, dist, points = self.predict_sparse(img, prob_thresh=prob_thresh)
        pts, probi, disti, _ = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh, verbose=False)
        lab = polygons_to_label(disti, pts, prob=probi, shape=np.asarray(img).shape, scale_dist=(1, 1))
        coord = dist_to_coord(disti, pts, scale_dist=(1, 1))
        return lab, {"coord": coord, "points": pts, "prob": probi}


def _raise(e):
    raise e


def onnx_for(model_dir: Path) -> Path | None:
    """模型文件夹对应的、与当前权重一致的 ONNX 文件（没有返回 None）。"""
    from . import accel
    model_dir = Path(model_dir)
    sig = accel._signature(model_dir)
    for p in (accel.onnx_path(model_dir), accel.onnx_cache_dir() / f"{sig or model_dir.name}.onnx"):
        if accel._sig_ok(p, sig) or (not sig and p.exists()):
            return p
    return None
