"""DeepSlice（切片在图谱中的自动定位）的 ONNX 版本（v0.8.1）：不需要 TensorFlow 和 DeepSlice 软件包。

与 DeepSlice 1.2.8 的 DSModel.predict(image_list, ensemble=True) 逐步对应：
1. 读图：PNG → RGB，最近邻缩放到 299×299（Keras load_img 的默认方式）；
2. 灰度化（skimage.rgb2gray），减去整张图的均值、除以标准差（ImageDataGenerator 的 samplewise 标准化）；
3. 灰度复制成 3 个通道，送进两个模型（Allen_Mixed_Best、Synthetic_data_final），两个结果取平均。
打包时由 packaging/make_onnx.py 从 DeepSlice 的权重转换，并核对与 TensorFlow 的结果一致。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PRIMARY = "deepslice_Allen_Mixed_Best.onnx"
SECONDARY = "deepslice_Synthetic_data_final.onnx"
_SESS: dict = {}


def source_signature(d: Path, h5: str) -> str:
    """权重文件的大小和开头 4 MB 的 MD5（文件 85 MB，全读太慢；权重只会整体替换）。"""
    p = Path(d) / h5
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return f"{p.stat().st_size}-{hashlib.md5(f.read(4 << 20)).hexdigest()}"


def onnx_ok(p: Path, sig: str) -> bool:
    s = p.with_suffix(".sig")
    return p.exists() and s.exists() and (not sig or s.read_text(encoding="utf-8").strip() == sig)


def available(d: Path) -> bool:
    return (Path(d) / PRIMARY).exists() and (Path(d) / SECONDARY).exists()


def preprocess(img_u8: np.ndarray) -> np.ndarray:
    """缩略图（uint8 灰度）→ 网络输入 (299, 299, 3) float32，与 DeepSlice 读 PNG 的结果逐位相同。"""
    from PIL import Image
    from skimage.color import rgb2gray
    im = Image.fromarray(np.asarray(img_u8)).convert("RGB")
    if im.size != (299, 299):
        im = im.resize((299, 299), Image.NEAREST)
    x = np.asarray(im, dtype=np.float32)
    x = rgb2gray(x).reshape(299, 299, 1)
    # Keras 设了 samplewise_std_normalization 时会自动同时打开 samplewise_center：先减均值再除标准差（原地，float32）
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + 1e-6
    out = np.zeros((299, 299, 3), np.float32)
    out[...] = x
    return out


def _session(p: Path):
    if p not in _SESS:
        import onnxruntime as ort
        from .cpu import main_threads
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = main_threads()
        so.inter_op_num_threads = 1
        _SESS[p] = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    return _SESS[p]


def predict(images: list[np.ndarray], d: Path) -> np.ndarray:
    """返回 (张数, 9) 的切面参数（QuickNII 格式 ox oy oz ux uy uz vx vy vz），两个模型取平均。"""
    d = Path(d)
    X = np.stack([preprocess(im) for im in images])
    outs = []
    for name in (PRIMARY, SECONDARY):
        s = _session(d / name)
        inp = s.get_inputs()[0].name
        # 一张一张算（DeepSlice 也是 batch_size=1），结果与批量算可能有极小的浮点差
        outs.append(np.concatenate([s.run(None, {inp: X[i:i + 1]})[0] for i in range(len(X))]).astype(np.float64))
    return np.mean(outs, axis=0)
