"""ONNX helpers and compatibility accessors for the current detection device."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("cfos")

PREF = "auto"            # "auto" = 有 GPU 就用；"cpu" = 只用 CPU（界面上可切换）
CHECK_TOL_PROB = 0.02    # 与 CPU 参考结果对比时，概率图允许的最大差别
CHECK_TOL_DIST = 0.25    # 距离图允许的最大差别（像素）

_GPU_EPS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
_EP_NAMES = {"CUDAExecutionProvider": "英伟达显卡（CUDA）",
             "DmlExecutionProvider": "显卡（DirectML）",
             "CoreMLExecutionProvider": "Apple 芯片加速（CoreML）",
             "CPUExecutionProvider": "CPU"}
_lock = threading.Lock()
_status = {"device": "", "detail": "", "failed": ""}


def _ort():
    try:
        import onnxruntime as ort
        return ort
    except Exception:  # noqa: BLE001
        return None


def gpu_provider() -> str | None:
    """本机可用的 GPU 计算方式；没有则为 None。"""
    ort = _ort()
    if ort is None:
        return None
    avail = ort.get_available_providers()
    for ep in _GPU_EPS:
        if ep in avail:
            if ep == "CoreMLExecutionProvider" and not (sys.platform == "darwin" and _is_apple_silicon()):
                continue
            return ep
    return None


def _is_apple_silicon() -> bool:
    import platform
    return platform.machine() in ("arm64", "aarch64")


def gpu_name() -> str:
    """Installed display adapters, independent of the device used for inference."""
    from .gpu_devices import gpu_names
    return "、".join(gpu_names())


def status_text() -> str:
    from . import cellpose4
    return cellpose4.status_text()


# ------------------------------------------------------------------ 模型转换
def _weights_file(model_dir: Path) -> Path | None:
    for n in ("weights_best.h5", "weights_last.h5"):
        if (model_dir / n).exists():
            return model_dir / n
    return None


def _signature(model_dir: Path) -> str:
    w = _weights_file(model_dir)
    if w is None:
        return ""
    h = hashlib.md5(w.read_bytes())          # 只看内容：安装、复制后文件时间会变
    cfg = model_dir / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    return h.hexdigest()


def onnx_path(model_dir: Path) -> Path:
    return Path(model_dir) / "model.onnx"


def export_onnx(keras_model, out: Path, sig: str = "") -> Path:
    """把处理二维图像的 Keras 网络转成 ONNX，保留动态输入尺寸。"""
    import tensorflow as tf
    import tf2onnx
    spec = (tf.TensorSpec((None, None, None, keras_model.inputs[0].shape[-1]), tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def f(x):
        return keras_model(x, training=False)

    tmp = out.with_suffix(".tmp")
    tf2onnx.convert.from_function(f, input_signature=spec, opset=17, output_path=str(tmp))
    tmp.replace(out)
    if sig:
        out.with_suffix(".sig").write_text(sig, encoding="utf-8")
    return out


def onnx_cache_dir() -> Path:
    """模型文件夹不可写（如 Windows 的 Program Files）时，转换结果放在用户目录。"""
    if sys.platform == "darwin":
        base = Path(os.path.expanduser("~/Library/Caches")) / "cfos脑片分析"
    else:
        base = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")) / "cfos脑片分析"
    return base / "onnx模型"


def _sig_ok(p: Path, sig: str) -> bool:
    sigf = p.with_suffix(".sig")
    return p.exists() and (not sigf.exists() or sigf.read_text(encoding="utf-8").strip() == sig)


def ensure_onnx(model, model_dir: Path) -> Path | None:
    """返回模型的 ONNX 文件，没有或已过期时转换一次（微调后的模型权重变了会重新转换）。

    通用模型的 model.onnx 在打包时就转换好随软件附带（packaging/make_onnx.py）；
    需要转换时先写模型文件夹，写不进去就写用户目录的缓存。"""
    model_dir = Path(model_dir)
    sig = _signature(model_dir)
    p = onnx_path(model_dir)
    if _sig_ok(p, sig):
        return p
    cache = onnx_cache_dir() / f"{sig or model_dir.name}.onnx"
    if _sig_ok(cache, sig):
        return cache
    for out in (p, cache):
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            return export_onnx(model.keras_model, out, sig)
        except Exception as e:  # noqa: BLE001  软件安装目录可能不可写，等等
            log.warning("模型转换为 ONNX 失败（%s：%s）", out, e)
    log.warning("模型无法转换为 ONNX，细胞识别使用 TensorFlow。")
    return None


# ------------------------------------------------------------------ 推理
class OrtNet:
    """代替 StarDist 的 keras_model：提供同样的 predict(x) → [prob, dist]。

    static=True（CoreML）时按输入大小分别建立固定大小的模型。"""

    def __init__(self, onnx_file: Path, provider: str, static: bool = False, adapter=None):
        self.onnx_file = Path(onnx_file)
        self.provider = provider
        self.static = static
        self.adapter = adapter
        self.label = f"ONNX Runtime（{_EP_NAMES.get(provider, provider)}）"
        if adapter is not None:
            self.label = f"ONNX Runtime（DirectML · {adapter.name}）"
        self._run_lock = threading.Lock()
        self._sessions: dict = {}

    def _session(self, shape, profile_prefix=None):
        if shape in self._sessions:
            return self._sessions[shape]
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        from .cpu import main_threads
        so.intra_op_num_threads = main_threads()
        so.inter_op_num_threads = 1
        if profile_prefix is not None:
            so.enable_profiling = True
            so.profile_file_prefix = str(profile_prefix)
        if self.provider == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            index = self.adapter.dml_index if self.adapter is not None else 0
            provs = [("DmlExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
        elif self.provider == "CoreMLExecutionProvider":
            provs = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                     "CPUExecutionProvider"]
        elif self.provider == "CUDAExecutionProvider":
            provs = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            provs = ["CPUExecutionProvider"]
        model = self.onnx_file
        if shape is not None:
            import onnx
            from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed, fix_output_shapes
            m = onnx.load(str(self.onnx_file))
            make_input_shape_fixed(m.graph, m.graph.input[0].name, list(shape))
            fix_output_shapes(m)
            model = m.SerializeToString()
        else:
            model = str(model)
        s = ort.InferenceSession(model, so, providers=provs)
        if self.provider not in s.get_providers():
            if profile_prefix is not None:
                s.end_profiling()
            raise RuntimeError(f"{self.provider} 创建失败，运行时仅启用了 {s.get_providers()}")
        s.disable_fallback()
        self._sessions[shape] = s
        return s

    def predict(self, x, **_):
        x = np.ascontiguousarray(x, dtype=np.float32)
        shape = tuple(x.shape) if self.static else None
        with self._run_lock:
            if self.provider in _GPU_EPS and shape not in self._sessions:
                with tempfile.TemporaryDirectory(prefix="cfos-ort-") as tmp:
                    s = self._session(shape, Path(tmp) / "stardist")
                    try:
                        result = s.run(None, {s.get_inputs()[0].name: x})
                    finally:
                        profile = s.end_profiling()
                    events = json.loads(Path(profile).read_text(encoding="utf-8"))
                    if not any(e.get("cat") == "Node" and e.get("args", {}).get("provider") == self.provider
                               and not str(e["args"].get("op_name", "")).startswith("Memcpy") for e in events):
                        self._sessions.pop(shape)
                        raise RuntimeError(f"模型没有在 {self.provider} 上执行计算节点")
            else:
                s = self._session(shape)
                result = s.run(None, {s.get_inputs()[0].name: x})
            while len(self._sessions) > 6:
                self._sessions.pop(next(iter(self._sessions)))
            return result


class _CheckedNet:
    """Check outputs against the CPU model and retry a failed tile on that model."""

    def __init__(self, ort_net: OrtNet, tf_model, name: str):
        self.ort_net, self.tf_model, self.name = ort_net, tf_model, name
        self.checked = False
        self.use_ort = True
        self.failure = ""
        self._predict_lock = threading.Lock()

    def __getattr__(self, item):           # 其他属性（训练等）照旧交给 Keras 模型
        return getattr(self.tf_model, item)

    def predict(self, x, **kw):
        with self._predict_lock:
            return self._predict(x, kw)

    def _predict(self, x, kw):
        if not self.use_ort:
            _status.update(device="CPU", failed=self.failure)
            return self.tf_model.predict(x, **kw)
        ref = None
        try:
            ys = self.ort_net.predict(x)
            if not self.checked:
                ref = self.tf_model.predict(x, **kw)
                if len(ys) < 2 or len(ref) < 2 or any(a.shape != b.shape for a, b in zip(ys[:2], ref[:2])):
                    raise ValueError("GPU 输出形状与 CPU 不一致")
                dp = float(np.max(np.abs(ys[0] - ref[0])))
                dd = float(np.max(np.abs(ys[1] - ref[1])))
                if not (dp <= CHECK_TOL_PROB and dd <= CHECK_TOL_DIST):
                    raise RuntimeError(f"与 CPU 结果不一致（概率差 {dp:.3f}，距离差 {dd:.3f}）")
                self.checked = True
                log.info("模型 %s 使用 %s；与 CPU 结果对比：概率差 %.4f，距离差 %.4f",
                         self.name or "通用模型", _EP_NAMES.get(self.ort_net.provider), dp, dd)
                _status.update(device=self.ort_net.label, failed="")
                return ref
            _status.update(device=self.ort_net.label, failed="")
            return ys
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:120] if str(e) else type(e).__name__
            log.warning("GPU 识别出错，改用 CPU：%s", e)
            _status.update(device="CPU", failed=msg)
            self.use_ort, self.failure = False, msg
            self.ort_net._sessions.clear()
            return ref if ref is not None else self.tf_model.predict(x, **kw)


def stardist_net(path, name=""):
    """Select a predictor for a bundled or locally converted StarDist ONNX model."""
    cpu = OrtNet(path, "CPUExecutionProvider")
    ep = gpu_provider() if PREF == "auto" else None
    if ep is None:
        _status.update(device=cpu.label, failed="")
        return cpu
    adapter = None
    if ep == "DmlExecutionProvider":
        from .gpu_devices import gpu_adapters
        available = (a for a in gpu_adapters() if a.dml_index is not None)
        adapter = max(available, key=lambda a: a.dedicated_bytes or 0, default=None)
    gpu = OrtNet(path, ep, static=ep == "CoreMLExecutionProvider", adapter=adapter)
    return _CheckedNet(gpu, cpu, name)


def probe_device(cancel=None) -> str:
    from . import cellpose4
    return cellpose4.probe_device(cancel=cancel)


def accelerate(model, model_dir, name: str = ""):
    """给 StarDist 模型装上 GPU 推理；没有 GPU 或设置为只用 CPU 时什么也不做。返回所用设备说明。"""
    tf_model = getattr(model, "_cfos_tf_model", None) or model.keras_model
    model._cfos_tf_model = tf_model
    if PREF == "cpu" or model_dir is None:
        model.keras_model = tf_model
        return "CPU"
    ep = gpu_provider()
    if ep is None:
        model.keras_model = tf_model
        return "CPU"
    with _lock:
        p = ensure_onnx(model, Path(model_dir))
    if p is None:
        model.keras_model = tf_model
        _status["failed"] = "模型转换失败"
        return "CPU"
    static = ep == "CoreMLExecutionProvider" or os.environ.get("CFOS_TEST_STATIC") == "1"
    model.keras_model = _CheckedNet(OrtNet(p, ep, static), tf_model, name)
    return _EP_NAMES.get(ep, ep)


def use_onnx(model, model_dir, name: str = "") -> bool:
    """Convert a local model when needed and use its ONNX network for detection."""
    if _ort() is None or model_dir is None:
        return False
    tf_model = getattr(model, "_cfos_tf_model", None) or model.keras_model
    model._cfos_tf_model = tf_model
    with _lock:
        p = ensure_onnx(model, Path(model_dir))
    if p is None:
        return False
    net = stardist_net(p, name)
    if os.environ.get("CFOS_CHECK_ONNX") == "1" and isinstance(net, OrtNet):
        model.keras_model = _CheckedNet(net, tf_model, name)
    else:
        model.keras_model = net
    return True


def backend_text() -> str:
    from . import cellpose4
    return cellpose4.backend_text()


def net_backend() -> str:
    """Select ONNX Runtime by default. CFOS_NET=tf explicitly selects TensorFlow."""
    env = os.environ.get("CFOS_NET", "").lower()
    if env in ("tf", "onnx"):
        return env
    return "onnx"


def set_pref(pref: str, models: dict | None = None):
    """Set the current detection preference and clear a caller-owned legacy cache."""
    from . import cellpose4
    global PREF
    PREF = "cpu" if pref == "cpu" else "auto"
    cellpose4.set_pref(PREF)
    _status.update(device="", failed="")
    if models is not None:
        models.clear()
