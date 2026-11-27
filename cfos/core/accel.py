"""ONNX helpers and compatibility accessors for the current detection device."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

PREF = "auto"            # "auto" = 有 GPU 就用；"cpu" = 只用 CPU（界面上可切换）

_GPU_EPS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
_EP_NAMES = {"CUDAExecutionProvider": "英伟达显卡（CUDA）",
             "DmlExecutionProvider": "显卡（DirectML）",
             "CoreMLExecutionProvider": "Apple 芯片加速（CoreML）",
             "CPUExecutionProvider": "CPU"}


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


# ------------------------------------------------------------------ 推理
class OrtNet:
    """执行 ONNX 网络，返回各输出节点的张量。

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
                    s = self._session(shape, Path(tmp) / "profile")
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


def probe_device(cancel=None) -> str:
    from . import cellpose4
    return cellpose4.probe_device(cancel=cancel)


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
    if models is not None:
        models.clear()
