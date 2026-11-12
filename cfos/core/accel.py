"""细胞识别神经网络的运行方式。

v0.7 起默认用 ONNX Runtime（CPU）跑 StarDist 的网络（use_onnx_cpu，见 net_backend）。
下面的 GPU 加速部分（accelerate()）【暂未启用】：安装包只带 CPU 版 onnxruntime。

StarDist 的神经网络部分改用 ONNX Runtime 运行：
- Windows：DirectML（英伟达 / AMD / Intel 显卡都能用，不需要另装 CUDA）；
- Mac（M 系列芯片）：CoreML（GPU + 神经网络引擎）；
- 没有可用的 GPU，或 GPU 出错时，自动回到 CPU。
网络之后的步骤（从概率图得到细胞轮廓、测量）和以前完全一样。

每个模型第一次在 GPU 上运行时，会用同一块图像和原来的 CPU 版本（TensorFlow）对比一次，
结果不一致就不用 GPU，保证换设备不改变识别结果。
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("cfos")

PREF = "auto"            # "auto" = 有 GPU 就用；"cpu" = 只用 CPU（界面上可切换）
CHECK_TOL_PROB = 0.02    # 与 TensorFlow 对比时，概率图允许的最大差别
CHECK_TOL_DIST = 0.25    # 距离图允许的最大差别（像素）

_GPU_EPS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
_EP_NAMES = {"CUDAExecutionProvider": "英伟达显卡（CUDA）",
             "DmlExecutionProvider": "显卡（DirectML）",
             "CoreMLExecutionProvider": "Apple 芯片 GPU（CoreML）",
             "CPUExecutionProvider": "CPU"}
_lock = threading.Lock()
_status = {"device": "CPU", "detail": "", "failed": ""}


def _ort():
    try:
        import onnxruntime as ort
        return ort
    except Exception:  # noqa: BLE001
        return None


def gpu_provider() -> str | None:
    """本机可用的 GPU 计算方式；没有则为 None。"""
    import os
    ort = _ort()
    if ort is None:
        return None
    forced = os.environ.get("CFOS_TEST_EP")   # 仅测试用：在没有 GPU 的机器上走一遍 GPU 流程
    if forced:
        return forced
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
    """显卡名称（只用于显示，取不到时为空）。"""
    try:
        if sys.platform == "win32":
            import subprocess
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_VideoController).Name"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            names = [s.strip() for s in out.splitlines() if s.strip()]
            nv = [s for s in names if "nvidia" in s.lower()]
            return (nv or names or [""])[0]
        if sys.platform == "darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def status_text() -> str:
    """界面上显示的当前设备说明。"""
    ep = gpu_provider()
    if PREF == "cpu":
        return "计算设备：CPU（已设置为只用 CPU）"
    if ep is None:
        return "计算设备：CPU（本机没有可用的 GPU 加速）"
    if _status["failed"]:
        return f"计算设备：CPU（GPU 未能使用：{_status['failed']}）"
    return f"计算设备：GPU 加速可用（{_EP_NAMES.get(ep, ep)}）"


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
    """把 StarDist 的 Keras 网络转成 ONNX（输入大小不固定）。"""
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

    def __init__(self, onnx_file: Path, provider: str, static: bool = False):
        self.onnx_file = Path(onnx_file)
        self.provider = provider
        self.static = static
        self._sessions: dict = {}

    def _session(self, shape):
        if shape in self._sessions:
            return self._sessions[shape]
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if self.provider == "CPUExecutionProvider":
            from .cpu import main_threads
            so.intra_op_num_threads = main_threads()
            so.inter_op_num_threads = 1
        if self.provider == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            provs = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
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
        self._sessions[shape] = s
        return s

    def predict(self, x, **_):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not self.static:
            s = self._session(None)
        else:
            # CoreML 需要固定大小：每种分块大小各建一个（不补边，保证与 CPU 结果相同）；最多保留几个
            s = self._session(tuple(x.shape))
            while len(self._sessions) > 6:
                self._sessions.pop(next(iter(self._sessions)))
        return s.run(None, {s.get_inputs()[0].name: x})


class _CheckedNet:
    """第一次调用时与 TensorFlow 结果对比；不一致或出错时退回 TensorFlow（CPU）。"""

    def __init__(self, ort_net: OrtNet, tf_model, name: str):
        self.ort_net, self.tf_model, self.name = ort_net, tf_model, name
        self.checked = False
        self.use_ort = True

    def __getattr__(self, item):           # 其他属性（训练等）照旧交给 Keras 模型
        return getattr(self.tf_model, item)

    def predict(self, x, **kw):
        if not self.use_ort:
            return self.tf_model.predict(x, **kw)
        try:
            ys = self.ort_net.predict(x)
            if not self.checked:
                ref = self.tf_model.predict(x, **kw)
                dp = float(np.max(np.abs(ys[0] - ref[0])))
                dd = float(np.max(np.abs(ys[1] - ref[1])))
                self.checked = True
                if dp > CHECK_TOL_PROB or dd > CHECK_TOL_DIST:
                    raise RuntimeError(f"与 CPU 结果不一致（概率差 {dp:.3f}，距离差 {dd:.3f}）")
                log.info("模型 %s 使用 %s；与 CPU 结果对比：概率差 %.4f，距离差 %.4f",
                         self.name or "通用模型", _EP_NAMES.get(self.ort_net.provider), dp, dd)
                _status.update(device=_EP_NAMES.get(self.ort_net.provider, "CPU"), failed="")
                return ref           # 第一块用 TensorFlow 结果，和以前完全一致
            return ys
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:120] if str(e) else type(e).__name__
            log.warning("GPU 识别出错，改用 CPU：%s", e)
            _status.update(device="CPU", failed=msg)
            self.use_ort = False
            return self.tf_model.predict(x, **kw)


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


def use_onnx_cpu(model, model_dir, name: str = "") -> bool:
    """用 ONNX Runtime（CPU）代替 TensorFlow 跑神经网络。Windows 上 TensorFlow 的 CPU 计算在多核机器上
    可能慢几百倍（2026-11-12 在 32 核电脑上实测），ONNX Runtime 没有这个问题。
    需要模型文件夹里有 model.onnx（通用模型随软件附带；微调模型在有 tf2onnx 时自动转换）。
    返回是否成功；失败时保持 TensorFlow。"""
    if _ort() is None or model_dir is None:
        return False
    tf_model = getattr(model, "_cfos_tf_model", None) or model.keras_model
    model._cfos_tf_model = tf_model
    with _lock:
        p = ensure_onnx(model, Path(model_dir))
    if p is None:
        return False
    net = OrtNet(p, "CPUExecutionProvider", False)
    if os.environ.get("CFOS_CHECK_ONNX") == "1":     # 开发时核对；默认不做（TensorFlow 在问题机器上太慢）
        model.keras_model = _CheckedNet(net, tf_model, name)
    else:
        model.keras_model = net
    _status.update(device="CPU（ONNX Runtime）", failed="")
    return True


def backend_text() -> str:
    """当前细胞识别神经网络用什么在跑（诊断用）。"""
    d = _status.get("device") or "CPU"
    return f"{d}（设定：{net_backend()}）" + (f"；GPU 未能使用：{_status['failed']}" if _status.get("failed") else "")


def net_backend() -> str:
    """细胞识别神经网络用什么跑：'onnx' 或 'tf'。环境变量 CFOS_NET=tf 可强制用 TensorFlow。

    默认用 ONNX Runtime（CPU）：示例脑片实测比 TensorFlow 快约一倍（446 秒 → 230 秒，4 核），
    每个细胞的位置、面积、亮度都与 TensorFlow 相同（最大差 0.05 像素）；
    而且不会碰到 TensorFlow 在某些多核 Windows 电脑上慢几百倍的问题。
    没装 onnxruntime、或模型转换失败时，自动回到 TensorFlow。"""
    env = os.environ.get("CFOS_NET", "").lower()
    if env in ("tf", "onnx"):
        return env
    return "onnx"


def set_pref(pref: str, models: dict | None = None):
    """切换 自动 / 只用 CPU。已加载的模型下次识别时生效。"""
    global PREF
    PREF = "cpu" if pref == "cpu" else "auto"
    _status["failed"] = ""
    for m in (models or {}).values():
        tf_model = getattr(m, "_cfos_tf_model", None)
        if tf_model is not None:
            m.keras_model = tf_model
            m._cfos_accel_done = False
