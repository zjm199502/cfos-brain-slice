"""Run Cellpose 4 in its own Python environment, keeping one model resident between tiles."""
from __future__ import annotations

import atexit
from collections import deque
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np
from .cellpose_setup import WEIGHTS_SHA256, runtime_python, weights_path

MODEL_KEY = "cellpose4:cpsam_v2"
_INSTALL = "请在「④ 细胞识别 → 安装 / 修复 Cellpose 4 运行环境」中安装。"
_WORKER = Path(__file__).with_name("cellpose_worker.py")
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_lifecycle = threading.Lock()
_request = threading.Lock()
_worker = None
_preference = "auto"
_status = "Cellpose 4 尚未运行"


def _paths():
    python_env = os.environ.get("CFOS_CELLPOSE_PYTHON")
    weights_env = os.environ.get("CFOS_CELLPOSE_WEIGHTS")
    python = Path(python_env or runtime_python())
    if not python_env and not python.is_file() and importlib.util.find_spec("cellpose"):
        python = Path(sys.executable)
    weights = Path(weights_env or weights_path())
    if not python.is_file() or not weights.is_file():
        raise RuntimeError("缺少 Cellpose 4 解释器或 cpsam_v2 权重。" + _INSTALL)
    # A POSIX venv's bin/python symlink is part of its environment identity.
    return python.absolute(), weights.resolve()


@lru_cache(maxsize=4)
def _runtime_info(python, stamp, pref):
    try:
        p = subprocess.run([str(python), "-I", "-u", str(_WORKER), "--versions", pref], capture_output=True,
                           timeout=30, creationflags=_FLAGS, check=True)
        return json.loads(p.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise RuntimeError("无法读取 Cellpose 4 和 PyTorch 的安装版本。" + _INSTALL) from e


@lru_cache(maxsize=4)
def _verified_sha(path, size, stamp):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    if h.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("cpsam_v2 模型校验失败。" + _INSTALL)
    return h.hexdigest()


def model_identity():
    python, weights = _paths()
    st = weights.stat()
    return {"model": MODEL_KEY, "sha256": _verified_sha(weights, st.st_size, st.st_mtime_ns),
            **_runtime_info(python, python.stat().st_mtime_ns, _preference), "device_pref": _preference,
            "inference_revision": 1}


class _Worker:
    def __init__(self, python, weights, pref):
        self.key = (python, weights, pref)
        self.responses = queue.Queue()
        self.errors = deque(maxlen=8)
        self.proc = subprocess.Popen([str(python), "-I", "-u", str(_WORKER), str(weights), pref],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_FLAGS)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read(self):
        with self.proc.stdout as stream:
            for line in stream:
                self.responses.put(line)
        self.proc.wait()
        self.responses.put(None)

    def _read_errors(self):
        with self.proc.stderr as stream:
            for line in stream:
                self.errors.append(line.decode("utf-8", "replace").strip())

    def close(self):
        # EOF is monitored independently of the running inference by the child.
        if not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def infer(self, folder, cutoff, cancel, progress):
        message = {"folder": str(folder), "threshold": float(cutoff), "progress": progress is not None}
        self.proc.stdin.write((json.dumps(message)+"\n").encode("ascii"))
        self.proc.stdin.flush()
        while True:
            if cancel is not None and cancel():
                self.close()
                self.proc.wait(timeout=15)
                raise InterruptedError("已取消 Cellpose 识别")
            try:
                line = self.responses.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                detail = " ".join(self.errors)[-400:]
                raise RuntimeError(f"Cellpose 工作进程已退出。{detail}")
            response = json.loads(line)
            if "batch" in response:
                if progress is not None:
                    progress(*response["batch"])
                continue
            if "error" in response:
                raise RuntimeError(response["error"] + " " + _INSTALL)
            return response["device"]


def close():
    global _worker
    with _lifecycle:
        if _worker is not None:
            _worker.close()
            _worker = None
        _runtime_info.cache_clear()


def set_pref(pref, models=None):
    global _preference, _status
    mode = "cpu" if pref == "cpu" else "auto"
    if mode != _preference:
        close()
        _preference = mode
        _status = "Cellpose 4 尚未运行"


def backend_text():
    return _status


def status_text():
    return backend_text()


def predict_labels(image, cellprob_threshold=0.0, cancel=None, progress=None):
    """Return 2D instance IDs for one pre-normalized fluorescence channel."""
    global _worker, _status
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(cellprob_threshold):
        raise ValueError("Cellpose 4 需要二维通道图像和有效的细胞概率阈值。")
    with _request:
        if cancel is not None and cancel():
            raise InterruptedError("已取消 Cellpose 识别")
        model_identity()
        python, weights = _paths()
        with _lifecycle:
            key = (python, weights, _preference)
            if _worker is None or _worker.key != key or _worker.proc.poll() is not None:
                if _worker is not None:
                    _worker.close()
                _worker = _Worker(*key)
            worker = _worker
        try:
            with tempfile.TemporaryDirectory(prefix="fos-cellpose-") as tmp:
                folder = Path(tmp)
                np.save(folder/"image.npy", x, allow_pickle=False)
                device = worker.infer(folder, cellprob_threshold, cancel, progress)
                labels = np.load(folder/"labels.npy", allow_pickle=False)
                if labels.shape != x.shape or labels.dtype != np.int32:
                    raise ValueError("Cellpose 返回的标签尺寸或类型无效。")
                _status = "Cellpose 4 · " + device
                return labels
        except Exception:
            close()
            raise


def probe_device(cancel=None):
    yy, xx = np.mgrid[:128, :128]
    predict_labels(np.exp(-((xx-64)**2+(yy-64)**2)/40), cancel=cancel)
    return backend_text()


atexit.register(close)
