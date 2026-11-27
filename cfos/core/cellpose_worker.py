"""Standalone worker launched by cellpose4.py in the Cellpose Python environment."""
import contextlib
import json
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import time


def runtime_versions():
    from importlib.metadata import version
    cp = version("cellpose")
    major, minor = (int(n) for n in cp.split(".")[:2])
    if major != 4 or minor < 2:
        raise RuntimeError(f"cpsam_v2 需要 Cellpose 4.2 或更高的 4.x 版本。已安装 {cp}")
    return {"cellpose": cp, "torch": version("torch")}


def _windows_pipe_lines():
    import _winapi
    import msvcrt

    fd = sys.stdin.fileno()
    handle = msvcrt.get_osfhandle(fd)
    pending = bytearray()
    while True:
        # A pending synchronous stdin read can stall NumPy's native imports on Windows.
        available, _ = _winapi.PeekNamedPipe(handle, 0)
        if not available:
            time.sleep(0.02)
            continue
        block = os.read(fd, min(available, 65536))
        if not block:
            return
        pending.extend(block)
        while (end := pending.find(b"\n")) >= 0:
            yield bytes(pending[:end + 1])
            del pending[:end + 1]


def listen(commands):
    try:
        lines = _windows_pipe_lines() if os.name == "nt" else sys.stdin.buffer
        for line in lines:
            commands.put(line)
    finally:
        # Closing the GUI or cancelling a tile must stop even an in-flight PyTorch call.
        os._exit(0)


def select_device(torch, pref):
    device = torch.device("cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device("mps")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return device, bf16


def load_model(weights, pref):
    import torch
    from cellpose import models
    runtime_versions()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    device, bf16 = select_device(torch, pref)
    model = models.CellposeModel(device=device, use_bfloat16=bf16, pretrained_model=str(weights))
    actual = next(model.net.parameters()).device.type
    if actual != device.type:
        raise RuntimeError(f"模型权重位于 {actual}，所选设备为 {device.type}")
    if actual == "cuda":
        label = "CUDA · " + torch.cuda.get_device_name(device)
    elif actual == "mps":
        label = "Apple GPU（Metal / PyTorch MPS，部分算子可在 CPU 运行）"
    else:
        label = "CPU"
    return model, label


def device_error(error):
    text = str(error).lower()
    return (isinstance(error, (RuntimeError, NotImplementedError)) and any(
        fragment in text for fragment in ("out of memory", "not implemented", "not supported",
                                         "does not support", "not currently implemented",
                                         "cuda error", "mps backend", "failed to allocate")))


def release_device_cache():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def evaluate(model, source, batch_size, threshold, report=None):
    import numpy as np
    hook = None
    if report is not None:
        from math import ceil
        from cellpose.transforms import get_pad_yx
        # Same 2-D grid as the pinned Cellpose 4.2.1 run_net. Get the tile
        # size from the actual forward input, including after a device retry.
        seen = 0
        def after_forward(module, args, output):
            nonlocal seen
            subtiles, _, by, bx = args[0].shape
            h, w = source.shape[:2]
            top, bottom, left, right = get_pad_yx(h, w, min_size=(by, bx))
            ny = 1 if h + top + bottom <= by else ceil(1.2 * (h + top + bottom) / by)
            nx = 1 if w + left + right <= bx else ceil(1.2 * (w + left + right) / bx)
            seen += subtiles
            report(ceil(seen / batch_size), ceil(ny * nx / batch_size))
        hook = model.net.register_forward_hook(after_forward)
    try:
        labels, _, _ = model.eval(source, channel_axis=-1, normalize=False, batch_size=batch_size,
                                  cellprob_threshold=threshold)
    finally:
        if hook is not None:
            hook.remove()
    return np.asarray(labels, dtype=np.int32).reshape(source.shape[:2])


def check_environment(weights):
    import numpy as np
    import torch
    versions = runtime_versions()
    if versions["cellpose"] != "4.2.1":
        raise RuntimeError(f"需要 Cellpose 4.2.1，实际安装 {versions['cellpose']}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, label = load_model(weights, "cpu")
    yy, xx = np.mgrid[:64,:64]
    source = np.zeros((64,64,3), np.float32)
    source[..., 0] = np.exp(-((xx-32)**2+(yy-32)**2)/40)
    labels = evaluate(model, source, 1, 0.)
    if labels.shape != (64,64) or labels.dtype != np.int32 or labels.min() < 0:
        raise RuntimeError("Cellpose 推理自检的标签无效")
    return {**versions, "device": label, "shape": list(labels.shape)}


def serve(weights, pref):
    commands = queue.Queue()
    threading.Thread(target=listen, args=(commands,), daemon=True).start()
    protocol = sys.stdout.buffer
    def report(done, total):
        protocol.write((json.dumps({"batch": [done, total]}) + "\n").encode("ascii"))
        protocol.flush()

    model = None
    batch_size = 4
    while True:
        command = json.loads(commands.get())
        try:
            with contextlib.redirect_stdout(sys.stderr):
                import numpy as np
                if model is None:
                    try:
                        model, label = load_model(weights, pref)
                    except (RuntimeError, NotImplementedError) as e:
                        if pref != "auto" or not device_error(e):
                            raise
                        fallback = str(e).splitlines()[0][:100]
                    if model is None:
                        release_device_cache()
                        model, label = load_model(weights, "cpu")
                        label = f"CPU（GPU 加载失败：{fallback}）"
                folder = Path(command["folder"])
                x = np.load(folder/"image.npy", allow_pickle=False)
                # Explicit channel axis also handles narrow sections and one-pixel fields.
                source = np.zeros((*x.shape, 3), np.float32)
                source[..., 0] = x
                while True:
                    try:
                        labels = evaluate(model, source, batch_size, command["threshold"],
                                          report if command.get("progress") else None)
                        break
                    except (RuntimeError, NotImplementedError) as e:
                        if model.device.type == "cpu" or not device_error(e):
                            raise
                        fallback = str(e).splitlines()[0][:100]
                    if batch_size > 1 and "memory" in fallback.lower():
                        batch_size = 1
                        release_device_cache()
                        continue
                    model = None
                    release_device_cache()
                    model, _ = load_model(weights, "cpu")
                    label = f"CPU（GPU 推理回退：{fallback}）"
                np.save(folder/"labels.npy", labels, allow_pickle=False)
                response = {"device": label}
        except Exception as e:
            response = {"error": f"Cellpose 4 运行失败（{type(e).__name__}）：{e}"}
        protocol.write((json.dumps(response) + "\n").encode("ascii"))
        protocol.flush()


if __name__ == "__main__":
    # Conda-based venvs share the base environment's native dependencies.
    native = Path(sys.base_prefix) / "Library" / "bin"
    if os.name == "nt" and native.is_dir():
        os.environ["PATH"] = str(native) + os.pathsep + os.environ["PATH"]
        dll_handle = os.add_dll_directory(str(native))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.argv[1] == "--versions":
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            device, bf16 = select_device(torch, sys.argv[2])
            info = runtime_versions()
            info.update(platform=platform.system(), arch=platform.machine(), preferred_device=device.type,
                        precision="bfloat16" if bf16 else "float32")
            if device.type == "cuda":
                info["gpu"] = torch.cuda.get_device_name(device)
                info["cuda"] = torch.version.cuda
        print(json.dumps(info))
    elif sys.argv[1] == "--check-env":
        with contextlib.redirect_stdout(sys.stderr):
            info = check_environment(Path(sys.argv[2]))
        print(json.dumps(info))
    else:
        serve(Path(sys.argv[1]), sys.argv[2])
