"""Install an isolated, native Cellpose runtime without replacing the desktop app's Python packages."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from .components import components_root, platform_tag
from .resources import resources_dir
from .updater import _ssl_context

WEIGHTS_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2"
WEIGHTS_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
WEIGHTS_BYTES = 1_233_586_851
CELLPOSE_VERSION = "4.2.1"


def _root():
    return components_root() / "cellpose4"


def _python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_python() -> Path:
    root = _root()
    try:
        name = (root / "active.txt").read_text(encoding="utf-8").strip()
    except OSError:
        name = ""
    if not name.startswith("runtime-") or Path(name).name != name:
        name = "runtime"
    return _python(root / name)


def weights_path() -> Path:
    bundle = offline_bundle()
    if bundle.is_dir():
        return bundle / "cpsam_v2"
    return _root() / "cpsam_v2"


def offline_bundle() -> Path:
    return resources_dir() / "cellpose4"


def _base_python():
    base = Path(sys.executable)
    return base.with_name("python.exe") if base.name.lower() == "pythonw.exe" else base


def _offline_receipt(bundle):
    content = (bundle / "manifest.json").read_bytes()
    return {"manifest_sha256": hashlib.sha256(content).hexdigest(), "python": str(_base_python())}


def offline_ready() -> bool:
    """Quick first-launch check. Installation and repair also perform real inference."""
    python = runtime_python()
    if not python.is_file() or not weights_path().is_file():
        return False
    try:
        stored = (python.parent.parent / "fosfolio-offline.json").read_text(encoding="utf-8")
        return json.loads(stored) == _offline_receipt(offline_bundle())
    except (ValueError, OSError):
        return False


def _verify_file(path, size, sha, cancel):
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"离线组件缺失或大小不符：{path.name}。请重新解压或安装完整离线包。")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            _check(cancel)
            digest.update(chunk)
    if digest.hexdigest() != sha:
        raise ValueError(f"离线文件校验失败：{path.name}。请重新解压或安装完整离线包。")


def _offline_wheels(bundle, report, cancel):
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError("无法读取 Cellpose 离线清单。请重新解压或安装完整离线包。") from e
    if (not isinstance(manifest, dict) or manifest.get("format") != 1
            or manifest.get("platform") != platform_tag()
            or manifest.get("python") != f"{sys.version_info.major}.{sys.version_info.minor}"
            or manifest.get("cellpose") != CELLPOSE_VERSION):
        raise ValueError("此 Cellpose 离线包与当前系统、Python 或软件版本不匹配。")
    entries = manifest.get("wheels")
    if not isinstance(entries, list) or not entries:
        raise ValueError("离线清单没有依赖文件。")
    wheels = []
    seen = set()
    for entry in entries:
        name = entry.get("file") if isinstance(entry, dict) else None
        if (not isinstance(name, str) or not name.isascii() or not name.endswith(".whl")
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-" for c in name)
                or name.casefold() in seen):
            raise ValueError("离线清单包含无效或重复的 wheel 文件名。")
        seen.add(name.casefold())
        path = bundle / "wheels" / name
        report(f"正在校验离线组件（{len(wheels) + 1}/{len(entries)}）：{name}")
        _verify_file(path, entry.get("size"), entry.get("sha256"), cancel)
        wheels.append(path)
    report("正在校验随包的 Cellpose-SAM v2 模型…")
    _verify_file(bundle / "cpsam_v2", WEIGHTS_BYTES, WEIGHTS_SHA256, cancel)
    return wheels


def _check(cancel):
    if cancel and cancel():
        raise InterruptedError("已取消安装 Cellpose 4")


def _run(args, cancel, *, offline=False):
    _check(cancel)
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX"):
        env.pop(key, None)
    if offline:
        for key in list(env):
            if key.startswith("PIP_"):
                del env[key]
        env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", HF_HUB_OFFLINE="1")
    env.update(PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    with tempfile.TemporaryFile() as output:
        proc = subprocess.Popen(args, env=env, stdout=output, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while proc.poll() is None:
                _check(cancel)
                try:
                    proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if proc.poll() is None:
                import psutil
                try:
                    children = psutil.Process(proc.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill()
                proc.wait()
        if proc.returncode:
            size = output.tell()
            output.seek(max(0, size - 3000))
            detail = output.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"安装 Cellpose 4 时命令退出（{proc.returncode}）\n{detail}")


@contextmanager
def _install_lock(root):
    with (root / "install.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("其他 FosFolio 窗口正在安装 Cellpose 4。") from e
        yield


def _weights(report, cancel):
    target = weights_path()
    if target.is_file() and target.stat().st_size == WEIGHTS_BYTES:
        report("正在校验已有的 Cellpose 权重…")
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == WEIGHTS_SHA256:
                return
    _check(cancel)
    report("正在下载 Cellpose-SAM v2 模型…")
    part = target.with_suffix(".part")
    for attempt in range(3):
        _check(cancel)
        offset = part.stat().st_size if part.is_file() else 0
        if offset >= WEIGHTS_BYTES:
            break
        headers = {"User-Agent": "FosFolio/Cellpose4", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(WEIGHTS_URL, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30, context=_ssl_context()) as source:
                append = source.status == 206
                if append and not source.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("模型服务器返回了错误的分段位置，请稍后重试。")
                received = last = offset if append else 0
                with part.open("ab" if append else "wb") as output:
                    while block := source.read(1024 * 1024):
                        _check(cancel)
                        output.write(block)
                        received += len(block)
                        if received - last >= 16 * 1024**2:
                            report(f"正在下载 Cellpose-SAM v2 模型（{100 * received / WEIGHTS_BYTES:.0f}%）")
                            last = received
            if received >= WEIGHTS_BYTES:
                break
            raise ConnectionError("模型文件下载中断")
        except (OSError, http.client.HTTPException) as e:
            _check(cancel)
            if attempt == 2:
                raise RuntimeError("模型下载中断。再次安装时会从已保存的位置继续。") from e
    with part.open("rb") as stream:
        invalid = (part.stat().st_size != WEIGHTS_BYTES
                   or hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA256)
    if invalid:
        part.unlink()
        raise ValueError("Cellpose 模型文件校验失败，请重新安装。")
    part.replace(target)


def install(report=None, cancel=None, *, require_offline=False):
    """Publish the new environment only after its native torch and Cellpose imports succeed."""
    report = report or (lambda _: None)
    if sys.platform == "darwin" and platform.machine().lower() != "arm64":
        raise RuntimeError("请在 Apple 芯片 Mac 上运行 arm64 版 FosFolio，以启用 Metal GPU 加速。")
    bundle = offline_bundle()
    if require_offline and not bundle.is_dir():
        raise FileNotFoundError("未找到随包的 Cellpose 运行环境和模型。请使用完整离线安装包。")
    explicit_python = os.environ.get("CFOS_CELLPOSE_PYTHON")
    explicit_weights = os.environ.get("CFOS_CELLPOSE_WEIGHTS")
    if not bundle.is_dir() and explicit_python and explicit_weights:
        python, weights = Path(explicit_python), Path(explicit_weights)
        if python.is_file() and weights.is_file() and weights.stat().st_size == WEIGHTS_BYTES:
            report("正在验证当前配置的 Cellpose 4 环境…")
            with weights.open("rb") as file:
                verified = hashlib.file_digest(file, "sha256").hexdigest() == WEIGHTS_SHA256
            if verified:
                try:
                    _smoke(python, cancel, weights)
                    return "已验证当前配置的 Cellpose 4 环境。"
                except (RuntimeError, FileNotFoundError):
                    pass
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    with _install_lock(root):
        wheels = None
        if bundle.is_dir():
            wheels = _offline_wheels(bundle, report, cancel)
        else:
            _weights(report, cancel)
        result = _install_runtime(root, report, cancel, wheels)
        os.environ.pop("CFOS_CELLPOSE_PYTHON", None)
        os.environ.pop("CFOS_CELLPOSE_WEIGHTS", None)
        return result


def _smoke(python, cancel, weights=None, *, offline=False):
    worker = Path(__file__).with_name("cellpose_worker.py")
    _run([python, "-I", "-u", worker, "--check-env", weights or weights_path()], cancel, offline=offline)


def _install_runtime(root, report, cancel, wheels=None):
    current = runtime_python()
    offline = wheels is not None
    if current.is_file() and (not offline or offline_ready()):
        report("正在验证已安装的 Cellpose 4 环境…")
        try:
            _smoke(current, cancel, offline=offline)
            return "已有 Cellpose 4 运行环境和模型已通过验证。"
        except (RuntimeError, FileNotFoundError):
            pass
    if offline:
        required = 0
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                required += sum(entry.file_size for entry in archive.infolist())
        free = shutil.disk_usage(root).free
        if free < required:
            raise OSError(f"运行环境所在磁盘空间不足，可用 {free / 1024**3:.1f} GiB，"
                          f"仅解包依赖就需要 {required / 1024**3:.1f} GiB。")
    env_dir = Path(tempfile.mkdtemp(prefix="runtime-", dir=root))
    try:
        report("正在创建独立的 Cellpose 4 运行环境…")
        _run([_base_python(), "-I", "-m", "venv", env_dir], cancel, offline=offline)
        python = _python(env_dir)
        pip = [python, "-I", "-m", "pip", "install", "--progress-bar", "off"]
        if offline:
            report("正在从离线包安装 PyTorch、Cellpose 和依赖…")
            _run([*pip, "--no-index", "--no-deps", "--no-cache-dir", *wheels], cancel, offline=True)
            _run([python, "-I", "-m", "pip", "check"], cancel, offline=True)
        else:
            index = []
            if sys.platform == "win32":
                from .gpu_devices import gpu_names
                use_cuda = any("nvidia" in name.lower() for name in gpu_names())
                build = "cu128" if use_cuda else "cpu"
                index = ["--index-url", f"https://download.pytorch.org/whl/{build}"]
            report("正在安装 PyTorch，首次下载可能需要几分钟…")
            _run([*pip, *index, "torch==2.10.0", "torchvision==0.25.0"], cancel)
            report("正在安装 Cellpose 4 及其依赖…")
            _run([*pip, f"cellpose=={CELLPOSE_VERSION}"], cancel)
        report("正在验证模型和运行环境…")
        _smoke(python, cancel, offline=offline)
        if offline:
            (env_dir / "fosfolio-offline.json").write_text(json.dumps(_offline_receipt(offline_bundle())), encoding="utf-8")
        _check(cancel)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as pointer:
            pointer.write(env_dir.name)
        Path(pointer.name).replace(root / "active.txt")
    except BaseException:
        shutil.rmtree(env_dir, ignore_errors=True)
        raise
    return "Cellpose 4 已安装，点击“测试 Cellpose 设备”即可查看实际设备。"
