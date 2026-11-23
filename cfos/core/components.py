"""Optional TensorFlow runtime for DeepSlice and Keras-to-ONNX conversion.

The component archive and installation directory retain their historical names.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

TF_VERSION = "2.21.0"
REQUIRED = f"tf{TF_VERSION}-1"          # 组件包版本：TensorFlow 版本 + 打包修订号
MANIFEST = "components.json"
MAX_RETRY = 3


class ComponentMissing(RuntimeError):
    pass


def platform_tag() -> str:
    import platform
    m = platform.machine().lower()
    if sys.platform == "win32":
        return "win-amd64"
    if sys.platform == "darwin":
        return "mac-arm64" if m in ("arm64", "aarch64") else "mac-x86_64"
    return "linux-x86_64" if m in ("x86_64", "amd64") else f"linux-{m}"


def components_root() -> Path:
    env = os.environ.get("CFOS_COMPONENTS_DIR")
    if env:
        return Path(env)
    from .updater import updates_dir
    return updates_dir().parent / "组件"


def install_dir(version: str = REQUIRED) -> Path:
    return components_root() / f"微调组件_{version}_{platform_tag()}"


def _site(d: Path) -> Path:
    return d / "site-packages"


def installed_info() -> dict | None:
    f = install_dir() / "component.json"
    try:
        info = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if info.get("version") == REQUIRED and info.get("selftest_ok") else None


def activate() -> bool:
    """已经装好、自检通过的组件加到 sys.path 末尾（安装目录里的组件优先）。返回是否加了。"""
    if installed_info() is None:
        return False
    sp = str(_site(install_dir()))
    if sp not in sys.path:
        sys.path.append(sp)
        importlib.invalidate_caches()
    return True


def tensorflow_available() -> bool:
    """TensorFlow 和 tf2onnx 能不能用：安装包自带（v0.8 及以前）或已装好微调组件。"""
    def ok():
        return all(importlib.util.find_spec(m) is not None for m in ("tensorflow", "tf2onnx"))
    if ok():
        return True
    return activate() and ok()


def require_tensorflow(purpose: str = "脑图谱自动定位（DeepSlice）"):
    if not tensorflow_available():
        raise ComponentMissing(
            f"{purpose}需要 TensorFlow 兼容组件。使用随软件提供的 DeepSlice ONNX 模型时，"
            "请确认模型文件已安装，并清除 CFOS_NET=tf 后重新启动。")


# ------------------------------------------------------------------ 下载
def manifest_urls() -> list[str]:
    from . import updater
    env = os.environ.get("CFOS_COMPONENTS_URL")
    if env:
        return [env.rstrip("/") + "/" + MANIFEST]
    urls = []
    for url in updater.manifest_urls():
        folder = url.rsplit("/", 1)[0]
        for path in (updater.CHANNEL, updater.SERIES_CHANNEL):
            if folder.endswith("/" + path):
                folder = folder[:-len(path)] + "updates"
                break
        urls.append(f"{folder}/{MANIFEST}")
    return urls


def fetch_manifest() -> tuple[dict, str]:
    """返回 (清单, 清单所在的网址目录)。"""
    from . import updater
    last = None
    for u in manifest_urls():
        try:
            return json.loads(updater._get(u, timeout=20).decode("utf-8")), u.rsplit("/", 1)[0]
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"连不上下载服务器（{last}）。请检查网络，或用离线组件包安装。")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst: Path, size: int, progress=None, cancel=None):
    """断点续传下载到 dst（先写 dst.part）。"""
    import urllib.request
    from . import updater
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink(); have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"User-Agent": "FosMap-updater",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60, context=updater._ssl_context()) as r:
            if have and r.status != 206:      # 服务器不支持续传：从头下
                have = 0
            with open(part, "ab" if have else "wb") as f:
                while True:
                    if cancel is not None and cancel():
                        raise InterruptedError("已取消")
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); have += len(b)
                    if progress:
                        progress(have)
    part.replace(dst)


def download_pack(progress=None, cancel=None) -> Path:
    """下载本平台的组件包（各分卷核对 MD5 后合并成一个 zip），返回 zip 路径。"""
    import urllib.parse                      # 分卷文件名有中文，网址里要转义
    man, base = fetch_manifest()
    if man.get("version") != REQUIRED:
        raise RuntimeError(f"服务器上的微调组件版本（{man.get('version')}）与软件需要的（{REQUIRED}）不一致。"
                           "请先更新软件（帮助 → 检查更新）。")
    pack = (man.get("packs") or {}).get(platform_tag())
    if not pack:
        raise RuntimeError(f"服务器上没有适合这台电脑（{platform_tag()}）的微调组件。")
    dl = components_root() / "下载中"
    dl.mkdir(parents=True, exist_ok=True)
    total = sum(p["size"] for p in pack["parts"])
    done = 0
    for p in pack["parts"]:
        dst = dl / p["file"].rsplit("/", 1)[-1]
        for attempt in range(MAX_RETRY):
            if not (dst.exists() and dst.stat().st_size == p["size"] and _md5(dst) == p["md5"]):
                if dst.exists():
                    dst.unlink()
                _download(f"{base}/{urllib.parse.quote(p['file'])}", dst, p["size"],
                          (lambda n, d0=done: progress(d0 + n, total) if progress else None), cancel)
            if _md5(dst) == p["md5"]:
                break
            dst.unlink()                           # 校验不对：删掉重下
        else:
            raise RuntimeError(f"{dst.name} 下载 {MAX_RETRY} 次都校验失败，请稍后再试或用离线组件包。")
        done += p["size"]
    z = dl / f"微调组件_{REQUIRED}_{platform_tag()}.zip"
    with open(z, "wb") as out:
        for p in pack["parts"]:
            with open(dl / p["file"].rsplit("/", 1)[-1], "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
    if pack.get("md5") and _md5(z) != pack["md5"]:
        z.unlink()
        raise RuntimeError("组件包合并后校验失败，请重试。")
    for p in pack["parts"]:
        (dl / p["file"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    return z


# ------------------------------------------------------------------ 安装和自检
def read_pack_info(zip_path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        try:
            return json.loads(z.read("component.json").decode("utf-8"))
        except KeyError:
            raise ValueError("这不是微调组件包（里面没有 component.json）。") from None


def install_pack(zip_path, progress=None) -> dict:
    """解压组件包到用户目录并自检。返回 component.json 的内容。"""
    info = read_pack_info(zip_path)
    if info.get("version") != REQUIRED:
        raise ValueError(f"组件包版本是 {info.get('version')}，这个版本的软件需要 {REQUIRED}。")
    if info.get("platform") != platform_tag():
        raise ValueError(f"组件包是给 {info.get('platform')} 用的，这台电脑是 {platform_tag()}。")
    d = install_dir()
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for k, n in enumerate(names):
            if n.startswith("/") or ".." in Path(n).parts:
                raise ValueError(f"组件包里有不安全的路径：{n}")
            z.extract(n, tmp)
            if progress and k % 200 == 0:
                progress(k, len(names), "正在解压 TensorFlow 兼容组件…")
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    if progress:
        progress(0, 0, "正在核对 TensorFlow 和 ONNX 的推理结果…")
    ok, msg = selftest_subprocess(d)
    info["selftest_ok"] = ok
    info["selftest"] = msg
    (d / "component.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if not ok:
        raise RuntimeError(f"TensorFlow 兼容组件自检失败：{msg}\n请把诊断包发给开发者。")
    activate()
    # 清理旧版本组件
    for old in components_root().glob("微调组件_*"):
        if old != d and old.is_dir() and not old.name.endswith(".tmp"):
            shutil.rmtree(old, ignore_errors=True)
    return info


def selftest_subprocess(d: Path) -> tuple[bool, str]:
    """在单独的进程里自检（TensorFlow 出错或崩溃不会影响软件本身）。"""
    import subprocess
    from .updater import base_app_dir
    env = dict(os.environ, CFOS_COMPONENT_SITE=str(_site(d)), PYTHONNOUSERSITE="1")
    code = ("import sys, os; sys.path.append(os.environ['CFOS_COMPONENT_SITE']); "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); "
            "from cfos.core.components import selftest; ok, m = selftest(); print('RESULT', ok, m); sys.exit(0 if ok else 1)")
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        exe = exe[:-5] + ".exe"
    try:
        r = subprocess.run([exe, "-c", code], env=env, capture_output=True, text=True, timeout=900,
                           cwd=str(base_app_dir()), **({"creationflags": 0x08000000} if sys.platform == "win32" else {}))
    except Exception as e:  # noqa: BLE001
        return False, f"自检进程没能运行：{e}"
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), "")
    if r.returncode == 0 and line.startswith("RESULT True"):
        return True, line[len("RESULT True "):]
    from .log import log
    log.error("微调组件自检失败\n%s\n%s", r.stdout[-4000:], r.stderr[-4000:])
    return False, (line[len("RESULT False "):] if line else (r.stderr.strip().splitlines() or ["未知错误"])[-1])[:300]


def selftest() -> tuple[bool, str]:
    """Verify a known Keras transformation and its exported ONNX predictions."""
    import tempfile
    import time
    import numpy as np
    t0 = time.time()
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from . import accel
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(None, None, 1)),
            tf.keras.layers.Rescaling(scale=2.0, offset=1.0),
        ])
        x = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
        expected = 2 * x + 1
        np.testing.assert_allclose(model(x, training=False).numpy(), expected, rtol=1e-5)
        with tempfile.TemporaryDirectory() as td:
            p = accel.export_onnx(model, Path(td) / "model.onnx")
            converted, = accel.OrtNet(p, "CPUExecutionProvider").predict(x)
            np.testing.assert_allclose(converted, expected, rtol=1e-5)
        return True, f"TensorFlow {tf.__version__}，Keras 与 ONNX 推理通过（{time.time() - t0:.0f} 秒）"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def ensure_tensorflow(progress=None, cancel=None) -> str:
    """Install and validate the optional TensorFlow component when needed."""
    if tensorflow_available():
        return ""
    if progress:
        progress(0, 0, "正在连接服务器，准备下载 TensorFlow 兼容组件…")

    def rep(n, total):
        if progress:
            progress(n, total, f"正在下载 TensorFlow 兼容组件 {n / 1e6:.0f} / {total / 1e6:.0f} MB")
    z = download_pack(rep, cancel)
    info = install_pack(z, progress)
    z.unlink(missing_ok=True)
    if not tensorflow_available():
        raise RuntimeError("兼容组件已安装，但无法加载 TensorFlow。请把诊断包发给开发者。")
    return f"已安装 TensorFlow 兼容组件（{info.get('version')}）。"
