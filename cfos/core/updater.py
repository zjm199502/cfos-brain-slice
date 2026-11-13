"""软件更新（补丁包）。

补丁包只含程序代码（cfos 文件夹 + main.py，约 1 MB），不含 Python 和深度学习组件，
所以不用每次下载几百 MB 的完整安装包。

- 发布：packaging/make_patch.py 生成补丁 zip 和 latest.json（含更新公告），上传到 GitHub installers 分支的 updates/。
- 检查：启动后在后台读取 latest.json（先 GitHub，连不上再用 jsDelivr 镜像），有新版本就弹窗显示更新公告。
- 安装：下载 → 核对 SHA-256 → 解压到用户目录的“更新”文件夹（不改动安装目录）→ 重启后生效。
  启动时 main.py 把最新补丁的代码放在最前面（见 boot_path）。补丁有问题时可以“恢复到安装时的版本”。
  连续两次没能正常启动，会自动停用这个补丁。
- 也可以在 GitHub 上手动下载补丁 zip，在软件里点“从文件安装补丁…”。

补丁需要的完整安装包版本写在 min_base 里：依赖组件（Python 包、模型、图谱）变了就必须重新装完整包。

注意：本文件会被 main.py 在导入 cfos 之前直接执行（boot_path），所以顶层只能导入标准库。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = "zjm199502/cfos-brain-slice"
BRANCH = "installers"
MANIFEST_URLS = [
    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/updates/latest.json",
    f"https://cdn.jsdelivr.net/gh/{REPO}@{BRANCH}/updates/latest.json",      # 国内访问 GitHub 不稳定时
    f"https://fastly.jsdelivr.net/gh/{REPO}@{BRANCH}/updates/latest.json",
]
DOWNLOAD_PAGE = f"https://github.com/{REPO}/tree/{BRANCH}/updates"
FULL_PAGE = f"https://github.com/{REPO}/tree/{BRANCH}"
SOURCE_URL = f"https://github.com/{REPO}"
CONTACT = "张金铭，zjm199502@163.com，" + SOURCE_URL
MAX_FAILED_STARTS = 2


# ------------------------------------------------------------------ 版本和位置
def vtuple(v: str) -> tuple:
    out = []
    for p in str(v).strip().lstrip("vV").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def updates_dir() -> Path:
    if sys.platform == "darwin":
        base = Path(os.path.expanduser("~/Library/Application Support")) / "cfos脑片分析"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")) / "cfos脑片分析"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")) / "cfos脑片分析"
    return Path(os.environ.get("CFOS_UPDATES_DIR") or (base / "更新"))


def _read_version_file(init_py: Path) -> str:
    try:
        for line in init_py.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "0"


def base_app_dir() -> Path:
    """安装目录里的 app 文件夹（main.py 所在处）。"""
    return Path(os.environ.get("CFOS_BASE_APP") or Path(__file__).resolve().parents[2])


def base_version() -> str:
    return _read_version_file(base_app_dir() / "cfos" / "__init__.py")


def _state_file() -> Path:
    return updates_dir() / "current.json"


def _load_state() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(d: dict):
    f = _state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)


def active_patch() -> dict | None:
    """当前生效的补丁 {version, dir, ...}；没有、已停用或不比安装版本新时为 None。"""
    st = _load_state()
    v, d = st.get("version"), st.get("dir")
    if not v or not d or st.get("disabled"):
        return None
    app = updates_dir() / d / "app"
    if not (app / "cfos" / "__init__.py").exists():
        return None
    bv = base_version()
    if vtuple(v) <= vtuple(bv) or vtuple(st.get("min_base", "0")) > vtuple(bv):
        return None
    return {**st, "app": str(app)}


def current_version() -> str:
    p = active_patch()
    return p["version"] if p else base_version()


# ------------------------------------------------------------------ 启动（main.py 调用）
def boot_path(main_process: bool) -> str | None:
    """在导入 cfos 之前调用：有可用补丁就把它的代码放到 sys.path 最前面，返回补丁版本。
    后台计算进程（spawn）也会执行 main.py，所以它们也走这里，保证和主进程用同一份代码。"""
    try:
        p = active_patch()
        if p is None:
            return None
        if main_process:
            st = _load_state()
            n = int(st.get("failed_starts", 0))
            if n >= MAX_FAILED_STARTS:
                st["disabled"] = True
                st["disabled_reason"] = f"补丁 {p['version']} 连续 {n} 次没能正常启动，已自动停用"
                _save_state(st)
                os.environ["CFOS_PATCH_DISABLED"] = st["disabled_reason"]
                return None
            st["failed_starts"] = n + 1            # 正常打开主窗口后清零（mark_started_ok）
            _save_state(st)
        sys.path.insert(0, p["app"])
        os.environ["CFOS_PATCH_APP"] = p["app"]
        return p["version"]
    except Exception:  # noqa: BLE001  更新机制出问题不能影响软件启动
        return None


def mark_started_ok():
    st = _load_state()
    if st.get("failed_starts"):
        st["failed_starts"] = 0
        _save_state(st)


def revert_to_base():
    """停用所有补丁，下次启动用安装时的版本。"""
    st = _load_state()
    if st:
        st["disabled"] = True
        st["disabled_reason"] = "已手动恢复到安装时的版本"
        _save_state(st)


# ------------------------------------------------------------------ 联网检查
def _ssl_context():
    import ssl
    try:
        import certifi                       # Mac 版自带的 Python 找不到系统证书
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _get(url: str, timeout: float = 10.0) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "cfos-updater", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.read()


def manifest_urls() -> list[str]:
    env = os.environ.get("CFOS_UPDATE_URL")        # 测试用
    return [env] if env else MANIFEST_URLS


def fetch_manifest() -> tuple[dict, str]:
    """返回 (latest.json 内容, 它的网址)。全部连不上时抛出最后一个错误。"""
    err = None
    for u in manifest_urls():
        try:
            m = json.loads(_get(u).decode("utf-8"))
            if "version" in m:
                return m, u
        except Exception as e:  # noqa: BLE001
            err = e
    raise ConnectionError(f"连不上更新服务器（GitHub）：{err}")


def check() -> dict:
    """返回 {"status": "latest" | "patch" | "full", "manifest": ..., "url": ...}。
    patch = 有可直接安装的补丁；full = 有新版本但需要下载完整安装包（依赖组件变了）。"""
    m, url = fetch_manifest()
    cur, base = current_version(), base_version()
    if vtuple(m["version"]) <= vtuple(cur):
        return {"status": "latest", "manifest": m, "url": url}
    if vtuple(m.get("min_base", "0")) > vtuple(base):
        return {"status": "full", "manifest": m, "url": url}
    return {"status": "patch", "manifest": m, "url": url}


def skipped_version() -> str:
    return _load_state().get("skip", "")


def skip_version(v: str):
    st = _load_state(); st["skip"] = v; _save_state(st)


def auto_check_enabled() -> bool:
    return _load_state().get("auto_check", True)


def set_auto_check(on: bool):
    st = _load_state(); st["auto_check"] = bool(on); _save_state(st)


def download_patch(check_result: dict, progress=None) -> Path:
    """下载补丁 zip 并核对 SHA-256，返回本地路径。"""
    m, url = check_result["manifest"], check_result["url"]
    p = m["patch"]
    file_url = p.get("url") or url.rsplit("/", 1)[0] + "/" + p["file"]
    out = updates_dir() / "下载" / p["file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    req = urllib.request.Request(file_url, headers={"User-Agent": "cfos-updater"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or p.get("size") or 0)
        done = 0
        while True:
            b = r.read(256 * 1024)
            if not b:
                break
            f.write(b); h.update(b); done += len(b)
            if progress:
                progress(done, max(total, done), f"正在下载补丁 {done / 1e6:.1f} / {total / 1e6:.1f} MB")
    if h.hexdigest() != p["sha256"]:
        out.unlink(missing_ok=True)
        raise ValueError("下载的补丁文件不完整（校验码不一致），请重试，或在 GitHub 上手动下载。")
    return out


# ------------------------------------------------------------------ 安装
def read_patch_info(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        try:
            info = json.loads(z.read("patch.json").decode("utf-8"))
        except KeyError:
            raise ValueError("这不是 cFos 脑片分析的补丁包（里面没有 patch.json）。") from None
    for k in ("version", "min_base", "files"):
        if k not in info:
            raise ValueError(f"补丁包信息不完整（缺少 {k}）。")
    return info


def install_patch(zip_path: Path) -> dict:
    """把补丁解压到“更新”文件夹并逐个核对文件，成功后设为当前补丁（重启后生效）。返回补丁信息。"""
    zip_path = Path(zip_path)
    info = read_patch_info(zip_path)
    v, base = info["version"], base_version()
    if vtuple(info["min_base"]) > vtuple(base):
        raise ValueError(f"这个补丁（{v}）需要先安装 {info['min_base']} 或更新的完整安装包，"
                         f"你现在装的是 {base}。请到 GitHub 下载完整安装包：{FULL_PAGE}")
    if vtuple(v) <= vtuple(base):
        raise ValueError(f"这个补丁（{v}）不比你安装的版本（{base}）新，不需要安装。")
    root = updates_dir()
    tmp = root / f"{v}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for name, sha in info["files"].items():
            if name.startswith("/") or ".." in Path(name).parts or not name.startswith("app/"):
                raise ValueError(f"补丁包里有不允许的文件路径：{name}")
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(f"补丁包已损坏（{name} 校验不一致），请重新下载。")
            out = tmp / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    if not (tmp / "app" / "cfos" / "__init__.py").exists():
        raise ValueError("补丁包里没有程序代码。")
    if _read_version_file(tmp / "app" / "cfos" / "__init__.py") != v:
        raise ValueError("补丁包的版本信息不一致。")
    final = root / v
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    st = _load_state()
    for k in ("disabled", "disabled_reason", "failed_starts"):
        st.pop(k, None)
    st.update(version=v, dir=v, min_base=info["min_base"], notes=info.get("notes", ""))
    _save_state(st)
    _cleanup(keep={v, base})
    return info


def _cleanup(keep: set):
    """只保留当前补丁（旧补丁文件夹删掉，下载的 zip 也删掉）。"""
    root = updates_dir()
    for d in root.iterdir() if root.exists() else []:
        if d.is_dir() and d.name not in keep and d.name not in ("下载",) and (d / "app").exists():
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(root / "下载", ignore_errors=True)


def restart_app():
    """用同样的方式重新启动软件（新进程），由调用者随后退出当前进程。"""
    import subprocess
    main_py = base_app_dir() / "main.py"
    exe = sys.executable
    if sys.platform == "win32" and exe.lower().endswith("python.exe"):
        w = Path(exe).with_name("pythonw.exe")
        exe = str(w) if w.exists() else exe
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen([exe, str(main_py)] + [a for a in sys.argv[1:] if a != "--selftest"],
                     cwd=str(base_app_dir()), close_fds=True, **kw)


def version_text() -> str:
    p = active_patch()
    if p:
        return f"v{p['version']}（安装版本 v{base_version()} + 补丁）"
    return f"v{base_version()}"
