"""Build the shared in-app update for clients predating separate 0.8 and 0.9 feeds.

Usage: python packaging/make_series_update.py <0.8-channel-dir> <0.9-channel-dir> <out>
Publish the two native channels under updates/v0.8 and updates/v0.9. Publish this
archive and its manifest at the original updates/ URL in the same commit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


# The installed launcher sets CFOS_BASE_APP before its first import of cfos,
# including in spawned workers. Select a whole native package before importing
# any application code or backend. The literal version is read by old updaters.
SELECTOR = '''"""Select the application for this installation's bundled dependencies."""
__version__ = "{version}"

import importlib.util
import os
from pathlib import Path
import sys

base = Path(os.environ["CFOS_BASE_APP"]) / "cfos" / "__init__.py"
if base.resolve() == Path(__file__).resolve():
    raise ImportError("检测到兼容补丁覆盖了安装目录。请恢复原系列的基础安装，再在“帮助 → 从文件安装补丁”中选择 ZIP。")
for line in base.read_text(encoding="utf-8").splitlines():
    if line.startswith("__version__"):
        base_version = line.partition("=")[2].strip().strip("'\\\"")
        break
else:
    raise ImportError("无法读取 FosFolio 安装版本")
family = ".".join(base_version.split(".")[:2])
series = {{"0.7": "0.8", "0.8": "0.8", "0.9": "0.9"}}.get(family)
if series is None:
    raise ImportError(f"未知的 FosFolio 安装系列：{{base_version}}")
app = Path(__file__).resolve().parents[1] / "_series" / series
sys.path.insert(0, str(app))
os.environ["CFOS_PATCH_APP"] = str(app)
spec = importlib.util.spec_from_file_location(__name__, app / "cfos" / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[__name__] = package
spec.loader.exec_module(package)
'''


def read_native(folder: Path, series: str):
    manifest = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
    version = manifest["version"]
    if not version.startswith(series + "."):
        raise ValueError(f"{folder} 版本与 {series} 不匹配")
    descriptor = manifest["patch"]
    package = folder / descriptor["file"]
    if hashlib.sha256(package.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"{package} SHA-256 不一致")
    with zipfile.ZipFile(package) as source:
        info = json.loads(source.read("patch.json"))
        if info["version"] != version or info["min_base"] != manifest["min_base"]:
            raise ValueError(f"{package} 兼容版本不一致")
        files = {}
        for name, expected in info["files"].items():
            rel = Path(name)
            if not name.startswith("app/") or ".." in rel.parts or "\\" in name:
                raise ValueError(f"补丁文件路径无效：{name}")
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"{name} SHA-256 不一致")
            files[name.removeprefix("app/")] = data
    return manifest, files


def build(legacy_dir: Path, current_dir: Path, out: Path):
    legacy, old_files = read_native(legacy_dir, "0.8")
    current, new_files = read_native(current_dir, "0.9")
    if tuple(int(part) for part in current["min_base"].split(".")[:2]) < (0, 9):
        raise ValueError("v0.9.x 原生包必须要求安装版本 ≥ 0.9.0；旧客户端依靠该门槛阻止跨系列安装。")
    version = current["version"]
    name = f"cfos-update-{legacy['version']}-and-{version}.zip"
    notes = (f"# 按当前安装系列更新\n\n"
             f"如果你正在用 v0.8.x，点「现在更新」只会安装 v{legacy['version']}，继续使用 StarDist。"
             f"旧更新窗口顶部统一显示 v{version}。重启后可在「帮助 → 关于」核对实际版本。\n\n"
             f"v0.9.x 安装版会收到 v{version}，继续使用 Cellpose 4。\n\n"
             "两个系列均可在「⑧ 导出 Excel → 脑区范围 → 自选脑区…」搜索并勾选多个 Allen 脑区。"
             "保存后，手动导出和一键批处理都使用本项目的清单。")
    data = {"app/cfos/__init__.py": SELECTOR.format(version=version).encode("utf-8")}
    for series, files in (("0.8", old_files), ("0.9", new_files)):
        for path, content in files.items():
            data[f"app/_series/{series}/{path}"] = content
    info = {"version": version, "min_base": legacy["min_base"], "date": current["date"], "notes": notes,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}}
    out.mkdir(parents=True, exist_ok=True)
    package = out / name
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as dest:
        dest.writestr("patch.json", json.dumps(info, ensure_ascii=False, indent=1))
        for path, content in data.items():
            dest.writestr(path, content)
        dest.writestr("更新公告.md", notes)
        dest.writestr("安装说明.txt", "请在 FosFolio 的“帮助 → 检查更新”中点击“现在更新”，或在“从文件安装补丁”中选择同系列的 ZIP。请勿解压覆盖安装目录。")
    manifest = {k: info[k] for k in ("version", "min_base", "date", "notes")}
    manifest["patch"] = {"file": name, "size": package.stat().st_size,
                         "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
    # Old clients only reach this fallback when they cannot install the bridge.
    # Link them to the legacy-only page rather than a newer runtime installer.
    manifest["full_installer"] = legacy["full_installer"]
    (out / "latest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("legacy", "current", "out"):
        parser.add_argument(arg, type=Path)
    options = parser.parse_args()
    build(options.legacy, options.current, options.out)
