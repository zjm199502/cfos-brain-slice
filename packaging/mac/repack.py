"""Assemble the arm64 app from a previously verified Mac release.

Run with Python 3.11+ and pass the original ZIP, the converted DeepSlice models
directory and the output directory. For the offline edition pass the v0.9.0
release ZIP, the output directory and --offline-components <mac-arm64 folder>.
ZIP symlinks and Unix modes survive on Windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfos import __version__
from cfos.core.cellpose_setup import CELLPOSE_VERSION, WEIGHTS_BYTES, WEIGHTS_SHA256

BASE_SHA = "3979849aa708df7042a32e6acfdecddf99edafdab4648ca67716f1b7e88deab1"
RELEASE_090_SHA = "141ead9ec5e363786298f13c5c083264e344670e056d6b4fcde57efa7250e50a"
OLD = "cFos脑片分析.app/Contents/"
NEW = "FosFolio脑片图册.app/Contents/"
MODELS = ("deepslice_Allen_Mixed_Best", "deepslice_Synthetic_data_final")
GUIDE = """将 FosFolio脑片图册.app 拖入“应用程序”文件夹后双击打开。要求 Apple 芯片及 macOS 15 或更高版本。
首次打开如提示无法验证开发者，请在“系统设置 → 隐私与安全性”中选择“仍要打开”。
首次识别细胞前，在“④ 细胞识别”中点击“安装 / 修复 Cellpose 4 运行环境”；此步骤需要网络。
安装完成后点击“测试 Cellpose 设备”，查看本机实际使用的 Apple GPU（Metal / PyTorch MPS）或 CPU。
若无法启动，可双击同目录的“诊断启动.command”查看错误信息。
"""
OFFLINE_GUIDE = """FosFolio 脑片图册 · Apple 芯片 Mac 离线版

1. 在 macOS 15 或更高版本上解压 ZIP。先退出旧版本，将 FosFolio脑片图册.app 拖入“应用程序”，再从那里打开。
2. 首次打开时按提示安装 Cellpose 4。应用内已附带 Cellpose、PyTorch、全部依赖和 Cellpose-SAM v2 权重，可在断网时安装。已安装的用户可以在“④ 细胞识别”中安装或修复运行环境。
3. 在“④ 细胞识别”点击“测试 Cellpose 设备”。软件会运行一次模型，显示实际使用的 Apple GPU（Metal / PyTorch MPS）或 CPU。

首次打开如提示无法验证开发者，可在“系统设置 → 隐私与安全性”中选择“仍要打开”。
"""


def runtime_or_resource(path):
    if path.startswith(("Resources/app/resources/atlas/", "Resources/app/resources/demo/",
                        "Resources/app/resources/teaching/")):
        return True
    if not path.startswith("Resources/python/"):
        return False
    if "/site-packages/" in path:
        package = path.split("/site-packages/", 1)[1].partition("/")[0].lower()
        retired = ("stardist", "csbdeep", "tensorflow", "deepslice", "keras")
        if any(package == name or package.startswith(name + "-") for name in retired):
            return False
        if package == "bin" and "/bin/stardist-" in path:
            return False
    return True


def write_file(archive, path, data, mode=0o644):
    info = zipfile.ZipInfo(path)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def app_files(archive, models, guide=GUIDE):
    for file in sorted((ROOT / "cfos").rglob("*")):
        if file.is_file() and file.suffix in {".py", ".svg", ".png", ".txt"}:
            write_file(archive, NEW + "Resources/app/" + file.relative_to(ROOT).as_posix(), file.read_bytes())
    for name in ("main.py", "README.md", "THIRD_PARTY.md", "requirements.txt"):
        write_file(archive, NEW + "Resources/app/" + name, (ROOT / name).read_bytes())
    notes = Path("packaging", "更新公告", f"{__version__}.md")
    notes_data = (ROOT / notes).read_bytes()
    write_file(archive, NEW + "Resources/app/" + notes.as_posix(), notes_data)
    write_file(archive, "更新公告.md", notes_data)
    if models is not None:
        for stem in MODELS:
            for suffix in (".onnx", ".sig"):
                file = models / (stem + suffix)
                write_file(archive, NEW + "Resources/app/resources/deepslice/" + file.name, file.read_bytes())
    mac = Path(__file__).resolve().parent
    info = plistlib.loads((mac / "Info.plist").read_bytes())
    info.update(CFBundleVersion=__version__, CFBundleShortVersionString=__version__)
    write_file(archive, NEW + "Info.plist", plistlib.dumps(info))
    write_file(archive, NEW + "PkgInfo", b"APPL????")
    write_file(archive, NEW + "Resources/cfos.icns", (mac / "cfos.icns").read_bytes())
    write_file(archive, NEW + "MacOS/cfos", (mac / "launcher.sh").read_bytes().replace(b"\r\n", b"\n"), 0o755)
    write_file(archive, "诊断启动.command", (mac / "诊断启动.command").read_bytes().replace(b"\r\n", b"\n"), 0o755)
    write_file(archive, "安装说明.txt", guide.encode())


def write_offline_components(archive, components):
    manifest_path = components / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest["format"], manifest["platform"], manifest["python"], manifest["cellpose"]) != (
        1, "mac-arm64", "3.13", CELLPOSE_VERSION
    ):
        raise ValueError("Cellpose offline manifest does not match the Mac Python runtime")
    prefix = NEW + "Resources/app/resources/cellpose4/"
    files = [("cpsam_v2", WEIGHTS_BYTES, WEIGHTS_SHA256)]
    wheels = manifest["wheels"]
    names = {wheel["file"] for wheel in wheels}
    if not names or len(names) != len(wheels):
        raise ValueError("Missing or duplicate offline wheels")
    for wheel in wheels:
        name = wheel["file"]
        if "/" in name or "\\" in name or ":" in name or not name.endswith(".whl"):
            raise ValueError(f"Invalid wheel name: {name}")
        files.append(("wheels/" + name, wheel["size"], wheel["sha256"]))
    for relpath, size, checksum in files:
        file = components / relpath
        if file.stat().st_size != size:
            raise ValueError(f"Size mismatch: {file}")
        info = zipfile.ZipInfo(prefix + relpath)
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.file_size = size
        info.compress_type = zipfile.ZIP_STORED if relpath.endswith(".whl") else zipfile.ZIP_DEFLATED
        digest = hashlib.sha256()
        with file.open("rb") as source, archive.open(info, "w") as dest:
            while block := source.read(1024 * 1024):
                digest.update(block)
                dest.write(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"SHA-256 mismatch: {file}")
    for name in ("NOTICE.txt", "LICENSE-Cellpose.txt"):
        write_file(archive, prefix + name, (components / name).read_bytes())
    write_file(archive, prefix + manifest_path.name, manifest_path.read_bytes())


def build(base, models, out, offline_components=None):
    offline = offline_components is not None
    if offline != (models is None):
        raise ValueError("Provide either the converted models directory or --offline-components")
    with base.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != (RELEASE_090_SHA if offline else BASE_SHA):
            raise ValueError("Incorrect base release ZIP")
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"FosFolio-{__version__}-macOS-arm64{'-Offline' if offline else ''}.zip"
    with tempfile.TemporaryDirectory(dir=out, prefix=".repack-") as staging:
        temp = Path(staging) / target.name
        with zipfile.ZipFile(base, metadata_encoding="utf-8") as source, zipfile.ZipFile(temp, "w") as archive:
            prefix = NEW if offline else OLD
            for original in source.infolist():
                if not original.filename.startswith(prefix):
                    continue
                part = original.filename[len(prefix):]
                if not runtime_or_resource(part) and not (offline and part.startswith("Resources/app/resources/deepslice/")):
                    continue
                entry = zipfile.ZipInfo(NEW + part, original.date_time)
                entry.create_system = 3
                entry.external_attr = original.external_attr
                entry.compress_type = original.compress_type
                with source.open(original) as src, archive.open(entry, "w") as dest:
                    shutil.copyfileobj(src, dest, 1024 * 1024)
            app_files(archive, models, OFFLINE_GUIDE if offline else GUIDE)
            if offline:
                write_offline_components(archive, offline_components)
        with zipfile.ZipFile(temp) as archive:
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"CRC mismatch: {damaged}")
        temp.replace(target)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("models", nargs="?", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--offline-components", type=Path)
    args = parser.parse_args()
    build(args.base, args.models, args.out, args.offline_components)
