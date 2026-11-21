"""Assemble the arm64 app using the native runtime in the published v0.8 Mac ZIP.

Run with Python 3.11+ and pass the original ZIP, the converted DeepSlice models
directory and the output directory. ZIP symlinks and Unix modes survive on Windows.
"""
import argparse
import hashlib
from pathlib import Path
import plistlib
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfos import __version__

BASE_SHA = "3979849aa708df7042a32e6acfdecddf99edafdab4648ca67716f1b7e88deab1"
OLD = "cFos脑片分析.app/Contents/"
NEW = "FosFolio脑片图册.app/Contents/"
MODELS = ("deepslice_Allen_Mixed_Best", "deepslice_Synthetic_data_final")
GUIDE = """将 FosFolio脑片图册.app 拖入“应用程序”文件夹后双击打开。要求 Apple 芯片及 macOS 15 或更高版本。
首次打开如提示无法验证开发者，请在“系统设置 → 隐私与安全性”中选择“仍要打开”。
首次识别细胞前，在“④ 细胞识别”中点击“安装 / 修复 Cellpose 4 运行环境”；此步骤需要网络。
安装完成后点击“测试 Cellpose 设备”，查看本机实际使用的 Apple GPU（Metal / PyTorch MPS）或 CPU。
若无法启动，可双击同目录的“诊断启动.command”查看错误信息。
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


def app_files(archive, models):
    for file in sorted((ROOT / "cfos").rglob("*")):
        if file.is_file() and file.suffix in {".py", ".svg", ".png", ".txt"}:
            write_file(archive, NEW + "Resources/app/" + file.relative_to(ROOT).as_posix(), file.read_bytes())
    for name in ("main.py", "README.md", "THIRD_PARTY.md", "requirements.txt"):
        write_file(archive, NEW + "Resources/app/" + name, (ROOT / name).read_bytes())
    notes = Path("packaging", "更新公告", f"{__version__}.md")
    notes_data = (ROOT / notes).read_bytes()
    write_file(archive, NEW + "Resources/app/" + notes.as_posix(), notes_data)
    write_file(archive, "更新公告.md", notes_data)
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
    write_file(archive, "安装说明.txt", GUIDE.encode())


def build(base, models, out):
    with base.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != BASE_SHA:
            raise ValueError("Expected the official v0.8 Mac Apple Silicon complete ZIP")
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"FosFolio-{__version__}-macOS-arm64.zip"
    temp = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(base, metadata_encoding="utf-8") as source, zipfile.ZipFile(temp, "w") as archive:
        for original in source.infolist():
            if not original.filename.startswith(OLD):
                continue
            part = original.filename[len(OLD):]
            if not runtime_or_resource(part):
                continue
            entry = zipfile.ZipInfo(NEW + part, original.date_time)
            entry.create_system = 3
            entry.external_attr = original.external_attr
            entry.compress_type = original.compress_type
            with source.open(original) as src, archive.open(entry, "w") as dest:
                shutil.copyfileobj(src, dest, 1024 * 1024)
        app_files(archive, models)
    with zipfile.ZipFile(temp) as archive:
        damaged = archive.testzip()
        if damaged:
            raise ValueError(f"CRC mismatch: {damaged}")
    temp.replace(target)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("base", "models", "out"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    build(args.base, args.models, args.out)
