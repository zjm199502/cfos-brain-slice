"""Create and check a single ZIP64 containing every disk of the offline Setup."""
import hashlib
from pathlib import Path
import sys
import zipfile


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def package(setup, stage, bundle):
    disks = sorted(setup.parent.glob(setup.stem + "-*.bin"))
    expected = {f"{setup.stem}-{index}.bin" for index in range(1, len(disks) + 1)}
    if not disks or {p.name for p in disks} != expected:
        raise ValueError("Incomplete or unexpectedly named Inno Setup disk set")
    installed_bytes = sum(p.stat().st_size for folder in (stage, bundle)
                          for p in folder.rglob("*") if p.is_file())
    wheel_bytes = 0
    for path in (bundle / "wheels").glob("*.whl"):
        with zipfile.ZipFile(path) as wheel:
            wheel_bytes += sum(item.file_size for item in wheel.infolist())
    guide = setup.parent / "Windows离线安装说明.txt"
    compact_note = ("精简离线包需要先展开临时文件并还原官方软件包，请在临时目录所在磁盘预留 6 GiB。"
                    "首次安装的准备时间会更长，完成后自动清理临时文件。\n"
                    if "-Compact-" in setup.stem else "")
    guide.write_text(
        f"FosFolio Windows x64 离线安装\n\n"
        f"1. 把 ZIP 完整解压到同一文件夹，保留所有 .bin 分卷的原名。\n"
        f"2. 退出已打开的 FosFolio，双击 {setup.name}。已有版本可沿用原安装目录升级。\n"
        f"3. 向导会安装程序、从随包文件配置 Cellpose 4 和 PyTorch，并用随包模型进行 CPU 推理自检。"
        f"首次配置可能需要数分钟，可全程断网。\n"
        f"4. 打开「④ 细胞识别」，可点「测试 Cellpose 设备」核对 CPU 或兼容的 NVIDIA GPU。"
        f"软件可从安装目录内的离线组件修复运行环境。\n\n"
        f"{compact_note}"
        f"安装目录文件合计约 {installed_bytes / 1024**3:.1f} GiB。当前用户组件目录另需约 "
        f"{wheel_bytes / 1024**3:.1f} GiB 来展开依赖，并需预留安装临时空间。\n",
        encoding="utf-8-sig")
    files = [setup, *disks, guide]
    checksums = {p.name: sha256(p) for p in files}
    sums = setup.parent / "SHA256SUMS.txt"
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8")
    checksums[sums.name] = sha256(sums)
    archive = setup.with_name(setup.stem.removesuffix("-Setup") + ".zip")
    partial = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zip_out:
        for path in (*files, sums):
            zip_out.write(path, path.name)
    with zipfile.ZipFile(partial) as zip_in:
        if set(zip_in.namelist()) != checksums.keys():
            raise RuntimeError("Offline installation media are incomplete")
        for name, digest in checksums.items():
            with zip_in.open(name) as entry:
                if hashlib.file_digest(entry, "sha256").hexdigest() != digest:
                    raise RuntimeError(f"Offline media verification failed: {name}")
    partial.replace(archive)
    print(f"{sha256(archive)}  {archive.name}", flush=True)


if __name__ == "__main__":
    package(*(Path(p) for p in sys.argv[1:4]))
