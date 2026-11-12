"""“导出诊断包”：出问题时，把排查需要的信息一键打成一个 zip，发给开发者。

包含：运行日志（软件的 + 项目里的）、项目设置文件（cfos_project.json，不含图片和细胞表格）、
电脑和软件版本信息、各图片的基本信息（大小、层数、通道），以及最近一次批量处理的检查清单。
不包含任何图片像素和细胞数据，文件一般只有几百 KB。"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
import zipfile
from pathlib import Path


def _versions() -> dict:
    out = {}
    try:
        from importlib import metadata
        for p in ("numpy", "scipy", "pandas", "tensorflow", "tensorflow-cpu", "keras", "stardist", "onnxruntime",
                  "napari", "PySide6", "itk-elastix", "DeepSlice", "tifffile", "imagecodecs", "scikit-image"):
            try:
                out[p] = metadata.version(p)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


def system_info() -> dict:
    import cfos
    info = {"软件版本": getattr(cfos, "__version__", "?"),
            "时间": _dt.datetime.now().isoformat(timespec="seconds"),
            "系统": platform.platform(), "Python": sys.version.split()[0],
            "处理器": platform.processor() or platform.machine(),
            "安装位置": str(Path(cfos.__file__).resolve().parent.parent)}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["内存"] = f"{vm.total / 2**30:.1f} GB（可用 {vm.available / 2**30:.1f} GB）"
        du = psutil.disk_usage(str(Path.home()))
        info["主目录所在磁盘"] = f"剩余 {du.free / 2**30:.1f} GB / 共 {du.total / 2**30:.1f} GB"
    except Exception:  # noqa: BLE001
        pass
    try:
        from .cpu import describe
        info["CPU 使用"] = describe()
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import accel
        info["细胞识别神经网络"] = accel.backend_text()
    except Exception:  # noqa: BLE001
        pass
    for k in ("TF_ENABLE_ONEDNN_OPTS", "CFOS_MAX_CORES", "CFOS_WORKERS", "CFOS_NET", "OMP_NUM_THREADS"):
        if os.environ.get(k) is not None:
            info.setdefault("环境变量", {})[k] = os.environ[k]
    info["组件版本"] = _versions()
    return info


def _image_info(project) -> list[dict]:
    rows = []
    from .slide_io import open_slide
    for im in project.images:
        r = {"图片": im.key, "路径": im.path, "存在": Path(im.path).exists(), "脑片数": len(im.sections)}
        try:
            r["文件大小 MB"] = round(Path(im.path).stat().st_size / 2**20, 1)
            sl = open_slide(im.path)
            try:
                r.update({"像素大小 µm": sl.pixel_size_um, "层数": sl.n_levels, "z 层数": sl.n_z,
                          "第0层大小": list(sl.level_shapes[0]), "通道": list(sl.channels)})
            finally:
                sl.close()
        except Exception as e:  # noqa: BLE001
            r["读取出错"] = f"{type(e).__name__}: {e}"
        rows.append(r)
    return rows


def _add_tail(z: zipfile.ZipFile, src: Path, arcname: str, max_bytes: int = 3_000_000):
    """日志可能很大，只放最后一部分。"""
    try:
        size = src.stat().st_size
        with open(src, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            z.writestr(arcname, f.read())
    except OSError:
        pass


def make_bundle(project=None, out_dir: Path | None = None, note: str = "") -> Path:
    """生成诊断包，返回 zip 路径。默认放在项目文件夹（没有项目时放在日志文件夹）。"""
    from .log import log_dir, log
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir or (project.folder if project is not None else log_dir()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"诊断包_{stamp}.zip"
    for h in log.handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("电脑和软件信息.json", json.dumps(system_info(), ensure_ascii=False, indent=2, default=str))
        if note:
            z.writestr("问题描述.txt", note)
        ld = log_dir()
        for f in sorted(ld.glob("运行日志.txt*")):
            _add_tail(z, f, f"软件日志/{f.name}")
        st = ld / "自检结果.txt"
        if st.exists():
            z.write(st, "软件日志/自检结果.txt")
        if project is not None:
            pf = Path(project.folder)
            for n in ("cfos_project.json", "cfos_project.json.bak"):
                if (pf / n).exists():
                    z.write(pf / n, f"项目/{n}")
            for f in sorted(pf.glob("运行日志.txt*")):
                _add_tail(z, f, f"项目/{f.name}")
            try:
                z.writestr("项目/图片信息.json", json.dumps(_image_info(project), ensure_ascii=False, indent=2, default=str))
            except Exception as e:  # noqa: BLE001
                z.writestr("项目/图片信息.json", json.dumps({"出错": str(e)}, ensure_ascii=False))
            for f in list(pf.glob("*检查清单*.xlsx"))[:3]:
                if f.stat().st_size < 5_000_000:
                    z.write(f, f"项目/{f.name}")
    log.info("已导出诊断包 %s", out)
    return out
