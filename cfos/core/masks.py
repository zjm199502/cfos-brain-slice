"""导出细胞 mask：每张脑片、每个通道一张标签图（TIFF），分辨率为原图的 1/3。

标签值 = 细胞编号 + 1（0 为背景）。不同通道中同一个细胞（共标）使用同一个编号，
因此两张 mask 中数值相同的区域就是同一个细胞。只需要黑白 mask 时，把“>0”的像素当作细胞即可。
"""
from __future__ import annotations

import datetime as _dt
import csv
import json
import os
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import polygon as skpoly

from .project import Project, processed_section_names
from .slide_io import open_slide

README = """细胞 mask 说明
================
文件名：<图片>_<脑片>_<通道>.tif
分辨率：原始图像的 1/{factor}，各图像的像素尺寸已写入 TIFF 的分辨率信息。
坐标范围：对应该脑片在原图中的方框（左上角在原图中的像素坐标见 脑片位置.csv）
像素值：0 = 背景；>0 = 细胞编号 + 1。
同一个细胞在不同通道的 mask 中编号相同（例如 DAPI 与 NeuN 共标的细胞），可用来核对共标结果。
在 Fiji/ImageJ 中直接打开即可；只需要黑白 mask 时，阈值设为 1 以上。
"""


def _mask_filename(image_name, section_name, channel_name):
    safe = "".join(ch if ch not in '\\/:*?"<>|' else "_" for ch in channel_name)
    if any(ch in str(image_name) + str(section_name) for ch in '/\\:'):
        raise ValueError("图片或脑片名称含有路径分隔符。")
    return f"{image_name}_{section_name}_{safe}.tif"


def _previous_masks(out_dir, channels):
    """Read only filenames listed in a mask export, including legacy position tables."""
    positions = out_dir / "脑片位置.csv"
    readme = out_dir / "说明.txt"
    if not (positions.is_file() and readme.is_file()):
        return set()
    files = set()
    try:
        with readme.open(encoding="utf-8") as f:
            if f.readline().strip() != "细胞 mask 说明":
                return set()
        with positions.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                image, section = row["图片"], row["脑片"]
                names = (json.loads(row["mask文件"]) if "mask文件" in row else
                         [_mask_filename(image, section, c) for c in channels])
                for name in names:
                    if (isinstance(name, str) and Path(name).name == name and
                            not any(ch in name for ch in '/\\:') and
                            name.startswith(f"{image}_{section}_") and name.endswith(".tif")):
                        files.add(name)
    except (KeyError, OSError, ValueError, TypeError, csv.Error):
        return set()
    return files


def _write_section_masks(out_dir, image, section, cells, names, factor, pixel_um):
    x0, y0, w, h = section.bbox0
    width, height = int(np.ceil(w / factor)), int(np.ceil(h / factor))
    ids = cells["merged_id"].to_numpy() + 1 if "merged_id" in cells else np.arange(1, len(cells) + 1)
    files = []
    for c, nm in enumerate(names):
        label = np.zeros((height, width), np.uint32)
        if len(cells):
            selected = (cells["channel"] == c).to_numpy()
            for contour, cell_id in zip(cells["contour"].to_numpy()[selected], ids[selected]):
                if len(contour) >= 3:
                    a = np.asarray(contour, float)
                    rr, cc = skpoly((a[:, 1] - y0) / factor, (a[:, 0] - x0) / factor, label.shape)
                    label[rr, cc] = cell_id
        name = _mask_filename(image.key, section.name, nm)
        tifffile.imwrite(out_dir / name, label, compression="zlib",
                         resolution=(1e4 / pixel_um, 1e4 / pixel_um), resolutionunit="CENTIMETER")
        files.append(name)
    return files


def export_masks(project: Project, out_dir=None, factor: int = 3, progress=None) -> Path:
    if out_dir is None:
        out_dir = project.folder / f"细胞mask_{_dt.datetime.now():%Y%m%d_%H%M}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = project.channel_names
    previous = _previous_masks(out_dir, names)
    written = set()
    rows = []
    from .pipeline import drop_excluded
    for i, im in enumerate(project.images):
        df = project.load_cells(im)
        processed = processed_section_names(im, df)
        if not processed:
            continue
        df = drop_excluded(df, im)
        sl = open_slide(im.path)
        try:
            px_used = sl.pixel_size_um * factor
        finally:
            sl.close()
        for s in im.sections:
            if s.name not in processed:
                continue
            if progress:
                progress(i, len(project.images), f"正在保存 mask：{im.key} {s.name}")
            g = df[df["section"] == s.name] if len(df) else df
            files = _write_section_masks(out_dir, im, s, g, names, factor, px_used)
            written.update(map(os.path.normcase, files))
            x0, y0, w, h = s.bbox0
            rows.append({"图片": im.key, "脑片": s.name, "左上角X(原图像素)": x0, "左上角Y(原图像素)": y0,
                         "宽(原图像素)": w, "高(原图像素)": h, "mask文件": json.dumps(files, ensure_ascii=False)})
    for name in previous:
        if os.path.normcase(name) in written:
            continue
        target = out_dir / name
        if target.is_file() and not target.is_symlink():
            target.unlink()
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "脑片位置.csv", index=False, encoding="utf-8-sig")
    (out_dir / "说明.txt").write_text(README.format(factor=factor), encoding="utf-8")
    return out_dir
