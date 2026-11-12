"""导出细胞 mask：每张脑片、每个通道一张标签图（TIFF），分辨率为原图的 1/3。

标签值 = 细胞编号 + 1（0 为背景）。不同通道中同一个细胞（共标）使用同一个编号，
因此两张 mask 中数值相同的区域就是同一个细胞。只需要黑白 mask 时，把“>0”的像素当作细胞即可。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import polygon as skpoly

from .project import Project
from .slide_io import open_slide

README = """细胞 mask 说明
================
文件名：<图片>_<脑片>_<通道>.tif
分辨率：原始图像的 1/{factor}（每像素 {px} µm，已写入 TIFF 的分辨率信息）
坐标范围：对应该脑片在原图中的方框（左上角在原图中的像素坐标见 脑片位置.csv）
像素值：0 = 背景；>0 = 细胞编号 + 1。
同一个细胞在不同通道的 mask 中编号相同（例如 DAPI 与 NeuN 共标的细胞），可用来核对共标结果。
在 Fiji/ImageJ 中直接打开即可；只需要黑白 mask 时，阈值设为 1 以上。
"""


def export_masks(project: Project, out_dir=None, factor: int = 3, progress=None) -> Path:
    if out_dir is None:
        out_dir = project.folder / f"细胞mask_{_dt.datetime.now():%Y%m%d_%H%M}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = project.channel_names
    rows = []
    jobs = []
    for im in project.images:
        df = project.load_cells(im)
        if df is None or df.empty:
            continue
        from .pipeline import drop_excluded
        df = drop_excluded(df, im)
        for s in im.sections:
            if s.enabled:
                jobs.append((im, s, df))
    px_used = None
    for k, (im, s, df) in enumerate(jobs):
        if progress:
            progress(k, len(jobs), f"正在保存 mask：{im.key} {s.name}")
        sl = open_slide(im.path)
        px = sl.pixel_size_um
        sl.close()
        px_used = px * factor
        x0, y0, w, h = s.bbox0
        W, H = int(np.ceil(w / factor)), int(np.ceil(h / factor))
        g = df[df["section"] == s.name]
        ids = g["merged_id"].to_numpy() + 1 if "merged_id" in g.columns else np.arange(1, len(g) + 1)
        for c, nm in enumerate(names):
            lab = np.zeros((H, W), np.uint32)
            sel = (g["channel"] == c).to_numpy()
            for cont, cid in zip(g["contour"].to_numpy()[sel], ids[sel]):
                if len(cont) < 3:
                    continue
                a = np.asarray(cont, float)
                rr, cc = skpoly((a[:, 1] - y0) / factor, (a[:, 0] - x0) / factor, (H, W))
                lab[rr, cc] = cid
            safe = "".join(ch if ch not in '\\/:*?"<>|' else "_" for ch in nm)
            tifffile.imwrite(out_dir / f"{im.key}_{s.name}_{safe}.tif", lab, compression="zlib",
                             resolution=(1e4 / px_used, 1e4 / px_used), resolutionunit="CENTIMETER")
        rows.append({"图片": im.key, "脑片": s.name, "左上角X(原图像素)": x0, "左上角Y(原图像素)": y0,
                     "宽(原图像素)": w, "高(原图像素)": h})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "脑片位置.csv", index=False, encoding="utf-8-sig")
    (out_dir / "说明.txt").write_text(README.format(factor=factor, px=round(px_used or 0, 3)), encoding="utf-8")
    return out_dir
