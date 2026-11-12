"""从一张真实玻片中截取一张脑片，生成教学用示例图（0.5 µm/像素，多分辨率 OME-TIFF，约十几 MB）。
用法：python tools/make_demo.py <qptiff> <项目文件夹(已拆分脑片)> <脑片名> <输出 .ome.tif>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from cfos.core.project import Project
from cfos.core.slide_io import open_slide

src, proj, sec, out = sys.argv[1:5]
p = Project.load(proj)
im = next(i for i in p.images if Path(i.path).name == Path(src).name)
s = next(x for x in im.sections if x.name == sec)
sl = open_slide(src)
lv = sl.level_for_pixel_size(0.5)
ds = sl.downsample(lv)
x, y, w, h = s.bbox0
m = int(300 / sl.pixel_size_um)  # 四周留 300 µm 空白，方便练习“拆分脑片”
a = sl.read(lv, (x - m, y - m, w + 2 * m, h + 2 * m))
px = sl.pixel_size_um * ds
names = [c.name for c in sl.channels]
print(a.shape, a.dtype, px, names)
Path(out).parent.mkdir(parents=True, exist_ok=True)
levels = 5
with tifffile.TiffWriter(out, bigtiff=False, ome=True) as tw:
    opts = dict(tile=(512, 512), compression="jpeg", compressionargs={"level": 80}, photometric="minisblack")
    tw.write(a, subifds=levels - 1, metadata={"axes": "CYX", "PhysicalSizeX": px, "PhysicalSizeY": px,
                                                "Channel": {"Name": names}}, resolution=(1e4 / px, 1e4 / px),
             resolutionunit="CENTIMETER", **opts)
    b = a
    for k in range(1, levels):
        H, W = b.shape[1] // 2 * 2, b.shape[2] // 2 * 2
        b = b[:, :H, :W].reshape(b.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4)).astype(a.dtype)
        tw.write(b, subfiletype=1, resolution=(1e4 / px / 2 ** k, 1e4 / px / 2 ** k), resolutionunit="CENTIMETER", **opts)
print(Path(out).stat().st_size / 1e6, "MB")
t = open_slide(out); print(t.channels, t.pixel_size_um, t.level_shapes)
