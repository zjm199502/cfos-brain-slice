"""质量检查冒烟测试：在真实脑片上人为制造模糊、错位、缺损、过曝，确认能被标出。
用法：python tests/quality_smoke.py <项目文件夹>"""
import copy, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage as ndi
from cfos.core.project import Project
from cfos.core.slide_io import open_slide
from cfos.core import quality as Q, registration as R
from cfos.core.coloc import nucleus_channel

p = Project.load(sys.argv[1])
im = p.images[0]; s = im.sections[0]
sl = open_slide(im.path); nuc = nucleus_channel(p.channel_names)
t = time.time()
base = Q.section_quality(p, im, sl, s)
print("正常:", {k: v for k, v in base.items() if k != "signature"}, f"{time.time() - t:.1f}s", Q.grade(base))

# 1. 错位
s2 = copy.deepcopy(s); r = R.Registration.from_dict(s2.registration); r.shift_x += 0.12; r.ap_shift += 30; r.nonlinear = False
s2.registration = r.to_dict()
d, _ = Q.registration_metrics(p, im, sl, s2, nuc); print("错位 dice:", d)
assert d < base["dice"] - 0.05

# 2. 一半模糊 + 过曝
orig = sl.read
def blurred(lv, bbox=None, chs=None):
    a = orig(lv, bbox, chs).copy()
    if bbox == s.bbox0 and a.shape[1] > 1000:
        h = a.shape[2] // 2
        for c in range(a.shape[0]):
            a[c, :, :h] = ndi.gaussian_filter(a[c, :, :h].astype(np.float32), 4).astype(a.dtype)
        a[-1][a[-1] > np.percentile(a[-1], 97)] = 255
    return a
sl.read = blurred
b, sh, sat = Q.image_metrics(p, sl, s, nuc); print("半边模糊:", b, "过曝:", sat)
assert b > 30 and max(sat.values()) > 2
sl.read = orig

# 3. 组织缺损
th = R.section_thumbnail
def torn(slide, sec, n):
    img, m, ds = th(slide, sec, n); m = m.copy()
    ys, xs = np.nonzero(m); cy, cx = int(ys.mean()), int(xs.mean())
    m[cy - 120:cy + 120, cx - 150:cx + 150] = False
    return img, m, ds
R.section_thumbnail = torn
_, g = Q.registration_metrics(p, im, sl, s, nuc); print("缺损 %:", g)
assert g > base["damage"] + 5
# 画排除区域后不再算缺损
ds = th(sl, s, nuc)[2]
_, m0, _ = th(sl, s, nuc); ys, xs = np.nonzero(m0); cy, cx = int(ys.mean()), int(xs.mean())
X0, Y0 = s.bbox0[0], s.bbox0[1]
im2 = copy.deepcopy(im); im2.exclusions = [[(X0 + (cx - 155) * ds, Y0 + (cy - 125) * ds), (X0 + (cx + 155) * ds, Y0 + (cy - 125) * ds),
                                            (X0 + (cx + 155) * ds, Y0 + (cy + 125) * ds), (X0 + (cx - 155) * ds, Y0 + (cy + 125) * ds)]]
_, g2 = Q.registration_metrics(p, im2, sl, s, nuc); print("画排除区域后缺损 %:", g2)
assert g2 < g - 5
R.section_thumbnail = th
print(Q.grade({"dice": d, "blur": b, "damage": g, "saturation": sat}))
df = Q.quality_table(p); print(df[["图片", "脑片", "质量", "质量分", "需要检查的问题"]]); print(Q.cached_levels(p))
print("OK")
