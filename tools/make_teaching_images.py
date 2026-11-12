"""用示例项目（已跑完整流程）生成教学模式的参考图和参考答案，写到 resources/teaching。
用法：python tools/make_teaching_images.py <示例项目文件夹> [统计结果文件夹(含 脑区热图.png)]"""
import copy, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
from cfos.core.project import Project
from cfos.core.slide_io import open_slide
from cfos.core import detection as det, qc, registration as R
from cfos.core.coloc import merged_cells, nucleus_channel
from cfos.core.resources import resources_dir

proj = Path(sys.argv[1]); out = resources_dir() / "teaching"; out.mkdir(parents=True, exist_ok=True)
p = Project.load(proj); im = p.images[0]; s = [x for x in im.sections if x.enabled][0]
sl = open_slide(im.path); names = p.channel_names; nuc = nucleus_channel(names)
mk = next(c for c in range(len(names)) if c != nuc)
df = p.load_cells(im)

def save(img, name, width=None):
    if width and img.width > width:
        img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    img.save(out / name, optimize=True); print(name, img.size)

# 1. 脑片拆分
lv = sl.level_for_pixel_size(16); ds = sl.downsample(lv)
a = sl.read(lv).astype(np.float32)
rgb = np.zeros(a.shape[1:] + (3,), np.float32)
for c, col in ((nuc, (0.3, 0.5, 1.0)), (mk, (1.0, 0.25, 0.25))):
    rgb += np.clip(a[c] / np.percentile(a[c], 99.5), 0, 1)[..., None] * np.array(col)
img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(img)
x, y, w, h = [v / ds for v in s.bbox0]; d.rectangle([x, y, x + w, y + h], outline=(255, 220, 0), width=3)
d.text((x + 4, y + 4), s.name, fill=(255, 220, 0))
save(img, "sections_good.png", 700)

# 选一块细胞多的区域（标记物细胞密度最高的 400 µm 方块）
px = sl.pixel_size_um
m = df[df["channel"] == mk]
H2, edges_x, edges_y = np.histogram2d(m["x"], m["y"], bins=[np.arange(s.bbox0[0], s.bbox0[0] + s.bbox0[2], 400 / px),
                                                           np.arange(s.bbox0[1], s.bbox0[1] + s.bbox0[3], 400 / px)])
i, j = np.unravel_index(np.argmax(H2), H2.shape)
cx, cy = (edges_x[i] + edges_x[i + 1]) / 2, (edges_y[j] + edges_y[j + 1]) / 2

def crop(size_um, level=0):
    ds_ = sl.downsample(level); n = int(size_um / px)
    box = (int(cx - n / 2), int(cy - n / 2), n, n)
    return sl.read(level, box).astype(np.float32), box

# 2. 亮度
a, box = crop(300)
def comp(a, hi_scale):
    rgb = np.zeros(a.shape[1:] + (3,), np.float32)
    for c, col in ((nuc, (0.3, 0.5, 1.0)), (mk, (1.0, 0.2, 0.2))):
        hi = np.percentile(a[c], 99.7) * hi_scale
        rgb += np.clip(a[c] / max(hi, 1), 0, 1)[..., None] * np.array(col)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
save(comp(a, 1.0), "contrast_good.png", 520)
save(comp(a, 0.12), "contrast_bright.png", 520)
save(comp(a, 6.0), "contrast_dim.png", 520)

# 3. 识别：不同阈值
cp = p.detection.channels[mk]
lvd = sl.level_for_pixel_size(p.detection.pixel_um); dsd = sl.downsample(lvd); pix = px * dsd
n = int(200 / pix) // 16 * 16
bx = (int(cx - n * dsd / 2), int(cy - n * dsd / 2), int(n * dsd), int(n * dsd))
raw = sl.read(lvd, bx, [mk])[0][:n, :n]
lo, hi, _ = det._norm_percentiles(sl, s, mk, cp.norm_low, cp.norm_high)
x_in = det.prepare_input(raw, cp, lo, hi, pix)
model = det.get_model(cp.model)
show = np.clip(raw.astype(np.float32) / np.percentile(raw, 99.7), 0, 1)
for name, th in (("detect_good.png", cp.prob_thresh), ("detect_low.png", 0.08), ("detect_high.png", 0.85)):
    lab, _ = model.predict_instances(x_in, prob_thresh=th, verbose=False)
    # 太低阈值时再把面积/强度筛选也去掉，模拟“什么都收”
    rgb = np.stack([show, show * 0.25, show * 0.25], -1)
    rgb[find_boundaries(lab, mode="outer")] = (1, 0.9, 0)
    save(Image.fromarray((rgb * 255).astype(np.uint8)).resize((520, 520), Image.NEAREST), name)
    print("  thresh", th, "cells", lab.max())

# 4. 共标
a, box = crop(200)
img = comp(a, 1.0); d = ImageDraw.Draw(img)
cells = merged_cells(df[df["section"] == s.name], len(names))
inb = cells[(cells["x"] > box[0]) & (cells["x"] < box[0] + box[2]) & (cells["y"] > box[1]) & (cells["y"] < box[1] + box[3])]
r = 5 / px
for _, c in inb.iterrows():
    if not c.get(f"pos_ch{mk}", False):
        continue
    col = (255, 230, 0) if c.get(f"pos_ch{nuc}", False) else (255, 0, 255)
    X, Y = c["x"] - box[0], c["y"] - box[1]
    d.ellipse([X - r, Y - r, X + r, Y + r], outline=col, width=3)
save(img, "coloc_example.png", 520)

# 5. 配准：正确 / 偏移
good = qc.section_qc_image(p, im, sl, s, df)
save(qc.section_qc_image(p, im, sl, s, None), "reg_good.png", 700)  # 不画细胞，边界更清楚
s2 = copy.deepcopy(s); rg = R.Registration.from_dict(s2.registration)
rg.ap_shift += 24; rg.shift_x += 0.07; rg.shift_y -= 0.04; rg.rotation += 8; rg.nonlinear = False; s2.registration = rg.to_dict()
save(qc.section_qc_image(p, im, sl, s2, None), "reg_bad.png", 700)

# 6. 质控图局部
W, H = good.size
save(good.crop((W // 3, H // 3, W // 3 + W // 6, H // 3 + H // 6)), "qc_example.png", 520)

# 7. 热图
if len(sys.argv) > 2 and (Path(sys.argv[2]) / "脑区热图.png").exists():
    hm = Image.open(Path(sys.argv[2]) / "脑区热图.png"); save(hm.convert("RGB"), "heatmap_example.png", 900)

# 参考答案
from cfos.core.export import build_tables  # 与 Excel“按脑片汇总”中学生看到的数字一致
row = build_tables(p)["按脑片汇总"].iloc[0]
counts = {f"{nm}+": int(row[f"{nm}+"]) for nm in names if f"{nm}+" in row}
json.dump({"counts": counts, "image": Path(im.path).name, "section": s.name},
          open(out / "示例答案.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(counts)
