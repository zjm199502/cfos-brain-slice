"""v0.8.5 配准改进的核心测试（不需要界面）。
用法：python tests/v085_reg.py <含 2 张以上脑片的图片> [NeuN 通道序号]
检查：自动校正前后位置、网格 40 + 逐级放宽、需要检查的提示、同一动物统一角度、旧项目兼容、耗时。
给了 NeuN 通道（不参与配准）时，再用“灰质 / 白质和脑室 的 NeuN 亮度比”对比新旧流程。"""
import sys, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cfos.core import registration as R, series, autoreg
from cfos.core.atlas import load_atlas, sample_plane, Plane
from cfos.core.project import Project, ImageEntry
from cfos.core.sections import find_sections
from cfos.core.slide_io import open_slide

path = sys.argv[1]
neun = int(sys.argv[2]) if len(sys.argv) > 2 else None
td = Path(tempfile.mkdtemp())
sl = open_slide(path)
p = Project(folder=td / "proj"); p.ensure_channels(sl.channels) if hasattr(sl, "channels") else None
im = ImageEntry(path=str(path)); im.sections = find_sections(sl); p.images.append(im)
p.results_dir.mkdir(parents=True, exist_ok=True)
nuc = 0

# ---- 旧流程（v0.8.4）：DeepSlice + 网格 20，折叠就退回只用轮廓
old, thumbs = R.auto_register(sl, im.sections, nuc, str(td), return_thumbs=True)
R.NL_GRIDS = (20.0,)
old_w = {}
for s in im.sections:
    r = old[s.name]; img, mask = thumbs[s.name]
    old_w[s.name] = R.nonlinear_fit(r, R.orient(mask, r.orientation), R.orient(img, r.orientation)); r.nonlinear = True
R.NL_GRIDS = (40.0, 60.0)

# ---- 新流程
t0 = time.time()
new = autoreg.run(p, [im], nuc)[im.path]
dt = time.time() - t0
print(f"新流程 {len(new)} 张脑片共 {dt:.1f} 秒（每张 {dt / len(new):.1f} 秒）")
assert set(new) == {s.name for s in im.sections}

at = load_atlas(); st = at.structures
cls = np.zeros(at.n_regions + 1, np.uint8)
for idx, pa in zip(st["index"], st["path"].astype(str)):
    q = pa.split("/"); cls[idx] = 2 if ("1009" in q or "73" in q) else (1 if "8" in q else 0)
lv = sl.level_for_pixel_size(R.THUMB_UM)


def ratio(reg, warp, s):
    img, mask = thumbs[s.name]; H, W = mask.shape
    lab, _ = R.label_maps(reg, W, H, warp)
    red = R.orient(sl.read(lv, s.bbox0)[neun].astype(np.float32), reg.orientation)
    c = cls[lab]; m = R.orient(mask, reg.orientation)
    return float(red[(c == 1) & m].mean() / red[(c == 2) & m].mean())


for s in im.sections:
    ro, rn = old[s.name], new[s.name]
    wn = np.load(R.warp_path(p.results_dir, im.key, s.name))
    assert rn.done and rn.nonlinear and R.min_jacobian(wn) > 0.05
    line = (f"{s.name}: Bregma {R.adjusted_plane(ro).ap_mm():+.2f} → {R.adjusted_plane(rn).ap_mm():+.2f} mm；"
            f"方法 {ro.nl_method} → {rn.nl_method}；{rn.status}；提示 {rn.warnings}")
    if neun is not None:
        a, b = ratio(ro, old_w[s.name], s), ratio(rn, (wn[0], wn[1]), s)
        line += f"；NeuN 灰质/白质 {a:.2f} → {b:.2f}"
        assert b > a, "新流程的独立检验指标应不低于旧流程"
    print(line)
    assert rn.nl_method == R.TEXTURE
    assert abs(rn.ap_auto * 25) <= R.AP_RANGE_UM

# ---- 滑块只显示手动部分；恢复自动结果后仍保留自动校正
r = R.Registration.from_dict(new[im.sections[0].name].to_dict())
assert r.ap_shift == 0.0
ap0 = R.adjusted_plane(r).ap_mm(); r.ap_shift = 4
assert abs(R.adjusted_plane(r).ap_mm() - ap0 - 0.1) < 0.02   # 切面有倾斜时略小于 0.1

# ---- 旧项目（没有 ap_auto / warnings 字段）照常读取，位置不变
d = old[im.sections[0].name].to_dict(); d.pop("ap_auto"); d.pop("warnings")
r0 = R.Registration.from_dict(d)
assert r0.ap_auto == 0.0 and r0.warnings == []
assert np.allclose(R.adjusted_plane(r0).to_list(), Plane.from_list(d["plane"]).to_list())

# ---- 提示：相邻脑片位置跳变
ra = R.Registration(plane=list(old[im.sections[0].name].plane)); rb = R.Registration(plane=list(ra.plane), ap_shift=30)
assert set(autoreg.neighbor_warnings({"a": ra, "b": rb})) == {"a", "b"}
rb.ap_shift = 4
assert autoreg.neighbor_warnings({"a": ra, "b": rb}) == {}

# ---- 统一角度：3 张（其中 1 张是以前配准好的，不改）
base = Plane.from_list(old[im.sections[0].name].plane)
regs = []
for deg in (-6.0, 0.0, 5.0):
    Rm = R._rot(base.v, deg); u, v = Rm @ base.u, Rm @ base.v; c = base.center()
    regs.append(R.Registration(plane=Plane(c - 0.5 * u - 0.5 * v, u, v).to_list()))
ctx_plane = list(regs[2].plane)
assert not series.unify_new(regs[:2], [])            # 不足 3 张不动
assert series.unify_new(regs[:2], [regs[2]])
ang = [series.angles(r)[0] for r in regs[:2]]
assert abs(ang[0] - ang[1]) < 0.01 and regs[2].plane == ctx_plane and regs[0].plane_raw
assert series.undo(regs[0]) and abs(series.angles(regs[0])[0] - series.angles(regs[1])[0]) > 5

# ---- 等间距联合校正把自动校正的平移算在内
ra = R.Registration(plane=base.to_list(), ap_auto=8.0); rb = R.Registration(plane=base.to_list()); rc = R.Registration(plane=base.to_list(), ap_auto=-3.0)
series.harmonize([ra, rb, rc], spacing_um=100.0, fix_angle=False)
aps = sorted(R.adjusted_plane(x).ap_mm() for x in (ra, rb, rc))
assert abs((aps[1] - aps[0]) - 0.1) < 0.02 and abs((aps[2] - aps[1]) - 0.1) < 0.02, aps
print("v0.8.5 配准核心测试通过")
