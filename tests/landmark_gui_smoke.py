"""⑥ 标志点界面测试：拖动添加、单击两下添加、拖动挪位置、删除、撤销、全部清除、切面滑块、Esc、旧版修正。
不需要真实图片和图谱：用合成的 DAPI 图和一个合成的 25 µm 图谱，脑片的配准直接写入（不跑 giRAff / elastix）。
用法：QT_QPA_PLATFORM=offscreen python tests/run.py tests/landmark_gui_smoke.py <输出目录>"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfos.core import atlas as A  # noqa: E402

# ---- 合成图谱：椭球形大脑，左右各一个 "DG" 团块
D, V, M = A.SHAPE_ASR
ap, dv, ml = np.ogrid[:D, :V, :M]
brain = ((ap - 264) / 250.) ** 2 + ((dv - 160) / 140.) ** 2 + ((ml - 228) / 200.) ** 2 < 1
annotation = brain.astype(np.uint16)
for c in (150, 306):
    annotation[(((dv - 110) / 22.) ** 2 + ((ml - c) / 34.) ** 2 < 1) & brain] = 2
hemispheres = np.where(brain, np.where(ml < 228, 1, 2), 0).astype(np.uint8) * np.ones((D, 1, 1), np.uint8)
reference = (annotation * 90).astype(np.uint8)
structures = pd.DataFrame({"id": [997, 1080], "acronym": ["root", "DG"], "name": ["root", "Dentate gyrus"],
                           "name_zh": ["脑", "齿状回"], "index": [1, 2]})
FAKE = A.Atlas(annotation, reference, hemispheres, structures)
A._load_atlas = lambda name: FAKE
A.load_atlas = lambda name=None: FAKE

from napari.qt import get_qapp  # noqa: E402
from cfos.core import registration as R  # noqa: E402
from cfos.core.project import Project  # noqa: E402
from cfos.gui.i18n import install_chinese  # noqa: E402
from cfos.gui.main_window import MainWindow  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
proj = out / "proj"; shutil.rmtree(proj, ignore_errors=True)

# ---- 合成脑片：4 µm/像素，组织椭圆 + 两条亮带（相对图谱向下偏 40 µm）
H0, W0 = 640, 912
yy, xx = np.mgrid[:H0, :W0]
tissue = ((yy - 320) / 280.) ** 2 + ((xx - 456) / 400.) ** 2 < 1
dapi = np.where(tissue, 900, 30).astype(np.float32)
for c in (300, 612):
    dapi[(((yy - 230) / 44.) ** 2 + ((xx - c) / 68.) ** 2 < 1)] = 3000
rng = np.random.default_rng(0)
dapi = (dapi + rng.normal(0, 40, dapi.shape)).clip(0, 4095).astype(np.uint16)
tif = out / "synthetic.tif"
tifffile.imwrite(tif, np.stack([dapi, dapi // 2]), photometric="minisblack", resolution=(2500, 2500),
                 resolutionunit="CENTIMETER", metadata={"axes": "CYX"})

app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1500, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True)
w.error = lambda t: (_ for _ in ()).throw(AssertionError(t))


def pump(sec=.5):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents(); time.sleep(.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(.2)
    pump(.5)


class Ev:
    def __init__(self, t, pos): self.type, self.position, self.button = t, pos, 1


def gesture(points):
    """按下 points[0]，依次移动，在最后一点松开（画布坐标 (x, y)）。"""
    ev = Ev("mouse_press", points[0][::-1])
    gen = w._reg_lm_mouse(w.viewer, ev)
    if gen is None:
        return
    next(gen)
    for p in points[1:]:
        ev.type, ev.position = "mouse_move", p[::-1]
        try:
            gen.send(None)
        except StopIteration:
            return
    ev.type = "mouse_release"
    try:
        gen.send(None)
    except StopIteration:
        pass
    pump(.3)


p = Project(folder=proj); p.save(); w._load_project(p)
w.add_images_programmatic([str(tif)]); pump(1)
w.steps.setCurrentRow(1); w.auto_sections(False); wait_task()
im = p.images[0]
assert im.sections, "没有拆出脑片"
s = im.sections[0]
for other in im.sections[1:]:
    other.enabled = False
# 配准直接写入：冠状面 AP 264，图像宽度对应全部 ML，高度对应全部 DV；自动非线性结果为恒等。
ds_, (H, W), _ = __import__("cfos.core.regions", fromlist=["_thumb_geometry"])._thumb_geometry(w.slide, s)
reg = R.Registration(plane=[0., 264., 319., 456., 0., 0., 0., 0., -320.], status="已配准", method="giRAff",
                     bbox0=list(s.bbox0))
reg.warp_file = reg.base_warp_file = R.save_warp(p.results_dir, np.zeros((2, H, W), np.float32))
reg.nonlinear = True
s.registration = reg.to_dict(); p.save()
w.steps.setCurrentRow(5); pump(1)
w._sync_registration_panel(); pump(1)
s_, reg = w._reg_current()
assert s_ is s and reg.done
assert not w.reg_lm_list.isVisible() and not w.reg_lm_fit.isEnabled()
base = reg.base_warp_file


def data_xy(fx, fy):
    """脑片内的相对位置 → 画布坐标。"""
    return s.bbox0[0] + fx * s.bbox0[2], s.bbox0[1] + fy * s.bbox0[3]


def current():
    return R.Registration.from_dict(s.registration)


# 1. 拖动添加：把左侧 "DG" 的下缘拖到亮带下缘
w.reg_lm_btn.setChecked(True); pump(.3)
assert w._lm_on and not w.viewer.camera.mouse_pan
a, b = data_xy(.33, .44), data_xy(.33, .50)
gesture([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), b])
r = current()
assert len(r.landmarks) == 1 and r.landmark_fit == "preview" and r.warp_file != base, r.landmarks
assert w.reg_lm_list.count() == 1 and w.reg_lm_fit.isEnabled()
print("list:", w.reg_lm_list.item(0).text(), flush=True)
assert "自动结果偏了" in w.reg_lm_list.item(0).text()
assert "标志点" in w.viewer.layers and len(w.viewer.layers["标志点"].data) == 1
ds, H, W, warp = w._reg_geometry(s)
assert max(R.landmark_offsets_um(r, np.stack(warp), W, H, w._reg_pixel_um(s))) < 4

# 2. 单击两下添加（右侧）
c, d = data_xy(.67, .44), data_xy(.67, .51)
gesture([c]); assert w._lm_pending is not None
gesture([d]); assert w._lm_pending is None
assert len(current().landmarks) == 2 and w.reg_lm_list.count() == 2

w.grab().save(str(out / "landmark_two.png"))

# 3. 拖动已有的标志点挪位置
t = w._reg_lm_targets(s, current())[0]
gesture([t, (t[0], t[1] + 10)])
assert abs(w._reg_lm_targets(s, current())[0][1] - (t[1] + 10)) < 2

# 4. 方向矛盾的标志点被拒绝，记录不变
before = current().landmarks
e = data_xy(.33, .56)   # 在标志点 1 下方的结构，却拖到它上方：顺序颠倒，必然折叠
gesture([e, data_xy(.33, .40)])
assert current().landmarks == before

# 5. 删除所选 → 撤销 → 全部清除
w.reg_lm_list.setCurrentRow(0); pump(.2)
w._reg_lm_delete(); pump(.2)
assert len(current().landmarks) == 1
w._reg_lm_undo_last(); pump(.2)
assert current().landmarks == before
w._reg_lm_clear(); pump(.2)
r = current()
assert not r.landmarks and r.warp_file == base and r.landmark_fit == ""
assert not w.reg_lm_list.isVisible() and "标志点" not in w.viewer.layers
w._reg_lm_undo_last(); pump(.2)
assert current().landmarks == before

# 6. 切面滑块：标志点保留并按新切面重新插值
slider = w.reg_sliders["shift_x"][0]
slider.setValue(slider.value() + 10); pump(1.2)
r = current()
assert r.landmarks == before and r.shift_x != 0
ds, H, W, warp = w._reg_geometry(s)
assert max(R.landmark_offsets_um(r, np.stack(warp), W, H, w._reg_pixel_um(s))) < 4

# 7. Esc 退出标志点模式
w.canvas_tools.escape_tools(); pump(.2)
assert not w._lm_on and w.viewer.camera.mouse_pan

# 8. 恢复自动结果（切面动过 → 会重新贴合；这里没有 elastix，只检查标志点已清空）
w.run_task = lambda *a, **k: print("refit requested", k.get("label"), flush=True)
w._reg_reset(); pump(.5)
r = current()
assert not r.landmarks and r.shift_x == 0 and r.warp_file == base

# 9. 旧版修正：提示并禁止添加，移除后可以添加
r.control_points = [[.3, .3, .32, .3, .05, 2]]
s.registration = r.to_dict(); w._reg_section_changed(w.reg_sec_combo.currentIndex()); pump(.5)
assert w.reg_legacy.isVisible()
w.reg_lm_btn.setChecked(True); pump(.2)
assert not w._lm_on
w._reg_remove_legacy(); pump(.3)
assert not current().control_points and not w.reg_legacy.isVisible()
w.reg_lm_btn.setChecked(True); pump(.2)
assert w._lm_on

w.grab().save(str(out / "landmark_panel.png"))
print("landmark GUI smoke OK", flush=True)
w.close(); app.quit()
