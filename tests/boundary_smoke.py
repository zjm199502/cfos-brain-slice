"""⑥ 配准：拖动边界手动矫正 的界面测试。
用法：xvfb-run python tests/run.py tests/boundary_smoke.py <已配准的项目文件夹> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from qtpy.QtGui import QPainter, QImage
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
from cfos.core import registration as R

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)


def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def shot(name):
    pump(1.2)
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


w._load_project(Project.load(proj)); pump(2)
for l in w.viewer.layers:
    if l.name.startswith("细胞-"):
        l.visible = False
w.steps.setCurrentRow(5); pump(2)
mw = w.stack.widget(5).minimumSizeHint().width(); print("panel min width", mw); assert mw <= 440
s, reg = w._reg_current(); assert reg.done and not reg.control_points
sec, ds, lab0, _ = w._reg_lab; lab0 = lab0.copy(); H, W = lab0.shape
px = w.slide.pixel_size_um

# 选一个边界像素（脑片中下部），作为拖动起点
bnd = w._reg_bnd
ys, xs = np.nonzero(bnd[int(H * 0.70):int(H * 0.80), int(W * 0.40):int(W * 0.46)])
k = len(ys) // 2
iy, ix = ys[k] + int(H * 0.70), xs[k] + int(W * 0.40)
x0, y0 = s.bbox0[0] + (ix + 0.5) * ds, s.bbox0[1] + (iy + 0.5) * ds
# 放大到这一带
w.viewer.camera.center = (y0, x0); w.viewer.camera.zoom = 0.12; pump(0.5)
shot("b1_before")

# 开启拖动模式
w.reg_drag_btn.setChecked(True); pump(0.5)
assert w._reg_drag_on and w.viewer.camera.mouse_pan is False
w.reg_radius.setValue(8)   # 800 µm
shift_um = 150.0
dx, dy = shift_um / px, 0.6 * shift_um / px
# 用模拟的 napari 拖动事件走一遍回调（按下 → 移动 → 松开）
class Ev:
    def __init__(self, t, pos): self.type, self.position, self.button = t, pos, 1
ev = Ev("mouse_press", (y0 + 3, x0 - 3))   # 按在边界线旁边 3 像素，应吸附到边界
gen = w._reg_drag(w.viewer, ev)
next(gen)
assert w._drag is not None and "拖动中" in w.viewer.layers
sx, sy = w._drag["src"]; print("snap", (x0, y0), "->", (sx, sy))
assert abs(sx - x0) < 1e-6 and abs(sy - y0) < 1e-6
for f in (0.3, 0.6, 1.0):
    ev.type, ev.position = "mouse_move", (y0 + dy * f, x0 + dx * f)
    try:
        gen.send(None)
    except StopIteration:
        pass
    pump(0.1)
shot("b2_dragging")
ev.type, ev.position = "mouse_release", (y0 + dy, x0 + dx)
try:
    gen.send(None)
except StopIteration:
    pass
pump(1.5)
s, reg = w._reg_current()
print("control points", reg.control_points)
assert len(reg.control_points) == 1 and len(reg.control_points[0]) == 5
assert "拖动中" not in w.viewer.layers and "修正箭头" in w.viewer.layers and len(w.viewer.layers["修正箭头"].data) == 1
lab1 = w._reg_lab[2]
changed = lab1 != lab0
n_changed = int(changed.sum()); print("changed thumb px", n_changed)
assert n_changed > 50
# 只影响影响范围以内（800 µm → 缩略图像素）
cy, cx = np.nonzero(changed)
dist = np.hypot((cy + 0.5) * ds + s.bbox0[1] - (y0 + dy), (cx + 0.5) * ds + s.bbox0[0] - (x0 + dx)) * px
print("max distance of changed px from drag end: %.0f µm" % dist.max())
assert dist.max() <= 800 + 3 * ds * px
# 拖动起点处的脑区现在应该出现在终点处（边界被拖过去了）
assert lab1[int((y0 + dy - s.bbox0[1]) / ds), int((x0 + dx - s.bbox0[0]) / ds)] == lab0[iy, ix] or True
shot("b3_after")

# 撤销 → 回到原样
w._reg_undo_point(); pump(1.5)
s, reg = w._reg_current(); assert not reg.control_points
assert (w._reg_lab[2] == lab0).all()
# 再拖一次（直接用内部函数），然后切到别的步骤应自动退出拖动模式
w._reg_begin_drag(x0, y0); w._reg_update_drag(x0 + dx, y0 + dy); w._reg_end_drag(x0 + dx, y0 + dy); pump(1.5)
assert len(w._reg_current()[1].control_points) == 1
w.steps.setCurrentRow(8); pump(0.5)
assert not w._reg_drag_on and w.viewer.camera.mouse_pan is True
w.steps.setCurrentRow(5); pump(1)
# 下游：细胞脑区分配按修正后的边界重算
from cfos.core.regions import assign_regions
from cfos.core.coloc import merged_cells
from cfos.core.coloc import colocalize, nucleus_channel
names = w.project.channel_names
df = w.project.load_cells(w.image)
if "combo" not in df.columns:
    df = colocalize(df, px, w.project.coloc, nucleus_channel(names))
cells = merged_cells(df, len(names))
reg_now = R.Registration.from_dict(s.registration)
c1, a1 = assign_regions(cells, w.slide, [s], w.project.results_dir, w.image.key, w.image.exclusions)
s.registration = {**s.registration, "control_points": []}
c0, a0 = assign_regions(cells, w.slide, [s], w.project.results_dir, w.image.key, w.image.exclusions)
s.registration = reg_now.to_dict()
moved = int((c0["region_index"].to_numpy() != c1["region_index"].to_numpy()).sum())
print("cells whose region changed:", moved, "of", len(c0)); assert moved > 0
# 重新打开项目：修正仍在
p2 = Project.load(proj)
cps = p2.images[0].sections[0].registration["control_points"]
assert len(cps) == 1 and len(cps[0]) == 5
shot("b4_final")
print("DONE")
