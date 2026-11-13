"""为使用说明书截图：用示例脑片在 v0.7 上走一遍完整流程。
用法：xvfb-run -s "-screen 0 1920x1080x24" python tests/run.py /home/user/shots/shoot.py <项目目录> <截图目录>"""
import sys, time, shutil, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, "/home/user/src/cfos_app")
import numpy as np
from napari.qt import get_qapp
from qtpy.QtGui import QPainter, QImage
from qtpy.QtWidgets import QMessageBox, QScrollArea
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
fresh = not (proj / "cfos_project.json").exists()
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
msgs = []
w.info = lambda t: (msgs.append(t), print("INFO:", t.replace("\n", " | "), flush=True))
w.error = lambda t: (msgs.append(t), print("ERROR:", t.replace("\n", " | "), flush=True))
w._open_folder = lambda p: print("open folder", p)


def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.3)
    pump(1.0)


def shot(name, widget=None):
    pump(1.2)
    if widget is not None:
        widget.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True); return
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


def panel(name):
    """整页右侧面板（含滚动看不到的部分）。"""
    shot(name, w.stack.currentWidget())


def msgbox(name, text, icon=QMessageBox.Information):
    m = QMessageBox(icon, "cFos 脑片分析", text, QMessageBox.Ok, w); m.show(); pump(0.8)
    shot(name, m); m.close()


def rect(x, y, rw, rh):
    return np.array([[y, x], [y, x + rw], [y + rh, x + rw], [y + rh, x]], float)


def hide_cells():
    for l in w.viewer.layers:
        if l.name.startswith("细胞-") or l.name.startswith("预览"):
            l.visible = False


demo = Path("/home/user/src/cfos_app/resources/demo/示例脑片.ome.tif")
w.teach_cb.setChecked(False); pump(0.5)
if fresh:
    shot("00_启动界面")
    proj.mkdir(parents=True, exist_ok=True)
    img = proj / "原图" / demo.name; img.parent.mkdir(exist_ok=True); shutil.copy(demo, img)
    p = Project(folder=proj); p.save(); w._load_project(p); pump(1)
    w.add_images_programmatic([img]); pump(2)
else:
    w._load_project(Project.load(proj)); pump(2)
w.steps.setCurrentRow(0); pump(1)
shot("01_项目与图片"); panel("01b_项目面板")

# ---- 自检
if "selftest" in sys.argv:
    w.run_selftest(); wait_task()
    if msgs:
        msgbox("01c_自检结果", msgs[-1])

# ---- ② 拆分
w.steps.setCurrentRow(1); pump(1)
if fresh:
    w.auto_sections(False); wait_task()
w.viewer.reset_view(); w._refresh_section_table(); pump(1)
shot("02_脑片拆分")

# ---- ③ 显示
w.steps.setCurrentRow(2); pump(1); shot("03_亮度对比度")

# ---- ④ 识别
w.steps.setCurrentRow(3); pump(1)
s = w.image.sections[0]; x, y, sw, sh = s.bbox0; px = w.slide.pixel_size_um
cx, cy = x + sw * 0.5, y + sh * 0.3
w.viewer.camera.center = (cy, cx); w.viewer.camera.zoom = 0.5; pump(1)
w.preview_detection(); wait_task()
shot("04_预览识别"); panel("04b_识别面板")
if fresh:
    t0 = time.time(); w.run_detection(False); wait_task(); print("detect s", time.time() - t0, flush=True)
w._draw_cells(); hide_cells()
for l in w.viewer.layers:
    if l.name.startswith("细胞-"):
        l.visible = True
w.viewer.camera.center = (cy, cx); w.viewer.camera.zoom = 0.5; pump(1)
shot("04c_识别结果")

# 准确率检查
red = w.project.channel_names.index("Red")
w.acc_ch.setCurrentIndex(red)
df = w.cells_for(w.image)
g = df[(df["channel"] == red) & (df["section"] == s.name)]
w.acc_size.setValue(150)
w.acc_new_box(); pump(0.5)
box = w._acc_box0()
inb = g[(g["x"] >= box[0]) & (g["x"] < box[0] + box[2]) & (g["y"] >= box[1]) & (g["y"] < box[1] + box[3])]
rng = np.random.default_rng(1)
keep = inb.iloc[2:]
man = keep[["x", "y"]].to_numpy() + rng.uniform(-1.5 / px, 1.5 / px, (len(keep), 2))
far = []
while len(far) < 3:
    q = np.array([box[0] + rng.uniform(0.1, 0.9) * box[2], box[1] + rng.uniform(0.1, 0.9) * box[3]])
    if len(inb) == 0 or np.min(np.hypot(*(inb[["x", "y"]].to_numpy() - q).T)) > 15 / px:
        far.append(q)
man = np.vstack([man, far])
w.viewer.camera.center = (box[1] + box[3] / 2, box[0] + box[2] / 2); w.viewer.camera.zoom = 1.6; pump(0.5)
w.viewer.layers["人工计数"].add(man[:, ::-1]); pump(0.5)
shot("04d_准确率_人工点数")
w.acc_compute(); pump(1)
shot("04e_准确率_结果")
if msgs:
    msgbox("04f_准确率_弹窗", msgs[-1])
for nm in ("准确率检查框", "人工计数", "准确率结果"):
    if nm in w.viewer.layers:
        w.viewer.layers.remove(nm)

# ---- ⑤ 共标
w.steps.setCurrentRow(4); pump(1)
w.recompute_coloc(); wait_task()
w.viewer.reset_view(); pump(1)
shot("05_共标统计"); panel("05b_共标面板")

# ---- ⑥ 配准
w.steps.setCurrentRow(5); pump(1)
if fresh or not s.registration.get("done", False):
    t0 = time.time(); w.auto_register(False); wait_task(); print("reg s", time.time() - t0, flush=True)
w.reg_sec_combo.setCurrentIndex(0); pump(2); hide_cells()
w.viewer.reset_view(); pump(1)
shot("06_配准结果"); panel("06b_配准面板")
# 拖动修正
s, reg = w._reg_current()
sec, ds, lab0, _ = w._reg_lab; H, W = lab0.shape
bnd = w._reg_bnd
ys, xs = np.nonzero(bnd[int(H * 0.70):int(H * 0.80), int(W * 0.40):int(W * 0.46)])
k = len(ys) // 2
iy, ix = ys[k] + int(H * 0.70), xs[k] + int(W * 0.40)
x0, y0 = s.bbox0[0] + (ix + 0.5) * ds, s.bbox0[1] + (iy + 0.5) * ds
w.viewer.camera.center = (y0, x0); w.viewer.camera.zoom = 0.12; pump(0.5)
shot("06c_拖动前")
w.reg_drag_btn.setChecked(True); pump(0.5)
w.reg_radius.setValue(8)
dx, dy = 150 / px, 90 / px


class Ev:
    def __init__(self, t, pos): self.type, self.position, self.button = t, pos, 1


ev = Ev("mouse_press", (y0, x0))
gen = w._reg_drag(w.viewer, ev); next(gen)
for f in (0.3, 0.6, 1.0):
    ev.type, ev.position = "mouse_move", (y0 + dy * f, x0 + dx * f)
    try:
        gen.send(None)
    except StopIteration:
        pass
    pump(0.1)
shot("06d_拖动中")
ev.type, ev.position = "mouse_release", (y0 + dy, x0 + dx)
try:
    gen.send(None)
except StopIteration:
    pass
pump(2)
shot("06e_拖动后")
w._reg_undo_point(); pump(1)
w.reg_drag_btn.setChecked(False); pump(0.5)

# ---- ⑦ 分组
w.steps.setCurrentRow(6); pump(1)
w._grp_guess(); pump(0.5)
w.project.set_sample(f"{w.image.key}/{s.name}", animal="M1", group="对照"); w.project.save()
w.steps.setCurrentRow(5); pump(0.3); w.steps.setCurrentRow(6); pump(1)
shot("07_分组与统计"); panel("07b_分组面板")

# ---- ⑧ 导出
w.steps.setCurrentRow(7); pump(1); shot("08_导出Excel")
from cfos.core.export import export_excel
print(export_excel(w.project, proj / "细胞统计结果.xlsx"), flush=True)

# ---- ⑨ 截图制图
w.steps.setCurrentRow(8); pump(1.5); hide_cells()
w.fig_draw_mode(); pump(0.5)
layer = w.viewer.layers["截图框"]
layer.add_rectangles([rect(x + sw * 0.30, y + sh * 0.40, 1500 / px, 1000 / px)]); pump(0.5)
w.fig_save_rois(); pump(0.5)
w.fig_list.item(0).setText("海马区"); pump(0.3)
w.fig_cells.setChecked(True); w.fig_regions.setChecked(True); w.fig_region_names.setChecked(True)
w.fig_inset.setChecked(True)
w.viewer.reset_view(); pump(1)
shot("09_截图制图"); panel("09b_截图制图面板")
w.fig_list.setCurrentRow(0); w.fig_preview(); pump(2)
if hasattr(w, "_fig_dialog"):
    shot("09c_预览", w._fig_dialog); w._fig_dialog.accept()
w.fig_export(False); wait_task()
w.fig_export_sections(False); wait_task()

# ---- 教学模式
w.teach_cb.setChecked(True); pump(1)
w.steps.setCurrentRow(0); pump(1); shot("10_教学模式_第1步")
w.steps.setCurrentRow(5); pump(1); shot("10b_教学模式_配准")
w.teach_cb.setChecked(False)
w.project.save()
print("DONE", flush=True)
app.quit()
