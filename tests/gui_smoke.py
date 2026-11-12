"""无人值守地走一遍界面流程并截图。用法：xvfb-run python tests/gui_smoke.py <图片> <输出目录> [已有细胞结果.pkl]"""
import shutil, sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from napari.qt import get_qapp
from qtpy.QtWidgets import QApplication
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project

img, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
proj = out / "proj"; shutil.rmtree(proj, ignore_errors=True)
app = get_qapp()
w = MainWindow(); w.resize(1600, 950); w.show()

def pump(sec=1.0):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents(); time.sleep(0.02)

def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)

def shot(name):
    pump(1.5)
    from qtpy.QtGui import QPainter, QImage
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)

p = Project(folder=proj); p.save(); w._load_project(p)
w.add_images_programmatic([img]); pump(2)
shot("1_project")
w.steps.setCurrentRow(1); w.auto_sections(False); wait_task()
w.viewer.reset_view(); pump(1)
w._refresh_section_table(); shot("2_sections")
w.steps.setCurrentRow(2); pump(); shot("3_display")
FULL = len(sys.argv) > 3 and sys.argv[3] == "full"
if len(sys.argv) > 3 and not FULL:
    df = pd.read_pickle(sys.argv[3]); p.save_cells(p.images[0], df); w._cells_cache.clear()
w.steps.setCurrentRow(3); pump()
# 放大到皮层一小块区域做预览
w.viewer.camera.center = (19200 + 1500, 9600 + 1500); w.viewer.camera.zoom = 0.25
pump(2); w.preview_detection(); wait_task(); shot("4_preview")
if FULL:
    t0 = time.time(); w.run_detection(False); wait_task(); print("full detection s", round(time.time() - t0), flush=True)
    print(w.det_summary.text(), flush=True)
w._draw_cells(); w.viewer.camera.zoom = 0.06; pump(2); shot("4b_cells")
w.steps.setCurrentRow(4); w.recompute_coloc(); wait_task(); shot("5_coloc")
w.steps.setCurrentRow(5); pump(1)
t0 = time.time(); w.auto_register(False); wait_task(); print("register s", round(time.time() - t0), flush=True)
w.reg_sec_combo.setCurrentIndex(0); pump(2)
for nm in [l.name for l in w.viewer.layers if l.name.startswith("细胞-") or l.name.startswith("预览")]:
    w.viewer.layers[nm].visible = False
shot("6_registration")
# 模拟手动调整：加一个修正点并拖动
s0 = p.images[0].sections[0]
layer = w.viewer.layers["修正点"]
bx, by, bw, bh = s0.bbox0
layer.add([by + bh * 0.3, bx + bw * 0.5]); pump(0.5)
d = layer.data.copy(); d[-1] = [by + bh * 0.3 + 400, bx + bw * 0.5]; layer.data = d; pump(2)
print("control points", s0.registration.get("control_points"), flush=True)
w._reg_redraw(); shot("6b_registration_point")
w.viewer.camera.center = (by + bh * 0.3, bx + bw * 0.5); w.viewer.camera.zoom = 0.12; pump(2); shot("6c_zoom")
from cfos.core.export import export_excel
p.save()
print(export_excel(p, out / "结果.xlsx"), flush=True)
