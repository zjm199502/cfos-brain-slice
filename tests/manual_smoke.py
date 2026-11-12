"""手动校正细胞测试。用法：xvfb-run python tests/manual_smoke.py <项目> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)
def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
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
w._load_project(Project.load(proj)); pump(2)
w.steps.setCurrentRow(3); pump(1)
df = w.cells_for(w.image); before = int((df.channel == 1).sum()); print("Red before", before)
w.man_ch.setCurrentText(w.project.channel_names[1]); w._man_mode("select"); pump(0.5)
L = w._man_layer()
# 放大到一块区域
s = w.image.sections[0]; x, y, ww, hh = s.bbox0
cx, cy = x + ww * 0.3, y + hh * 0.3
w.viewer.camera.center = (cy, cx); w.viewer.camera.zoom = 1.2; pump(1)
d = np.linalg.norm(L.data - [cy, cx], axis=1); idx = set(np.argsort(d)[:5].tolist())
L.selected_data = idx; w._man_delete()
w._man_mode("add"); L.add(np.array([[cy + 30, cx + 30], [cy - 40, cx + 10], [cy + 5, cx - 50]]))
print("cid tail", L.features["cid"].to_numpy()[-4:])
shot("m1_editing")
w.save_manual(); wait_task()
df = w.cells_for(w.image); print("Red after", int((df.channel == 1).sum()), "manual rows", int(df.get("manual", 0).sum()))
print(w.man_info.text())
# 删掉一个刚补加的
L = w._man_layer(); cid = L.features["cid"].to_numpy()
man_idx = [i for i, c in enumerate(cid) if df.loc[c, "manual"] == 1][:1]
L.selected_data = set(man_idx); w._man_delete(); w.save_manual(); wait_task()
print(w.man_info.text())
from cfos.core.export import build_tables
t = build_tables(w.project)["按脑片汇总"]; print(t.filter(like="人工").T)
shot("m2_saved")
print("DONE")
