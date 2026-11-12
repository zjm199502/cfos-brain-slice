"""标注和微调界面测试（少量训练轮数）。用法：xvfb-run python tests/finetune_smoke.py <项目> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
import cfos.core.finetune as F
proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)
from qtpy.QtWidgets import QMessageBox
QMessageBox.question = staticmethod(lambda *a, **k: (print("QUESTION:", a[2]), QMessageBox.Yes)[1])
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
w.ft_ch.setCurrentText(w.project.channel_names[1]); pump(0.5)
s = w.image.sections[0]; x, y, ww, hh = s.bbox0
w.viewer.camera.center = (y + hh * 0.35, x + ww * 0.35); w.viewer.camera.zoom = 1.0; pump(1)
for l in w.viewer.layers:
    if l.name.startswith("细胞-"): l.visible = False
w.ft_new_patch(); wait_task()
lay = w.viewer.layers["标注块"]; print("labels", int(lay.data.max()))
w.layer_tools.refresh(); pump(0.5)
shot("f1_patch")
# 模拟修改：擦掉一个细胞
d = np.asarray(lay.data); d[d == 1] = 0; lay.data = d
w.ft_save_patch(); pump(1); print(w.ft_info.text())
F_orig = F.finetune
F.finetune = lambda *a, **k: F_orig(*a, **{**k, "epochs": 2, "steps_per_epoch": 5})
import cfos.gui.main_window as MW
w.ft_train(); wait_task()
print(w.ft_info.text()); print("model now:", w.project.detection.channels[1].model)
w._sync_detection_panel(); pump(1); shot("f2_after_train")
print("DONE")
