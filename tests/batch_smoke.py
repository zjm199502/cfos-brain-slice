"""无人值守测试：一键批量处理 + 预处理预览。用法：xvfb-run python tests/batch_smoke.py <图片文件夹> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from cfos.gui.main_window import MainWindow
from cfos.core.detection import ChannelParams

folder, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp()
from cfos.gui.i18n import install_chinese; print('translators', install_chinese(app))
w = MainWindow(); w.resize(1600, 950); w.show()
msgs = []
w.info = lambda t: (msgs.append(t), print("INFO:", t, flush=True))
w.error = lambda t: (msgs.append(t), print("ERROR:", t, flush=True))

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

t0 = time.time()
w.run_batch(folder, True, True, False); wait_task()
print("batch s", round(time.time() - t0), flush=True)
w.steps.setCurrentRow(0); shot("b1_after_batch")
# 预处理预览
p = w.project
for c, cp in p.detection.channels.items():
    cp.deconv = True; cp.bg_radius_um = 20
w.steps.setCurrentRow(3); pump(1)
w.viewer.camera.center = (19200 + 1000, 9600 + 1000); w.viewer.camera.zoom = 0.6; pump(2)
w.preview_enhance(); wait_task(); shot("b2_preview_enhance")
print("xlsx", list(p.folder.glob("*.xlsx")), flush=True)
w.steps.setCurrentRow(1); pump(1); w.viewer.reset_view(); w.viewer.layers.selection.active = w.viewer.layers["脑片"]; pump(1); shot("b3_sections_tools")
from qtpy.QtWidgets import QMessageBox, QFileDialog
from qtpy.QtCore import QTimer
mb = QMessageBox(QMessageBox.Question, "测试", "对话框按钮是否为中文？", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, w)
mb.show(); pump(1); mb.grab().save(str(out / "b4_msgbox.png")); mb.close()
print("parts", sorted(str(p.relative_to(w.project.folder)) for p in (w.project.folder / "results" / "parts").rglob("*.pkl")), flush=True); print("DONE", flush=True)
