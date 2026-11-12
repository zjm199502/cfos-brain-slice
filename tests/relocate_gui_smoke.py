"""打开原图找不到的项目：提示、表格标红、一键找回。用法：xvfb-run python tests/relocate_gui_smoke.py <项目(原图已缺失)> <搜索文件夹> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from qtpy.QtWidgets import QMessageBox, QFileDialog
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
proj, search, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
asked = []
QMessageBox.question = staticmethod(lambda *a, **k: (asked.append(a[2]), QMessageBox.Yes)[1])
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: search)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)
def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)
w._load_project(Project.load(proj)); pump(0.1)
print("加载后状态:", w.images_table.item(0, 2).text())
w.grab().save(str(out / "missing.png"))
pump(1.5)
while w.task is not None and w.task.isRunning():
    pump(0.2)
pump(2)
print("提示:", asked[0][:60] if asked else None)
print("找回后状态:", w.images_table.item(0, 2).text(), "slide:", w.slide is not None, "layers:", len(w.viewer.layers))
print("min width", w.stack.widget(0).minimumSizeHint().width())
w.grab().save(str(out / "relocated.png"))
print("DONE")
