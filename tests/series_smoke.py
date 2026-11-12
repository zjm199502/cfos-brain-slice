"""连续切片联合校正界面测试。用法：xvfb-run python tests/series_smoke.py <项目> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
w._load_project(Project.load(proj)); pump(2)
w.steps.setCurrentRow(5); pump(2)
w.series_dialog(); pump(1)
d = w._series_dlg
from qtpy.QtWidgets import QDoubleSpinBox, QPushButton
d.findChild(QDoubleSpinBox).setValue(600)
[b for b in d.findChildren(QPushButton) if b.text() == "应用联合校正"][0].click()
pump(1); d.grab().save(str(out / "s1_series_dialog.png")); wait_task()
from cfos.core import registration as R
for s in w.project.images[0].sections:
    r = R.Registration.from_dict(s.registration); print(s.name, r.status, round(R.adjusted_plane(r).ap_mm(), 2), r.nonlinear)
[b for b in d.findChildren(QPushButton) if b.text() == "撤销联合校正"][0].click(); wait_task()
for s in w.project.images[0].sections:
    r = R.Registration.from_dict(s.registration); print("undo", s.name, r.status, round(R.adjusted_plane(r).ap_mm(), 2))
print("DONE")
