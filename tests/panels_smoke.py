"""各步骤面板截图 + 宽度检查 + 强度分级界面。用法：xvfb-run python tests/panels_smoke.py <项目> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
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
for l in w.viewer.layers:
    if l.name.startswith("细胞-"): l.visible = False
for i, nm in enumerate(STEPS):
    w.steps.setCurrentRow(i); pump(1.5)
    mw = w.stack.widget(i).minimumSizeHint().width()
    print(nm, "min width", mw, "超宽!" if mw > 440 else "")
w.steps.setCurrentRow(4); pump(1)
w.gr_ch.setCurrentIndex(0); w.grading_suggest(); pump(2)
print("grading", w.project.grading)
w.stack.parentWidget().parentWidget().verticalScrollBar().setValue(10000); pump(1)
w.grab().save(str(out / "p5_grading.png"))
print("DONE")
