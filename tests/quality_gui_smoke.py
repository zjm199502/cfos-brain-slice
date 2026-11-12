"""质量检查界面：② 面板的“质量”列和结果窗口，以及导出 Excel 的标色。用法：xvfb-run python tests/quality_gui_smoke.py <项目> <截图目录>"""
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
w.steps.setCurrentRow(1); pump(1.5)
print("min width", w.stack.widget(1).minimumSizeHint().width())
w.run_quality(); wait_task()
print("表格质量列:", [w.sec_table.item(i, 3).text() for i in range(w.sec_table.rowCount())])
w._q_dialog.grab().save(str(out / "quality_dialog.png"))
w.grab().save(str(out / "p2_quality.png"))
from cfos.core.export import export_excel
import pandas as pd
x = export_excel(w.project, out / "q.xlsx", masks=False, qc=False)
t = pd.read_excel(x, sheet_name=None)
print(list(t), t["按脑片汇总"].columns.tolist())
print("DONE")
