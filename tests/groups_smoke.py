"""分组与统计界面测试。用法：xvfb-run python tests/groups_smoke.py <项目> <假脑区表.pkl> <假分组.json> <截图目录>"""
import sys, time, json, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project

proj, fake, samples, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True)
w.error = lambda t: print("ERROR:", t, flush=True)

def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)

def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)

def shot(name):
    pump(1.0); w.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True)

w._load_project(Project.load(proj)); pump(2)
w.steps.setCurrentRow(6); pump(1)
print("rows", w.grp_table.rowCount())
w.grp_table.item(0, 3).setText("对照"); w.grp_table.item(1, 3).setText("电击"); pump(0.5)
w._grp_guess(); pump(0.5)
print("samples", w.project.samples)
from qtpy.QtWidgets import QFileDialog
f = str(out / "分组表.xlsx")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (f, ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (f, ""))
w._grp_export()
df = pd.read_excel(f); df.loc[1, "组别"] = "电击B"; df.to_excel(f, index=False)
w._grp_import(); print("after import", w.project.samples)
shot("g1_table")
# 用假数据（8 只动物）测试统计
w.project.samples = json.load(open(samples, encoding="utf-8")); w._sync_groups_panel(); pump(1)
w.st_ctrl.setCurrentText("对照"); w.st_depth.setValue(8)
w.run_stats(region_sec=pd.read_pickle(fake)); wait_task()
print(w.st_summary.text())
shot("g2_stats")
print("DONE")
