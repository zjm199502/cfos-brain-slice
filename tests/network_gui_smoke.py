"""功能网络界面测试。用法：xvfb-run python tests/network_gui_smoke.py <项目> <假脑区表.pkl> <假分组.json> <截图目录>"""
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
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)
def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)
w._load_project(Project.load(proj)); pump(2)
w.project.samples = json.load(open(samples, encoding="utf-8"))
w.steps.setCurrentRow(6); pump(1)
print("min width", w.stack.widget(6).minimumSizeHint().width())
w.st_metric.setCurrentText("Red+ 密度 (个/mm²)")
w.net_perm.setValue(300)
w.run_network(region_sec=pd.read_pickle(fake))
while w.task is not None and w.task.isRunning():
    pump(0.2)
pump(1)
print(w.net_summary.text()); print("saved params", w.project.network)
w.stack.parentWidget().parentWidget().verticalScrollBar().setValue(10000); pump(1)
w.grab().save(str(out / "net_panel.png"))
# 真实点击按钮（按钮信号会带 checked 参数，曾导致崩溃）
from qtpy.QtWidgets import QPushButton
for txt in ("计算组间统计并画图", "计算功能网络并画图"):
    b = next(x for x in w.findChildren(QPushButton) if x.text() == txt)
    b.click(); pump(0.5)
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(0.5); print("点击", txt, "完成")
print("DONE")
