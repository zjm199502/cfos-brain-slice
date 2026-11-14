"""v0.8.1 界面测试：新名称和图标、关于页（无姓名、无引用提示）、帮助菜单（离线安装微调组件）、
质量检查新列、分组表“行为分数”列和“脑区活动与行为相关”。
用法：xvfb-run python tests/run.py tests/v081_smoke.py <已配准且已识别的项目> <假脑区表.pkl> <假分组.json> <截图目录>"""
import json, sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from napari.qt import get_qapp
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QDialog
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
from cfos.core.project import Project
from cfos import APP_NAME

proj, fake, samples, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
from cfos.app import app_icon
app.setWindowIcon(app_icon())
w = MainWindow(); w.setWindowIcon(app_icon()); w.resize(1600, 950); w.show()
msgs = []
w.info = lambda t: (msgs.append(t), print("INFO:", t.replace("\n", " | "), flush=True))
w.error = lambda t: (msgs.append(t), print("ERROR:", t, flush=True))


def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)


def grab_dialog(name):
    def f():
        for d in QApplication.topLevelWidgets():
            if isinstance(d, QDialog) and d.isVisible():
                d.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True)
                d.reject(); return
        QTimer.singleShot(300, f)
    QTimer.singleShot(800, f)


w._load_project(Project.load(proj)); pump(2)
print("title", w.windowTitle())
assert w.windowTitle().startswith("FosMap 脑片活动图谱 v0.8.1"), w.windowTitle()
assert not w.windowIcon().isNull() and w.windowIcon().availableSizes(), "没有程序图标"
for i, nm in enumerate(STEPS):
    w.steps.setCurrentRow(i); pump(0.3)
    mw = w.stack.widget(i).minimumSizeHint().width()
    assert mw <= 440, (nm, mw)

# 1. 关于、帮助菜单
from cfos.gui.update_panel import about_html, contact_html
h = about_html()
assert APP_NAME in h and "<img" in h and "张金铭" not in h and "引用" not in h and "zjm199502@163.com" in h, h
assert "张金铭" not in contact_html()
acts = [a.text() for m in w.menuBar().actions() for a in m.menu().actions()]
print("help menu", acts)
assert any("关于 FosMap" in a for a in acts) and any("安装微调组件" in a for a in acts)
grab_dialog("v081_about"); w.show_about(); pump(0.5)

# 2. 质量检查：新列
from cfos.core.quality import quality_table
q = quality_table(w.project)
print(q.T.to_string())
assert "阳性细胞落在白质/脑室 %" in q.columns

# 3. 分组表：行为分数列（按动物记录）
w.steps.setCurrentRow(6); pump(1)
t = w.grp_table
assert t.columnCount() == 5 and t.horizontalHeaderItem(4).text() == "行为分数"
t.item(0, 2).setText("M甲"); pump(0.3)
t.item(0, 4).setText("35.5"); pump(0.3)
assert w.project.behavior.get("M甲") == 35.5, w.project.behavior
t.item(0, 4).setText("abc"); pump(0.3)
assert "不是数字" in msgs[-1] and w.project.behavior.get("M甲") == 35.5
assert t.item(0, 4).text() == "35.5"

# 4. 脑区活动与行为相关（假数据：8 只动物，行为分数与 ACA 的密度单调相关）
w.project.samples = json.load(open(samples, encoding="utf-8")); w._sync_groups_panel(); pump(0.5)
rs = pd.read_pickle(fake)
from cfos.core.stats import per_animal, metric_options
metric = metric_options(w.project.channel_names)[0]
an = per_animal(rs, w.project.samples)
dens = an[(an["脑区缩写"] == "ACA") & (an["半脑"] == "左右合计")].set_index("动物")[metric]
w.project.behavior = {a: float(np.log1p(v) * 10) for a, v in dens.items()}
w._sync_groups_panel(); pump(0.5)
assert len(w.project.behavior) == 8
w.st_metric.setCurrentText(metric); w.st_depth.setValue(8)
w.bh_name.setText("冻结时间 %"); w.bh_group.setCurrentText("全部动物")
w.run_behavior(region_sec=rs); wait_task()
print(w.bh_summary.text())
assert "ACA" in w.bh_summary.text() and "r = 1.00" in w.bh_summary.text(), w.bh_summary.text()
sa = w.stack.widget(6).parent()
w.grab().save(str(out / "v081_groups_behavior.png")); print("shot v081_groups_behavior")
# 只用一个组（4 只）：应提示太少
w.bh_group.addItem("对照"); w.bh_group.setCurrentText("对照"); w.run_behavior(region_sec=rs); wait_task()
assert "至少需要 5 只" in msgs[-1], msgs[-1]
print("DONE")
