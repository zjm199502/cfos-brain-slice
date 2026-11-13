"""v0.8 新功能的界面测试：厚切片默认去卷积 + 提醒、脑区平均荧光强度、帮助菜单（关于 / 检查更新 / 安装补丁）。
用法：xvfb-run python tests/run.py tests/v08_smoke.py <已配准且已识别的项目文件夹> <截图目录>
（检查更新部分需要设置 CFOS_UPDATE_URL 指向测试用的 latest.json；不设置时跳过）"""
import os, sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QDialog
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
from cfos.core.project import Project

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
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
    """把弹出的对话框截图后关闭。"""
    def f():
        for d in QApplication.topLevelWidgets():
            if isinstance(d, QDialog) and d.isVisible():
                d.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True)
                d.reject(); return
        QTimer.singleShot(300, f)
    QTimer.singleShot(800, f)


w._load_project(Project.load(proj)); pump(2)
print("title", w.windowTitle()); assert "v0.8" in w.windowTitle()
for i, nm in enumerate(STEPS):
    w.steps.setCurrentRow(i); pump(0.4)
    mw = w.stack.widget(i).minimumSizeHint().width()
    assert mw <= 440, (nm, mw)

# 1. 厚切片 → 默认去卷积 + 提醒
w.steps.setCurrentRow(3); pump(0.5)
w.thick_combo.setCurrentIndex(w.thick_combo.findData("thick")); w.apply_thickness(); pump(0.5)
assert all(p.deconv for p in w.project.detection.channels.values())
assert "去卷积" in msgs[-1] and "原图" in msgs[-1]
w.thick_combo.setCurrentIndex(w.thick_combo.findData("thin")); w.apply_thickness(); pump(0.5)
assert not any(p.deconv for p in w.project.detection.channels.values())

# 2. 帮助菜单
acts = [a.text() for m in w.menuBar().actions() for a in m.menu().actions()]
print("help menu", acts)
assert any("关于" in a for a in acts) and any("检查更新" in a for a in acts) and any("从文件安装补丁" in a for a in acts)
grab_dialog("u1_about"); w.show_about(); pump(0.5)
from cfos.gui.update_panel import about_html
assert "zjm199502@163.com" in about_html() and "张金铭" in about_html()

# 3. 检查更新（本地测试服务器）
if os.environ.get("CFOS_UPDATE_URL"):
    grab_dialog("u2_update_available")
    w.check_updates(silent=False)
    t = time.time()
    while time.time() - t < 15 and not (out / "u2_update_available.png").exists():
        pump(0.3)
    assert (out / "u2_update_available.png").exists()

# 4. 导出 Excel：脑区平均荧光强度
import pandas as pd
from cfos.core.export import export_excel
f = proj / "v08_test.xlsx"
export_excel(w.project, f)
r = pd.read_excel(f, sheet_name="按脑区×脑片")
cols = [c for c in r.columns if "平均荧光强度" in c]
print("intensity cols", cols)
assert len(cols) == 2 * len(w.project.channel_names)
sel = r[(r["半脑"] == "左右合计") & (r["脑区缩写"] == "root")]
print(sel[cols].to_string())
assert sel[cols[0]].notna().all()
pt = pd.read_excel(f, sheet_name="分析参数")
print(pt.head(6).to_string())
print("DONE")
