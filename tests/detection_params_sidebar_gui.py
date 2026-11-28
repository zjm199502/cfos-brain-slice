"""“④ 细胞识别”的各通道参数显示在左边“图层”框下方；窄窗口时退回右栏。
用法：xvfb-run python tests/detection_params_sidebar_gui.py [截图目录]"""
import sys, tempfile, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tifffile
from napari.qt import get_qapp
from qtpy.QtWidgets import QDoubleSpinBox, QFormLayout, QToolButton
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
from test_clear_results import _project, _populate

out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
app = get_qapp(); install_chinese(app)
tmp = tempfile.mkdtemp()
p = _project(tmp); p.ensure_channels([]); _populate(p)
for im in p.images:
    tifffile.imwrite(im.path, np.zeros((2, 64, 256), np.uint16))
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = print; w.error = print


def pump(s=0.4):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def wait_for(cond, s=5):
    t = time.time()
    while not cond():
        assert time.time() - t < s
        pump(0.05)


def in_sidebar():
    return w.det_params.parentWidget() is w.tools_scroll.params_scroll.widget()


def in_right_panel():
    return w.det_params.parentWidget() is w.stack.widget(3).widget()


w._load_project(Project.load(p.folder)); pump(1)
sidebar = w.tools_scroll
# 其他步骤：图层框里没有参数区
w.steps.setCurrentRow(2); pump()
assert not sidebar.params.isVisible()
# ④：参数在图层框里，右栏不再有每通道参数框
w.steps.setCurrentRow(3); pump()
assert w._layout_mode == "wide"
assert sidebar.params.isVisible() and in_sidebar() and w.det_params.isVisible()
assert sorted(w.det_forms) == [c for c in range(len(w.project.channel_names)) if c not in w.project.unused_channels]
assert sidebar.width() >= sidebar.PARAMS_WIDTH, sidebar.width()
if out:
    out.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(out / "p4_wide.png"))
# 参数修改照常保存
form = w.det_forms[0]
cutoff = next(form.itemAt(r, QFormLayout.FieldRole).widget() for r in range(form.rowCount())
              if form.itemAt(r, QFormLayout.LabelRole) and
              form.itemAt(r, QFormLayout.LabelRole).widget().text() == "Cellpose 掩膜阈值")
assert isinstance(cutoff, QDoubleSpinBox)
cutoff.setValue(-2.0)
wait_for(lambda: Project.load(p.folder).detection.channels[0].cellprob_threshold == -2.0)
# 折叠：点标题收起，重新同步后仍保持收起
head = next(b for b in w.det_params.findChildren(QToolButton) if b.text() == w.project.channel_names[0])
head.click(); pump()
assert not form.parentWidget().isVisible()
w._sync_detection_panel(); pump()
assert not w.det_forms[0].parentWidget().isVisible()
head = next(b for b in w.det_params.findChildren(QToolButton) if b.text() == w.project.channel_names[0])
head.click(); pump()
assert w.det_forms[0].parentWidget().isVisible()
# 运行任务时参数不能改
def slow(task):
    while not task.is_cancelled():
        time.sleep(.01)
    raise InterruptedError
w.run_task(slow, lambda _: None); pump(0.2)
assert not w.det_params.isEnabled()
w.task.cancel(); wait_for(lambda: not w.task.isRunning()); pump()
assert w.det_params.isEnabled()
# 离开 ④ 后参数区收起
w.steps.setCurrentRow(4); pump()
assert not sidebar.params.isVisible()
w.steps.setCurrentRow(3); pump()
assert in_sidebar() and sidebar.params.isVisible()
# 窄窗口（图层框默认收起）：参数回到右栏
w.resize(1100, 900); pump()
assert w._layout_mode == "compact" and not sidebar.isVisible()
assert in_right_panel() and w.det_params.isVisible()
if out:
    w.grab().save(str(out / "p4_compact.png"))
# 打开图层框：参数又进入图层框
w.layers_button.setChecked(True); pump()
assert in_sidebar() and sidebar.params.isVisible()
w.layers_button.setChecked(False); pump()
assert in_right_panel()
# 单栏：看步骤时参数在右栏，看图像且打开图层框时在图层框
w.resize(820, 900); pump()
assert w._layout_mode == "single"
w.layers_button.setChecked(True); pump()
assert in_right_panel() and w.det_params.isVisible()
w.pane_button.setChecked(True); pump()
assert in_sidebar()
w.pane_button.setChecked(False); pump()
assert in_right_panel() and w.det_params.isVisible()
w.close()
print("Detection parameters in the layers panel passed")
