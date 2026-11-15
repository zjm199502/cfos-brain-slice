"""v0.8.3 界面测试：③ 每个通道的显示颜色（常用颜色 / 默认）、看图色表跟着变；
新建项目时 Far Red 存成黄色的图片自动认成 Far red 并显示黄色；⑦ 指标下拉框首项是“X+ 占 DAPI+ 的比例 %”。
用法：xvfb-run python tests/run.py tests/v083_gui.py <已识别的项目文件夹> <截图目录>"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from qtpy.QtWidgets import QComboBox
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
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


def shot(name, widget=None):
    pump(1.0)
    if widget is not None:
        widget.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True); return
    from qtpy.QtGui import QPainter, QImage
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


w._load_project(Project.load(proj)); pump(2)
p = w.project
c = len(p.channel_names) - 1
# ③ 显示颜色
w.steps.setCurrentRow(2); pump(1)
boxes = [b for b in w.findChildren(QComboBox) if b.findText("自定义…") >= 0 and b.isVisible()]
assert len(boxes) == len(p.channel_names), len(boxes)
assert boxes[c].currentText() == "默认（按染色）"
shot("d1_显示颜色_默认")
boxes[c].setCurrentText("黄"); pump(1)
assert p.custom_color(c) == (255, 255, 0)
cm = w.image_layers[c].colormap
assert np.allclose(cm.colors[-1][:3], [1, 1, 0]), cm.colors
assert Project.load(proj).custom_color(c) == (255, 255, 0)      # 已保存进项目
for layer in w.viewer.layers:
    if layer.name.startswith("细胞-"):
        layer.visible = False
s0 = p.images[0].sections[0].bbox0
w.viewer.camera.center = (s0[1] + s0[3] * 0.35, s0[0] + s0[2] * 0.3); w.viewer.camera.zoom = 0.35; pump(2)
shot("d2_显示颜色_黄色")
# 换图片 / 重新打开项目后仍是黄色
w._load_project(Project.load(proj)); pump(2)
assert np.allclose(w.image_layers[c].colormap.colors[-1][:3], [1, 1, 0])
w.steps.setCurrentRow(2); pump(1)
boxes = [b for b in w.findChildren(QComboBox) if b.findText("自定义…") >= 0 and b.isVisible()]
assert boxes[c].currentText() == "黄"
shot("d3_显示颜色面板", w.display_rows.parentWidget())
boxes[c].setCurrentText("默认（按染色）"); pump(1)
assert w.project.custom_color(c) is None
# ⑦ 指标下拉框
w.steps.setCurrentRow(6); pump(1)
print("metric first", w.st_metric.itemText(0))
assert w.st_metric.itemText(0).endswith("占 DAPI+ 的比例 %")
shot("d4_组间统计指标", w.st_metric.parentWidget())
print("messages", msgs)
print("DONE")
app.quit()
