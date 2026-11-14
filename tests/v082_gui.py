"""v0.8.2 界面测试：② 通道设置（染色下拉框、不使用、拆分用通道）、④ 串色校正（开关、提示、滑块）、⑧ 荧光强度勾选框、
打开江丰 .kfbf 图片看图。
用法：xvfb-run python tests/run.py tests/v082_gui.py <已识别的项目文件夹> <截图目录> [某个 .kfbf 文件]"""
import sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from qtpy.QtWidgets import QComboBox, QCheckBox, QSlider
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project, ImageEntry

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
kf = sys.argv[3] if len(sys.argv) > 3 else None
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


w._load_project(Project.load(proj)); pump(2)
p = w.project
# ② 通道设置
w.steps.setCurrentRow(1); pump(1)
combos = w.channels_box.findChildren(QComboBox)
checks = w.channels_box.findChildren(QCheckBox)
print("channel combos", [c.currentText() for c in combos], "use", [c.isChecked() for c in checks])
assert len(combos) == len(p.channel_names) and combos[0].currentText() == "DAPI（AF405）"
w.channels_box.grab().save(str(out / "c1_通道设置.png"))
combos[1].setCurrentText("不使用"); pump(1)
assert 1 in p.unused_channels and not p.detection.channels[1].detect
print("after unused", p.channel_names, msgs[-1:] )
combos = w.channels_box.findChildren(QComboBox)
combos[1].setCurrentText("Red（AF594）"); pump(1)
assert 1 not in p.unused_channels and p.channel_names[1] == "Red"
combos = w.channels_box.findChildren(QComboBox)
combos[1].setCurrentText("DAPI（AF405）"); pump(1)
assert p.channel_names[1] == "Red" and "不能同名" in msgs[-1]
checks = w.channels_box.findChildren(QCheckBox)
checks[1].setChecked(False); pump(0.5)
assert p.tissue_channels() == [0]
checks = w.channels_box.findChildren(QCheckBox); checks[1].setChecked(True); pump(0.5)
assert p.tissue_channels() is None
# ④ 串色校正
w.steps.setCurrentRow(3); pump(1)
p.unmix = {"enabled": True}; w._sync_detection_panel(); pump(0.5)
assert w.um_on.isChecked() and not w.um_warn.isVisible()
w._unmix_estimate(); wait_task()
print("unmix", {k: v for k, v in p.unmix.items() if k != "detail"})
sl = w.um_on.parentWidget().findChildren(QSlider)
print("sliders", len(sl))
assert len(sl) == len(p.unmix["coef"])
sl[0].setValue(150); pump(0.5)
assert abs(p.unmix["coef"][sorted(p.unmix["coef"])[0]] - 0.15) < 1e-6
w.um_on.parentWidget().grab().save(str(out / "c2_串色校正_开.png"))
w.um_on.setChecked(False); pump(0.5)
assert not p.unmix["enabled"] and w.um_warn.isVisible() and "共标数会偏高" in msgs[-1]
w.um_on.parentWidget().grab().save(str(out / "c3_串色校正_关.png"))
w.um_on.setChecked(True); pump(0.5)
# ⑧ 荧光强度勾选框
w.steps.setCurrentRow(7); pump(0.5)
assert not w.exp_intensity.isChecked()
# 江丰 .kfbf
if kf:
    from cfos.core.slide_io import open_slide
    kp = Project(folder=out / "kfbf项目"); s_ = open_slide(kf); kp.ensure_channels(s_.channels); s_.close()
    kp.images.append(ImageEntry(path=kf)); kp.save()
    w._load_project(Project.load(kp.folder)); pump(3)
    print("kfbf channels", w.project.channel_names)
    print("kfbf levels", w.slide.n_levels, w.slide.level_shapes[:3], [c.name for c in w.slide.channels])
    w.steps.setCurrentRow(1); pump(2)
    mw = w.stack.widget(1).minimumSizeHint().width(); print("panel width", mw); assert mw <= 440
    w.grab().save(str(out / "c4_kfbf看图.png"))
print("DONE")
