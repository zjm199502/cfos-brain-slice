"""⑨ 截图制图 界面测试：画框、保存、改名、固定大小、预览、导出、重新打开后仍在。
用法：xvfb-run python tests/run.py tests/figure_smoke.py <已配准的项目文件夹> <截图目录>"""
import json, sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from qtpy.QtGui import QPainter, QImage
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
from cfos.core.project import Project

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
msgs = []
w.info = lambda t: (msgs.append(t), print("INFO:", t, flush=True)); w.error = lambda t: print("ERROR:", t, flush=True)
w._open_folder = lambda p: print("open folder", p)


def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)


def shot(name, widget=None):
    pump(1.2)
    if widget is not None:
        widget.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True); return
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


def rect(x, y, rw, rh):
    return np.array([[y, x], [y, x + rw], [y + rh, x + rw], [y + rh, x]], float)


w._load_project(Project.load(proj)); pump(2)
for l in w.viewer.layers:
    if l.name.startswith("细胞-"):
        l.visible = False
assert STEPS[8] == "⑨ 截图制图"
w.steps.setCurrentRow(8); pump(1.5)
mw = w.stack.widget(8).minimumSizeHint().width()
print("panel min width", mw, "超宽!" if mw > 440 else "")
assert mw <= 440
shot("f1_panel_empty")

# 1. 画两个框并保存
im = w.image; s = im.sections[0]; x, y, sw, sh = s.bbox0; px = w.slide.pixel_size_um
w.fig_draw_mode(); pump(0.5)
layer = w.viewer.layers["截图框"]
assert layer.mode == "add_rectangle"
layer.add_rectangles([rect(x + sw * 0.30, y + sh * 0.40, 1500 / px, 1000 / px),
                      rect(x + sw * 0.62, y + sh * 0.22, 600 / px, 450 / px)])
w.fig_save_rois(); pump(0.5)
print("rois", im.figure_rois)
assert [r["name"] for r in im.figure_rois] == ["ROI1", "ROI2"]
assert abs(im.figure_rois[0]["w"] * px - 1500) < 1
# 2. 改名
w.fig_list.item(0).setText("海马区"); pump(0.3)
assert im.figure_rois[0]["name"] == "海马区"
w.fig_list.setCurrentRow(0); w._fig_zoom(w.fig_list.item(0)); pump(0.5)
w.layer_tools.refresh()
shot("f2_rois_drawn")

# 3. 固定大小：再画一个随手的框，保存后应为 500×500 µm
w.fig_fixed.setChecked(True); w.fig_fw.setValue(500); w.fig_fh.setValue(500)
layer = w.viewer.layers["截图框"]  # 保存后图层已重建
layer.add_rectangles([rect(x + sw * 0.15, y + sh * 0.60, 300 / px, 900 / px)])
w.fig_save_rois(); pump(0.3)
r3 = im.figure_rois[2]
print("fixed roi", r3)
assert abs(r3["w"] * px - 500) < 1 and abs(r3["h"] * px - 500) < 1 and r3["name"] == "ROI3"
w.fig_fixed.setChecked(False)

# 4. 设置 + 预览
w.fig_cells.setChecked(True); w.fig_regions.setChecked(True); w.fig_region_names.setChecked(True)
w.fig_inset.setChecked(True); w.fig_separate.setChecked(True); w.fig_tif.setChecked(True); w.fig_svg.setChecked(True)
w.sb_len.setCurrentIndex(w.sb_len.findData(200.0)); w.sb_pos.setCurrentText("右下")
pump(0.8)
st = w.project.figure
print("settings", json.dumps(st, ensure_ascii=False)[:300])
assert st["overlay_cells"] and st["scale"]["length_um"] == 200 and set(st["formats"]) == {"png", "tif", "svg"}
w.fig_list.setCurrentRow(0); w.fig_preview(); pump(2)
assert hasattr(w, "_fig_dialog")
shot("f3_preview", w._fig_dialog); w._fig_dialog.accept()
shot("f3b_panel_settings")

# 5. 导出
w.fig_export(False); wait_task()
files = sorted((proj / "截图").glob("*"))
print("exported", len(files), [f.name for f in files])
assert any(f.name.endswith("海马区_合并.png") for f in files)
assert any(f.name.endswith("海马区_DAPI.tif") for f in files) and any(f.name.endswith("ROI2_合并.svg") for f in files)
assert msgs and msgs[-1].startswith("已导出")
w._refresh_step_marks(); print("step9 mark:", w.steps.item(8).text())
shot("f4_exported")

# 6. 删除一个框
w.fig_list.setCurrentRow(2); w.fig_delete_roi(); pump(0.3)
assert len(im.figure_rois) == 2

# 7. 重新打开项目：框和设置仍在
p2 = Project.load(proj)
assert [r["name"] for r in p2.images[0].figure_rois] == ["海马区", "ROI2"]
assert p2.figure["scale"]["length_um"] == 200
# 8. 所有面板宽度
for i, nm in enumerate(STEPS):
    w.steps.setCurrentRow(i); pump(0.6)
    mw = w.stack.widget(i).minimumSizeHint().width()
    print(nm, "min width", mw, "超宽!" if mw > 440 else "")
    assert mw <= 440
print("DONE")
