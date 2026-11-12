"""无人值守测试第 3 版新功能：z-stack、厚切片参数、排除/自定义区域、阈值重新筛选、模板、导出选项、中断续做。
用法：xvfb-run python tests/v3_smoke.py <已配准的项目> <z-stack 图片> <截图目录>"""
import sys, time, faulthandler, shutil; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project

proj, zfile, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
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
    pump(1.5)
    from qtpy.QtGui import QPainter, QImage
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)

# ---- 已有项目：区域、厚切片、重新筛选、导出
w._load_project(Project.load(proj)); pump(2)
im = w.image
x, y, ww, hh = im.sections[0].bbox0
im.exclusions = [[(x + ww * .45, y + hh * .45), (x + ww * .6, y + hh * .42), (x + ww * .62, y + hh * .6), (x + ww * .44, y + hh * .58)]]
im.rois = [{"name": "", "polygon": [(x + ww * .1, y + hh * .1), (x + ww * .4, y + hh * .1), (x + ww * .4, y + hh * .4), (x + ww * .1, y + hh * .4)]}]
w.steps.setCurrentRow(1); pump(1); w._draw_polygons(); pump(1)
w.save_polygons(); pump(1)
w.roi_list.item(0).setText("左侧皮层"); pump(1)
print("rois", im.rois[0]["name"], len(im.exclusions))
w.viewer.layers.selection.active = w.viewer.layers["排除区域"]; pump(1)
shot("v1_regions")
w.steps.setCurrentRow(3); pump(1)
w.thick_combo.setCurrentIndex(1); w.apply_thickness(); pump(1)
print("thick", {c: (p.bg_radius_um, p.min_contrast) for c, p in w.project.detection.channels.items()}, w.project.coloc.max_dist_um)
shot("v2_thick")
w.refilter(); wait_task(); print(w.det_summary.text())
w.steps.setCurrentRow(4); pump(1); shot("v3_coloc")
w.steps.setCurrentRow(6); pump(1); shot("v4a_export_panel")
w.exp_skip_layers.setChecked(True); w.exp_depth.setValue(8)
from qtpy.QtWidgets import QFileDialog
xl = str(w.project.folder / "v3结果.xlsx")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (xl, ""))
w.export(); wait_task(); shot("v4_export")
import pandas as pd
x_ = pd.read_excel(xl, sheet_name=None)
print({k: v.shape for k, v in x_.items()})
print(x_["按自定义区域"].T)
# 模板
w.project.save_template(out / "模板.json")
# ---- z-stack 新项目
zp = out / "zproj"; shutil.rmtree(zp, ignore_errors=True)
p = Project(folder=zp); p.save(); w._load_project(p); pump(1)
w.add_images_programmatic([zfile]); pump(2)
print("z info:", w.z_info.text())
notes = w.project.load_template(out / "模板.json"); print("template notes", notes)
w.steps.setCurrentRow(0); pump(1); shot("v5_zstack_project")
w.z_combo.setCurrentIndex(1); pump(2)
w.steps.setCurrentRow(1); w.auto_sections(False); wait_task()
print("z sections", [(s.name, s.bbox0) for s in w.image.sections])
if not w.image.sections:
    from cfos.core.sections import Section
    H, W = w.slide.level_shapes[0]
    w.image.sections = [Section(id=1, bbox0=(0, 0, W, H), contour0=[], name="S01")]
w.steps.setCurrentRow(3); w.run_detection(False); wait_task(); print(w.det_summary.text())
w.run_detection(False); wait_task(); print("second:", w.det_summary.text())
shot("v6_zstack_cells")
print("DONE", flush=True)
