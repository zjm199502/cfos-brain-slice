"""v0.8.5 配准界面测试：自动配准（前后位置校正、统一角度、⚠ 提示）、拖滑块快速预览、“已检查”按钮。
用法：xvfb-run python tests/run.py tests/v085_reg_gui.py <图片> <输出目录>"""
import shutil, sys, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from qtpy.QtGui import QPainter, QImage
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
from cfos.core import registration as R

img, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
proj = out / "proj"; shutil.rmtree(proj, ignore_errors=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t, flush=True); w.error = lambda t: print("ERROR:", t, flush=True)


def pump(sec=1.0):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents(); time.sleep(0.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)


def shot(name):
    pump(1.5)
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


p = Project(folder=proj); p.save(); w._load_project(p)
w.add_images_programmatic([img]); pump(2)
w.steps.setCurrentRow(1); w.auto_sections(False); wait_task()
w.steps.setCurrentRow(5); pump(1)
t0 = time.time()
while "itk" not in R._ITK and time.time() - t0 < 60:     # S1：进入配准页面后在后台预加载
    pump(0.5)
print(f"后台预加载 elastix 用了 {time.time() - t0:.1f} 秒", flush=True)
assert "itk" in R._ITK
assert w.reg_refine_ap.isChecked() and w.reg_unify.isChecked()
t0 = time.time(); w.auto_register(False); wait_task()
print(f"自动配准 {len(p.images[0].sections)} 张脑片用了 {time.time() - t0:.1f} 秒", flush=True)
for s in p.images[0].sections:
    r = R.Registration.from_dict(s.registration)
    print(s.name, r.status, f"Bregma {R.adjusted_plane(r).ap_mm():+.2f}", r.warnings, flush=True)
    assert r.done and r.nonlinear and r.nl_method == R.TEXTURE and "前后位置自动" in r.status
for nm in [l.name for l in w.viewer.layers if l.name.startswith("细胞-") or l.name.startswith("预览")]:
    w.viewer.layers[nm].visible = False
w.reg_sec_combo.setCurrentIndex(0); pump(2)
shot("r1_自动配准完成")

# ---- A3：人为加一条提示，检查 ⚠ 显示、“已检查”按钮
s0 = p.images[0].sections[0]
r = R.Registration.from_dict(s0.registration); r.warnings = ["与同一张图片里其他脑片的前后位置相差 0.7 mm 以上"]
s0.registration = r.to_dict(); w._sync_registration_panel(); pump(2)
assert w.reg_sec_combo.itemText(0).startswith("⚠"), w.reg_sec_combo.itemText(0)
assert w.reg_ok_btn.isVisible() and "重点检查" in w.reg_status.text()
shot("r2_需要检查的提示")
w.reg_ok_btn.click(); pump(1)
assert not w.reg_sec_combo.itemText(0).startswith("⚠") and not w.reg_ok_btn.isVisible()
assert R.Registration.from_dict(s0.registration).warnings == []

# ---- S2：拖滑块时先画快速预览，停手后再画精细边界；ap_auto 不受滑块影响
auto = R.Registration.from_dict(s0.registration).ap_auto
sl_ap = w.reg_sliders["ap_shift"][0]
times = []
orig = w._reg_redraw
def timed():
    fast = w._reg_fast; t = time.time(); orig(); times.append((fast, time.time() - t))
w._reg_redraw = timed
w._reg_timer.timeout.disconnect(); w._reg_timer.timeout.connect(timed)
w._reg_fine_timer.timeout.disconnect(); w._reg_fine_timer.timeout.connect(timed)
for v in range(1, 6):
    sl_ap.setValue(v); pump(0.15)
pump(1.0)
fast = [t for f, t in times if f]; fine = [t for f, t in times if not f]
print(f"快速预览 {len(fast)} 次，平均 {1000 * sum(fast) / max(len(fast), 1):.0f} ms；精细 {len(fine)} 次，"
      f"平均 {1000 * sum(fine) / max(len(fine), 1):.0f} ms", flush=True)
assert fast and fine and times[-1][0] is False
r = R.Registration.from_dict(s0.registration)
assert r.ap_auto == auto and abs(r.ap_shift - 5) < 1e-6 and r.status == "已手动调整"
shot("r3_手动调整后")
w._reg_reset(); pump(1)
r = R.Registration.from_dict(s0.registration)
assert r.ap_shift == 0 and r.ap_auto == auto, "恢复自动结果应保留自动校正的前后位置"
print("DONE", flush=True)
