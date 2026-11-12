"""教学模式界面：每步要点和参考图、步骤打勾、打开示例项目。用法：xvfb-run python tests/teaching_smoke.py <放示例项目的文件夹> <截图目录>"""
import sys, time, shutil, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from qtpy.QtWidgets import QFileDialog
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
base, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
shutil.rmtree(base / "cFos示例项目", ignore_errors=True); base.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(base))
w = MainWindow(); w.resize(1600, 950); w.show()
w.info = lambda t: print("INFO:", t[:60], flush=True); w.error = lambda t: print("ERROR:", t, flush=True)
def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)
w.teach_cb.setChecked(True); pump(1)
w.grab().save(str(out / "t0.png"))
w.open_demo_project(); pump(3)
print("项目", w.project.folder, "图片", [im.path for im in w.project.images], w.project.channel_names)
for i in range(len(STEPS)):
    w.steps.setCurrentRow(i); pump(1.2)
    mw = w.stack.widget(i).minimumSizeHint().width()
    print(w.steps.item(i).text(), "min width", mw, "超宽!" if mw > 440 else "")
    w.grab().save(str(out / f"t_step{i}.png"))
w.teach_cb.setChecked(False); pump(1)
print("关闭后可见:", [b.isVisible() for b in w._lesson_boxes][:3], w.steps.item(0).text())
print("DONE")
