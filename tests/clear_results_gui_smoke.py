"""各步骤页的“清除本步结果”按钮：确认框内容、清除后界面刷新。用法：xvfb-run python tests/clear_results_gui_smoke.py [截图目录]"""
import sys, tempfile, time, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tifffile
from napari.qt import get_qapp
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
w = MainWindow(); w.resize(1500, 920); w.show()
msgs = []; w.info = msgs.append; w.error = msgs.append
def pump(s=0.5):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)
w._load_project(Project.load(p.folder)); pump(1.5)
w._apply_scope({"img0": ("S0",)}); pump()
assert sorted(w.clear_buttons) == sorted(["all", "sections", "display", "detect", "coloc", "registration",
                                          "groups", "export", "figure"]), w.clear_buttons
w.steps.setCurrentRow(3); pump()
scroll = w.stack.currentWidget(); scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum()); pump()
if out:
    out.mkdir(parents=True, exist_ok=True)
    w.stack.currentWidget().grab().save(str(out / "p4_clear_button.png"))
from cfos.core import clear_results as CR
preview = CR.clear(w.project, "detect", w._processing_scope)
box, go, check = w._clear_dialog("detect", preview, f"同时清除人工修正的细胞（{preview.manual_records} 条补加 / 删除）")
box.show(); pump()
if out:
    box.grab().save(str(out / "p4_clear_dialog.png"))
box.close()
# 取消：什么都不删
assert w.clear_step_results("detect", confirm=lambda r: (False, False)) is None
assert set(w.project.load_cells(w.project.images[0])["section"]) == {"S0", "S1", "S2"}
r = w.clear_step_results("detect", confirm=lambda r: (True, False)); pump()
assert set(w.project.load_cells(w.project.images[0])["section"]) == {"S1", "S2"}, r
w.steps.setCurrentRow(0); pump()
w._refresh_project_progress(); pump()
states = [w.progress_table.item(3, c).text() for c in range(w.progress_table.columnCount())]
print("④ 行:", states)
assert "待识别" in states[0], states
w.steps.setCurrentRow(5); pump()
r = w.clear_step_results("registration", confirm=lambda r: (True, False)); pump()
assert not w.project.images[0].sections[0].registration
r = w.clear_step_results("coloc", confirm=lambda r: (True, False)); pump()
assert "没有" in msgs[-1], msgs   # S0 已经没有细胞
w._apply_scope({"img1": None}); pump()
w.steps.setCurrentRow(1); pump()
r = w.clear_step_results("sections", confirm=lambda r: (True, False)); pump()
assert w.project.images[1].sections == [] and w.sec_table.rowCount() == 0
assert Project.load(p.folder).images[1].sections == []
print("DONE")
