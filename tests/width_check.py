"""检查右侧面板各页的最小宽度不超过可用宽度（否则文字会被截断）。"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
app = get_qapp()
from cfos.gui.main_window import MainWindow, STEPS
from cfos.core.project import Project
w = MainWindow(); w.show(); app.processEvents()
if len(sys.argv) > 1:
    w._load_project(Project.load(sys.argv[1]))
bad = 0
for i in range(w.stack.count()):
    w.steps.setCurrentRow(i); app.processEvents()
    mw = w.stack.widget(i).minimumSizeHint().width()
    print(STEPS[i], mw, "超宽!" if mw > 440 else "")
    bad += mw > 440
sys.exit(1 if bad else 0)
