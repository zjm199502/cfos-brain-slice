"""Exercise the candidate through an isolated installation of the published offline app."""
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import unittest

root = Path(os.environ["RUNNER_TEMP"])
repo = Path(__file__).resolve().parents[1]
bundle = root / "unpacked/FosFolio脑片图册.app/Contents/Resources/app"
updates = root / "isolated-candidate"
assert platform.machine() == "arm64"
os.environ.update(CFOS_BASE_APP=str(bundle), CFOS_UPDATES_DIR=str(updates))
sys.path.insert(0, str(bundle))
if len(sys.argv) == 1:
    from cfos.core import updater
    assert updater.base_version() == "0.9.0"
    assert updater.install_patch(repo / "qa-mac/cfos-patch-0.9.4.zip")["version"] == "0.9.4"
    version = subprocess.check_output([sys.executable, str(bundle / "main.py"), "--version"], text=True)
    assert "0.9.4" in version, version
    print("Release launcher after restart:", version.strip(), flush=True)
    subprocess.run([sys.executable, "-u", __file__, "restarted"], check=True)
else:
    runpy.run_path(str(bundle / "main.py"))
    import cfos
    from napari.components import ViewerModel
    from napari._qt.qt_viewer import QtViewer
    import numpy as np
    from qtpy.QtWidgets import QApplication
    from cfos.core.atlas_contours import boundary_paths
    from cfos.gui.atlas_outline import set_outline_paths

    assert cfos.__version__ == "0.9.4" and Path(cfos.__file__).is_relative_to(updates)
    app = QApplication.instance() or QApplication([])
    sys.path.insert(0, str(repo / "tests"))
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_atlas_contours", "test_atlas_outline", "test_scope_dialog", "test_scoped_display",
        "test_processing_scope",
    ])
    assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

    viewer = ViewerModel()
    widget = QtViewer(viewer)
    widget.resize(500, 400)
    widget.show()
    layer = viewer.add_image(np.zeros((100, 90, 4), np.uint8), rgb=True,
                             scale=(2, 2), translate=(21, 11))
    labels = np.where(np.indices((100, 90))[0] < 32, 1, 2)
    set_outline_paths(widget, layer, boundary_paths(labels))
    viewer.camera.center = (84, 94)

    def yellow_rows():
        for _ in range(12):
            app.processEvents()
            time.sleep(.04)
        frame = widget.screenshot(flash=False)
        col = frame[:, frame.shape[1] // 2]
        return frame.shape[0], np.flatnonzero((col[:, 0] > 90) & (col[:, 1] > 90) & (col[:, 2] < 80))

    widths = []
    for zoom in (3, 10):
        viewer.camera.zoom = zoom
        height, rows = yellow_rows()
        print(f"Native {zoom}x framebuffer height={height}, outline rows={rows.tolist()}", flush=True)
        assert 1 <= len(rows) <= 6 and abs(rows.mean() - height / 2) <= 3
        widths.append(len(rows))
    assert abs(widths[0] - widths[1]) <= 1
    layer.opacity = 0
    assert len(yellow_rows()[1]) == 0
    layer.opacity = .9
    layer.visible = False
    assert len(yellow_rows()[1]) == 0
    widget.close()
    app.processEvents()
    print("Verified section selection and zoom-invariant outline in the installed arm64 app.", flush=True)
