import os
import runpy
import sys
import time
from pathlib import Path

base = Path(os.environ["CFOS_BASE_APP"])
sys.path.insert(0, str(base))
runpy.run_path(str(base / "main.py"))
from cfos.core import updater
if sys.argv[-1] == "seed":
    updater.install_patch(Path(__file__).with_name("candidate.zip"))
    sys.exit(0)
import cfos
print("Loaded", cfos.__version__, cfos.__file__, flush=True)
import numpy as np
import tifffile
from napari.qt import get_qapp
from cfos.gui.main_window import MainWindow
from cfos.core.project import Project
from cfos.core.sections import Section

root = Path(os.environ["RUNNER_TEMP"])
app = get_qapp()
w = MainWindow()
w.show()
def pump():
    for _ in range(20):
        app.processEvents()
        time.sleep(.01)
project = Project(root / "test-project")
project.save()
w._load_project(project)
for i, (size, channels, pyramid) in enumerate([((256,512),1,False), ((1024,2048),3,False),
                                              ((1024,19000),3,True)]):
    pixels = np.zeros((channels, *size), np.uint16)
    pixels[:, 20:160, 40:280] = 4000
    pixels = pixels[0] if channels == 1 else pixels
    axes = "YX" if channels == 1 else "CYX"
    path = root / f"scan-{i}.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        opts = dict(tile=(128,128), photometric="minisblack", compression="deflate")
        writer.write(pixels, metadata={"axes": axes}, subifds=2 if pyramid else None, **opts)
        if pyramid:
            for down in (2, 4):
                writer.write(pixels[..., ::down, ::down], subfiletype=1, **opts)
    del pixels
    print("Open", path, flush=True)
    w.add_images_programmatic([path])
    w.image_combo.setCurrentIndex(i)
    pump()
    assert len(w.image_layers) == channels, [layer.name for layer in w.image_layers]
    assert w.qt_viewer.screenshot(flash=False).size
    print("Rendered", channels, size, flush=True)
w.image_combo.setCurrentIndex(0)
pump()
w.close()
pump()
print("PASS native close", flush=True)
