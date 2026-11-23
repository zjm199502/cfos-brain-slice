"""Start an isolated released app at 0.9.7, install its native update, reopen and check CZI pixels."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

here = Path(__file__).resolve().parent
root = Path(os.environ["RUNNER_TEMP"]).resolve()
base = Path(os.environ["CFOS_BASE_APP"]).resolve()
mode = sys.argv[1]
if len(sys.argv) == 2:
    with tempfile.TemporaryDirectory(dir=root) as profile:
        env = dict(os.environ, CFOS_UPDATES_DIR=profile)
        env.pop("CFOS_UPDATE_URL", None)
        for stage in ("seed", "upgrade", "read"):
            subprocess.run([sys.executable, "-u", __file__, mode, stage], env=env, check=True)
            banner = subprocess.check_output([sys.executable, str(base / "main.py"), "--version"],
                                             env=env, text=True)
            assert ("0.9.7" if stage == "seed" else "0.9.8") in banner, banner
            print(banner.strip(), flush=True)
    print("PASS Apple Silicon", mode, flush=True)
    sys.exit()

sys.path.insert(0, str(base))
runpy.run_path(str(base / "main.py"))
import cfos
from cfos.core import updater

step = sys.argv[2]
if step == "seed":
    assert cfos.__version__ == "0.9.0"
    assert updater.install_patch(root / "prev.zip")["version"] == "0.9.7"
elif step == "upgrade":
    assert cfos.__version__ == "0.9.7"
    assert all(u.endswith("/updates/v0.9/macos/latest.json") for u in updater.manifest_urls())
    if mode == "staged":
        os.environ["CFOS_UPDATE_URL"] = (here / "candidate/latest.json").as_uri()
    result = updater.check()
    assert result["status"] == "patch", result
    assert result["manifest"] == json.loads((here / "candidate/latest.json").read_bytes())
    package = updater.download_patch(result)
    assert updater.read_patch_info(package)["platform"] == "macos"
    assert updater.install_patch(package)["version"] == "0.9.8"
else:
    import numpy as np
    from cfos.core.project import Project
    from napari.qt import get_qapp
    from qtpy.QtWidgets import QFileDialog
    from cfos.gui.main_window import MainWindow

    assert cfos.__version__ == "0.9.8", cfos.__version__
    assert Path(cfos.__file__).is_relative_to(Path(os.environ["CFOS_UPDATES_DIR"])), cfos.__file__
    source = root / "two-wells.czi"
    expected = ["df447547ba99429178b441fc2089abc94379df15e080c678f40c2c601cc858f7",
                "de5f4bbcea7c46d02b038a0de217c2953ccb08dc761d496b22d2d22f44a9506f"]
    app = get_qapp()
    window = MainWindow()
    errors = []
    window.error = errors.append
    window.info = print
    p = Project(root / f"project-{mode}")
    window._load_project(p)
    with patch.object(QFileDialog, "getOpenFileNames", return_value=([str(source)], "")) as pick:
        window.add_images()
        assert "*.czi" in pick.call_args.args[3]
        assert [im.scene_id for im in p.images] == [0, 1], errors
        window.add_images()
        assert len(p.images) == 2
    window._load_project(Project.load(p.folder))
    for i, checksum in enumerate(expected):
        window.image_combo.setCurrentIndex(i)
        assert len(window.image_layers) == 2 and window.slide.scene_id == i
        assert abs(window.slide.pixel_size_um - 0.45715068250381635) < 1e-10
        pixels = window.slide.read(0)
        assert pixels.shape == (2, 1416, 1960) and pixels.dtype == np.uint16
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == checksum
        print("Verified real ZEISS scene", i, "versus libCZI", checksum, flush=True)
    assert not errors, errors
    window.close()
    album = unittest.defaultTestLoader.discover(str(here / "candidate"), pattern="test_czi_reader.py")
    assert album.countTestCases() >= 6
    assert unittest.TextTestRunner(verbosity=2).run(album).wasSuccessful()
