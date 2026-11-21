import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
from contextlib import closing

root = Path(os.environ["RUNNER_TEMP"])
app = root / "unpacked/FosFolio脑片图册.app/Contents/Resources/app"
os.environ.update(CFOS_BASE_APP=str(app), CFOS_UPDATES_DIR=str(root / "isolated-updates"))
sys.path.insert(0, str(app))
if len(sys.argv) == 1:
    from cfos.core import updater
    assert updater.base_version() == "0.9.0"
    patch = Path(__file__).parent / "cfos-patch-0.9.3.zip"
    assert updater.install_patch(patch)["version"] == "0.9.3"
    launcher = subprocess.check_output([sys.executable, str(app / "main.py"), "--version"], text=True)
    assert "0.9.3" in launcher, launcher
    print("Verified released launcher:", launcher.strip(), flush=True)
    subprocess.run([sys.executable, __file__, "restarted"], check=True)
else:
    runpy.run_path(str(app / "main.py"))
    import cfos
    import numpy as np
    from cfos.core.project import Project
    assert cfos.__version__ == "0.9.3"
    assert Path(cfos.__file__).is_relative_to(root / "isolated-updates")
    project = Project(folder=root / "test-project")
    images, _ = project.import_images(root / "two-wells.czi")
    assert [e.scene_id for e in images] == [0, 1]
    project.save()
    expected = [
        "df447547ba99429178b441fc2089abc94379df15e080c678f40c2c601cc858f7",
        "de5f4bbcea7c46d02b038a0de217c2953ccb08dc761d496b22d2d22f44a9506f",
    ]
    for scene, checksum in zip(Project.load(project.folder).images, expected, strict=True):
        with closing(scene.open_slide()) as slide:
            assert abs(slide.pixel_size_um - 0.45715068250381635) < 1e-10
            pixels = slide.read(0)
            assert pixels.shape == (2, 1416, 1960) and pixels.dtype == np.uint16
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == checksum
        print("Verified", scene.display_name, checksum, flush=True)
