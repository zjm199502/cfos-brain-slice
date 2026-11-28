"""Native release checks. Run each phase with Python from the downloaded .app."""
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import tempfile


def report(message):
    print(message, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            print(f"- {message}", file=summary)


def verify_app(app):
    import numpy as np
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication
    from cfos import app as desktop
    from cfos.core.gpu_devices import gpu_names
    from cfos.core.selftest import MODULES
    from cfos.core.sections import find_sections
    from cfos.core.slide_io import open_slide
    from importlib import import_module

    for module, _ in MODULES:
        import_module(module)
    resources = app / "resources"
    assert (resources / "atlas/allen_ccfv3_25um.npz").is_file()
    from cfos.core import deep_refine
    nissl = deep_refine.nissl_volume()   # deep CCFv3 refinement falls back to a weaker texture without it
    assert nissl is not None and nissl.shape == (528, 320, 456), "Allen CCFv3 Nissl volume missing"
    slide = open_slide(resources / "demo/示例脑片.ome.tif")
    try:
        assert find_sections(slide), "Demo contains no recognized sections"
    finally:
        slide.close()
    if (app / "cfos/core/giraff.py").is_file():
        from cfos.core import atlas_choice, giraff
        for name in (atlas_choice.CCF, atlas_choice.STAM):
            atlas = giraff.load_atlas(name)
            assert atlas.n_regions > 0 and atlas.annotation.shape == atlas.hemispheres.shape
            if name == atlas_choice.STAM:
                # STAM's histological reference is 32 µm; its annotation is 40 µm.
                assert atlas.reference.shape == (438, 282, 357)
                assert atlas.annotation.shape == (350, 225, 285)
                np.testing.assert_allclose(np.array(atlas.reference.shape) * .032,
                                           np.array(atlas.annotation.shape) * .040, atol=.040)
            else:
                assert atlas.reference.shape == atlas.annotation.shape == (528, 320, 456)
        with tempfile.TemporaryDirectory() as trial:
            root = Path(trial)
            plane = atlas.reference[len(atlas.reference) // 2]
            giraff._write_plane(root / "plane.nii", plane.astype(np.uint8), .032)
            subprocess.run([giraff.NIFTYREG, "-ref", "plane.nii", "-flo", "plane.nii",
                            "-rigOnly", "-omp", "1", "-aff", "rigid.txt", "-res", "result.nii"],
                           cwd=root, check=True, timeout=60)
            affine = np.loadtxt(root / "rigid.txt")
            assert affine.shape == (4, 4) and np.isfinite(affine).all()
        report("Opened demo and both native atlases; bundled NiftyReg produced a finite rigid transform.")
    else:
        from cfos.core import deepslice_onnx
        image = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
        coords = deepslice_onnx.predict([image], resources / "deepslice")
        assert coords.shape == (1, 9) and np.isfinite(coords).all()
        report("Demo image opened; both DeepSlice ONNX models returned finite coordinates.")
    report(f"macOS reports display adapters: {gpu_names()}")

    after_start = desktop._after_start
    seen = []

    def inspect(window):
        after_start(window)
        seen.append(window.isVisible() and "FosFolio" in window.windowTitle())
        report(f"Main window: {window.windowTitle()}")
        QTimer.singleShot(250, QApplication.quit)

    desktop._after_start = inspect
    sys.argv = [str(app / "main.py"), "--no-update-check"]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as exc:
        if exc.code:
            raise
    assert seen == [True], "GUI did not show the FosFolio main window"


def verify_cellpose(offline=False):
    from cfos.core import cellpose4, cellpose_setup

    if offline:
        launcher = Path(os.environ["CFOS_BASE_APP"]).parents[1] / "MacOS/cfos"
        subprocess.run([launcher, "--install-cellpose-offline"], check=True)
    else:
        print(cellpose_setup.install(report=lambda message: print(message, flush=True)))
    cellpose4.set_pref("cpu")
    cpu = cellpose4.probe_device()
    assert "CPU" in cpu, cpu
    cellpose4.close()
    report(f"Native Cellpose inference with forced CPU: {cpu}")
    cellpose4.set_pref("auto")
    device = cellpose4.model_identity()["preferred_device"]
    if device == "mps":
        label = cellpose4.probe_device()
        assert "PyTorch MPS" in label, label
        report(f"Native Cellpose inference with MPS: {label}")
    else:
        assert device == "cpu", device
        report("PyTorch reports MPS unavailable on this runner. Hardware MPS inference was not exercised.")
    cellpose4.close()


if __name__ == "__main__":
    bundle, phase = Path(sys.argv[1]).resolve(), sys.argv[2]
    assert sys.platform == "darwin" and platform.machine() == "arm64"
    assert Path(sys.executable).resolve().is_relative_to(bundle)
    app = bundle / "Contents/Resources/app"
    os.environ["CFOS_BASE_APP"] = str(app)
    sys.path.insert(0, str(app))
    if phase == "app":
        verify_app(app)
    elif phase == "cellpose":
        verify_cellpose()
    elif phase == "cellpose-offline":
        # Run both the CLI installer and inference with network access denied.
        with tempfile.TemporaryDirectory(prefix="fosfolio-offline-") as testing:
            env = os.environ.copy()
            for key in ("CFOS_CELLPOSE_PYTHON", "CFOS_CELLPOSE_WEIGHTS"):
                env.pop(key, None)
            env["CFOS_COMPONENTS_DIR"] = str(Path(testing) / "components")
            env["CFOS_UPDATES_DIR"] = str(Path(testing) / "updates")
            subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
                 sys.executable, __file__, bundle, "cellpose-offline-isolated"], check=True, env=env
            )
    elif phase == "cellpose-offline-isolated":
        verify_cellpose(offline=True)
    else:
        raise ValueError(phase)
