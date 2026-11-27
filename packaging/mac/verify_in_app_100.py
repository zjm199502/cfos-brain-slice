"""Exercise native old/new updater code in separate fresh Python processes."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import unittest

app = Path(os.environ["BASE_APP"])
sys.path.insert(0, str(app))
runpy.run_path(str(app / "main.py"), run_name="qa_boot")
import cfos
from cfos.core import updater

phase, expected, feed = sys.argv[1:4]
assert cfos.__version__ == updater.current_version() == expected, (cfos.__version__, updater.current_version())
assert updater.base_version() == "0.9.0"
updates = Path.home() / "Library/Application Support/cfos脑片分析/更新"
assert updater.updates_dir() == updates
assert "CFOS_UPDATES_DIR" not in os.environ
os.environ["CFOS_UPDATE_URL"] = feed

if phase == "install":
    result = updater.check()
    assert result["status"] == "patch", result
    wanted, digest = sys.argv[4:6]
    assert result["manifest"]["version"] == wanted
    assert result["manifest"]["patch"]["sha256"] == digest
    patch = updater.download_patch(result)
    size = patch.stat().st_size
    with patch.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
    assert updater.install_patch(patch)["version"] == wanted
    print(f"Actual {expected} updater verified and installed {wanted}, {size} bytes.")
elif phase in ("snapshot", "restarted"):
    from cfos.core import cellpose_setup
    runtime, weights = cellpose_setup.runtime_python(), cellpose_setup.weights_path()
    def identify(path):
        st = path.stat()
        with path.open("rb") as stream:
            return [str(path), st.st_ino, st.st_mtime_ns, st.st_size, hashlib.file_digest(stream, "sha256").hexdigest()]
    identity = [identify(p) for p in (runtime, weights)]
    witness = Path(os.environ["RUNNER_TEMP"])/"offline-identity.json"
    if phase == "snapshot":
        witness.write_text(json.dumps(identity))
    else:
        assert json.loads(witness.read_text()) == identity
        assert updater.check()["status"] == "latest"
        state = json.loads((updates/"current.json").read_text())
        assert state["version"] == expected and state["failed_starts"] == 0, state
        assert Path(cfos.__file__).is_relative_to(updates)
        from cfos.core import cellpose4
        try:
            print("Preserved runtime:", cellpose4.probe_device(), flush=True)
        finally:
            cellpose4.close()
        test_dir = str(Path(os.environ["GITHUB_WORKSPACE"])/"tests")
        for filename in ("test_coloc_cache_coordinates.py", "test_rotated_pan.py", "test_cell_ring_navigation.py"):
            suite = unittest.defaultTestLoader.discover(test_dir, pattern=filename)
            assert suite.countTestCases() and unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        print("Two restarts, default state, patch version, local model, runtime and next check passed.", flush=True)
