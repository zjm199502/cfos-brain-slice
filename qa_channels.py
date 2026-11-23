"""Check the real base launcher, shared bridge and platform feed using an isolated profile."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
base = Path(os.environ["CFOS_BASE_APP"]).resolve()
platform = {"darwin": "macos", "win32": "windows"}[sys.platform]
other = "windows" if platform == "macos" else "macos"
feed = here / "stage/updates/v0.9"
mode = sys.argv[1]
if len(sys.argv) == 2:
    env = dict(os.environ, CFOS_UPDATES_DIR=tempfile.mkdtemp(prefix="fosfolio-channel-"))
    env.pop("CFOS_UPDATE_URL", None)
    for phase in ("seed", "bridge", "platform"):
        subprocess.run([sys.executable, __file__, mode, phase], env=env, check=True)
        banner = subprocess.check_output([sys.executable, str(base / "main.py"), "--version"], env=env,
                                         encoding="utf8").strip()
        assert ("0.9.6" if phase == "seed" else "0.9.7") in banner, banner
        print(banner, flush=True)
    print("PASS", platform, mode, flush=True)
    sys.exit()

sys.path.insert(0, str(base))
runpy.run_path(str(base / "main.py"))
import cfos
from cfos.core import updater
phase = sys.argv[2]
if phase == "seed":
    assert cfos.__version__ == "0.9.0"
    archive = here / "prev.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "d7e61ffd2f9a4ccdb2871d8b0b298d0ad98e01fb1738cbabb5b51356dbea7e39"
    updater.install_patch(archive)
else:
    folder = feed if phase == "bridge" else feed/platform
    suffix = "/updates/v0.9/latest.json" if phase == "bridge" else f"/updates/v0.9/{platform}/latest.json"
    assert all(url.endswith(suffix) for url in updater.manifest_urls())
    if mode == "local":
        os.environ["CFOS_UPDATE_URL"] = (folder / "latest.json").as_uri()
    result = updater.check()
    manifest = json.loads((folder / "latest.json").read_bytes())
    assert result["manifest"] == manifest, (result["url"], result["manifest"]["version"])
    assert result["status"] == ("patch" if phase == "bridge" else "latest")
    zip_path = updater.download_patch(result)
    if phase == "bridge":
        assert cfos.__version__ == "0.9.6"
        assert updater.install_patch(zip_path)["version"] == "0.9.7"
    else:
        assert cfos.__version__ == "0.9.7" and manifest["platform"] == platform
        assert updater.read_patch_info(zip_path)["platform"] == platform
        wrong = next((feed / other).glob("*.zip"))
        before = updater._state_file().read_bytes()
        try:
            updater.install_patch(wrong)
        except ValueError as exc:
            assert other in str(exc)
        else:
            raise AssertionError("Wrong-platform patch accepted")
        assert before == updater._state_file().read_bytes()
