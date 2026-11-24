"""Install previous release, upgrade and restart using the original runtime."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

here=Path(__file__).resolve().parent
base=Path(os.environ["CFOS_BASE_APP"])
platform={"darwin":"macos","win32":"windows"}[sys.platform]
candidate=Path(os.environ.get("EDGE_FEED",here/"candidate"))
mode=sys.argv[1]
if len(sys.argv)==2:
    with tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as td:
        env=dict(os.environ,CFOS_UPDATES_DIR=str(Path(td)/"updates"),PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(Path(td)/"mpl"))
        if platform=="windows":
            env.update(LOCALAPPDATA=td,APPDATA=td,QT_QPA_PLATFORM="windows")
        for key in ("CFOS_UPDATE_URL","CFOS_PATCH_APP","PYTHONPATH"): env.pop(key,None)
        for phase in ("seed","upgrade","verify"):
            subprocess.run([sys.executable,"-B",__file__,mode,phase],env=env,check=True)
            banner=subprocess.check_output([sys.executable,str(base/"main.py"),"--version"],env=env,text=True)
            assert ("0.9.8" if phase=="seed" else "0.9.9") in banner,banner
            print(banner.strip(),flush=True)
    sys.exit()
sys.path.insert(0,str(base))
runpy.run_path(str(base/"main.py"))
import cfos
from cfos.core import updater
phase=sys.argv[2]
if phase=="seed":
    prev=Path(os.environ.get("EDGE_PREVIOUS",Path(os.environ["RUNNER_TEMP"])/"prev.zip"))
    digest={"macos":"2da64cccefbc4ddfe5745855bf05e30cb4b14aee367fbab0fc13e556436091cb","windows":"0823106085e39214ddac9e53d590609ca83c47b9eb9accdb31239796bc7ace93"}[platform]
    assert hashlib.sha256(prev.read_bytes()).hexdigest()==digest
    assert updater.install_patch(prev)["version"]=="0.9.8"
elif phase=="upgrade":
    assert cfos.__version__=="0.9.8"
    assert all(u.endswith("/v0.9/"+platform+"/latest.json") for u in updater.manifest_urls())
    if mode=="stage":os.environ["CFOS_UPDATE_URL"]=(candidate/"latest.json").resolve().as_uri()
    result=updater.check()
    assert result["status"]=="patch" and result["manifest"]==json.loads((candidate/"latest.json").read_bytes()),result["status"]
    archive=updater.download_patch(result)
    assert updater.read_patch_info(archive)["platform"]==platform
    assert updater.install_patch(archive)["version"]=="0.9.9"
else:
    assert cfos.__version__=="0.9.9" and Path(cfos.__file__).is_relative_to(updater.updates_dir())
    suite=unittest.defaultTestLoader.discover(str(candidate),pattern="test_fine_edges.py")
    assert suite.countTestCases()>=4 and unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    from cfos.core.fosfolio_adapter import preview_hippocampal_alignment
    check_gui=runpy.run_path(str(here/"edge_gui_qa.py"))["check_gui"]
    with tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as td:check_gui(Path(td))
    print("PASS",platform,mode,flush=True)

