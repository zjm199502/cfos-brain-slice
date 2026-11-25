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
            assert ("0.9.10" if phase=="seed" else "0.9.11") in banner,banner
            print(banner.strip(),flush=True)
    sys.exit()
sys.path.insert(0,str(base))
runpy.run_path(str(base/"main.py"))
import cfos
from cfos.core import updater
phase=sys.argv[2]
if phase=="seed":
    prev=Path(os.environ.get("EDGE_PREVIOUS",Path(os.environ["RUNNER_TEMP"])/"prev.zip"))
    digest={"macos":"b737dee07dc8ae6ecefc84e8691f6e8c187461ee0bb42ff7a78ec1435e593015","windows":"9f4c7e785e4b9dfb8a6b8e0e97d2daae02bb2a94ec2c1d27e4e29c17ec182cd4"}[platform]
    assert hashlib.sha256(prev.read_bytes()).hexdigest()==digest
    assert updater.install_patch(prev)["version"]=="0.9.10"
elif phase=="upgrade":
    assert cfos.__version__=="0.9.10"
    assert all(u.endswith("/v0.9/"+platform+"/latest.json") for u in updater.manifest_urls())
    if mode=="stage":os.environ["CFOS_UPDATE_URL"]=(candidate/"latest.json").resolve().as_uri()
    result=updater.check()
    assert result["status"]=="patch" and result["manifest"]==json.loads((candidate/"latest.json").read_bytes()),result["status"]
    archive=updater.download_patch(result)
    assert updater.read_patch_info(archive)["platform"]==platform
    assert updater.install_patch(archive)["version"]=="0.9.11"
else:
    assert cfos.__version__=="0.9.11" and Path(cfos.__file__).is_relative_to(updater.updates_dir())
    suite=unittest.defaultTestLoader.discover(str(candidate),pattern="test_simplified_detection.py")
    assert suite.countTestCases()>=2 and unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    from cfos.core.fosfolio_adapter import preview_hippocampal_alignment
    check_gui=runpy.run_path(str(here/"cells_gui_qa.py"))["check_gui"]
    project_folder=Path(os.environ["CFOS_UPDATES_DIR"]).parent/"gui-project"
    project_folder.mkdir()
    check_gui(project_folder)
    print("PASS",platform,mode,flush=True)


