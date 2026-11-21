import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
import zipfile


PACKER = Path(__file__).resolve().parents[1] / "packaging" / "make_series_update.py"


def make_channel(parent, version, base):
    folder = parent / version
    folder.mkdir()
    data = {
        "app/cfos/__init__.py": f'__version__ = "{version}"\n'.encode(),
        "app/cfos/backend.py": f'BUILD = "{version}"\n'.encode(),
    }
    metadata = {"version": version, "min_base": base, "date": "2026-11-21",
                "notes": "# Version\n\nRelease notes"}
    patch = folder / "patch.zip"
    with zipfile.ZipFile(patch, "w") as archive:
        digests = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()}
        archive.writestr("patch.json", json.dumps({**metadata, "files": digests}))
        for name, content in data.items():
            archive.writestr(name, content)
    descriptor = {"file": patch.name, "size": patch.stat().st_size,
                  "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
    manifest = {**metadata, "patch": descriptor, "full_installer": ""}
    (folder / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class SeriesUpdateTests(unittest.TestCase):
    def test_legacy_manifest_and_import_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            shared = root / "shared"
            build = runpy.run_path(str(PACKER))["build"]
            build(make_channel(root, "0.8.11", "0.7"), make_channel(root, "0.9.2", "0.9.0"), shared)
            manifest = json.loads((shared / "latest.json").read_text(encoding="utf-8"))
            with zipfile.ZipFile(shared / manifest["patch"]["file"]) as archive:
                self.assertEqual(json.loads(archive.read("patch.json"))["version"], "0.9.2")
                archive.extractall(shared)
            patch_app = shared / "app"
            selector = (patch_app / "cfos" / "__init__.py").read_text(encoding="utf-8")
            # The old desktop updater reads the first version assignment without importing cfos.
            assignment = next(line for line in selector.splitlines() if line.startswith("__version__"))
            self.assertEqual(assignment.split("=", 1)[1].strip().strip("\"'"), manifest["version"])
            base_app = root / "base"
            (base_app / "cfos").mkdir(parents=True)
            probe = """
import json, os, sys
sys.path[:0] = [sys.argv[1], os.environ['CFOS_BASE_APP']]
import cfos
from cfos import backend
print(json.dumps([cfos.__version__, cfos.__file__, backend.BUILD, backend.__file__, os.environ['CFOS_PATCH_APP']]))
"""
            for base, native in [("0.7", "0.8.11"), ("0.8", "0.8.11"), ("0.9.0", "0.9.2")]:
                with self.subTest(base=base):
                    (base_app / "cfos" / "__init__.py").write_text(f'__version__ = "{base}"\n', encoding="utf-8")
                    output = subprocess.check_output(
                        [sys.executable, "-I", "-B", "-c", probe, str(patch_app)],
                        env={**os.environ, "CFOS_BASE_APP": str(base_app)}, text=True)
                    package_version, package_file, module_version, module_file, active = json.loads(output)
                    expected = patch_app / "_series" / native.rsplit(".", 1)[0]
                    self.assertEqual((package_version, module_version), (native, native))
                    self.assertEqual(Path(package_file), expected / "cfos" / "__init__.py")
                    self.assertEqual(Path(module_file), expected / "cfos" / "backend.py")
                    self.assertEqual(Path(active), expected)
