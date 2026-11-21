import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from cfos.core import updater


class LegacyUpdateLaneTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / "base"
        self.updates = self.root / "updates"
        env = patch.dict(os.environ, CFOS_BASE_APP=str(self.base), CFOS_UPDATES_DIR=str(self.updates))
        env.start()
        self.addCleanup(env.stop)
        self.write_version(self.base, "0.8")

    @staticmethod
    def write_version(app, version):
        path = app / "cfos" / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")

    def make_patch(self, version, min_base, shared=False):
        files = {"app/cfos/__init__.py": f'__version__ = "{version}"\n'.encode()}
        if shared:
            files["app/_series/0.8/cfos/__init__.py"] = b'__version__ = "0.8.11"\n'
            files["app/_series/0.9/cfos/__init__.py"] = b'__version__ = "0.9.2"\n'
        info = {"version": version, "min_base": min_base,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        archive = self.root / "patch.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("patch.json", json.dumps(info))
            for name, data in files.items():
                z.writestr(name, data)
        return archive

    def check_manifest(self, version, min_base):
        manifest = {"version": version, "min_base": min_base, "full_installer": "https://example.org/v0.9.0"}
        with patch.object(updater, "fetch_manifest", return_value=(manifest, "test-url")):
            return updater.check()

    def test_online_misrouted_manifest_never_offers_new_series(self):
        for version, minimum in [("0.9.2", "0.7"), ("0.9.2", "0.9.0"), ("0.8.12", "0.9")]:
            with self.subTest(version=version, min_base=minimum):
                with self.assertRaisesRegex(ValueError, "v0.8 专用补丁"):
                    self.check_manifest(version, minimum)
        self.assertFalse(self.updates.exists())

    def test_native_manifest_and_full_installer_stay_on_legacy_lane(self):
        self.assertEqual(self.check_manifest("0.8", "0.7")["status"], "latest")
        self.assertEqual(self.check_manifest("0.8.11", "0.7")["status"], "patch")
        self.write_version(self.base, "0.7")
        update = self.check_manifest("0.8.12", "0.8")
        self.assertEqual(update["status"], "full")
        self.assertEqual(update["manifest"]["full_installer"], updater.FULL_PAGE)
        self.assertTrue(updater.FULL_PAGE.endswith("/updates/v0.8/README.md"))

    def test_manual_foreign_and_shared_zips_leave_active_install_intact(self):
        updater.install_patch(self.make_patch("0.8.11", "0.7"))
        self.assertEqual(updater.current_version(), "0.8.11")
        def snapshot():
            return {p.relative_to(self.updates): p.read_bytes() if p.is_file() else None
                    for p in self.updates.rglob("*")}
        original = snapshot()
        for minimum, shared in [("0.7", False), ("0.9", False), ("0.7", True)]:
            with self.subTest(min_base=minimum, shared=shared):
                archive = self.make_patch("0.9.2", minimum, shared)
                with self.assertRaisesRegex(ValueError, "v0.8 专用补丁"):
                    updater.read_patch_info(archive)
                with self.assertRaisesRegex(ValueError, "v0.8 专用补丁"):
                    updater.install_patch(archive)
                self.assertEqual(snapshot(), original)
                self.assertEqual(updater.current_version(), "0.8.11")

    def test_active_state_rejects_wrong_series_and_preserves_valid_shared_entry(self):
        app = self.updates / "0.9.2" / "app"
        self.write_version(app, "0.9.2")
        state = {"dir": "0.9.2", "version": "0.9.2", "min_base": "0.7"}
        pointer = self.updates / "current.json"
        pointer.write_text(json.dumps(state), encoding="utf-8")
        self.assertIsNone(updater.active_patch())
        pointer.write_text(json.dumps({**state, "version": "0.8.11"}), encoding="utf-8")
        self.assertIsNone(updater.active_patch())

        native = app / "_series" / "0.8"
        self.write_version(native, "0.8.11")
        pointer.write_text(json.dumps(state), encoding="utf-8")
        self.assertEqual(updater.active_patch()["app"], str(native))
        self.assertEqual(updater.current_version(), "0.8.11")


if __name__ == "__main__":
    unittest.main()
