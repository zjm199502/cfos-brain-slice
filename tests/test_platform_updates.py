import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from cfos.core import updater


class PlatformUpdates(unittest.TestCase):
    def test_single_platform_then_default_release(self):
        builder = Path(__file__).resolve().parents[1] / "packaging" / "make_platform_updates.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v0.9"
            notes = Path(tmp) / "notes.md"
            notes.write_text("平台分流测试", encoding="utf8")
            for platform in ("windows", "macos"):
                folder = root / platform
                folder.mkdir(parents=True)
                (folder / "更新公告.md").write_text("# 公告\n", encoding="utf8")
                (folder / "latest.json").write_text(json.dumps(dict(platform=platform, version="0.9.7",
                    min_base="0.9.0", full_installer="https://example.org/base")), encoding="utf8")
            cmd = [sys.executable, str(builder), str(root), "--notes", str(notes)]
            old = (root / "windows" / "latest.json").read_bytes()
            subprocess.run(cmd + ["--platform", "macos"], check=True)
            self.assertEqual(old, (root / "windows" / "latest.json").read_bytes())
            subprocess.run(cmd, check=True)
            for platform, version in (("windows", "0.9.8"), ("macos", "0.9.9")):
                directory = root / platform
                manifest = json.loads((directory / "latest.json").read_bytes())
                self.assertEqual((manifest["platform"], manifest["version"]), (platform, version))
                archive = directory / manifest["patch"]["file"]
                with zipfile.ZipFile(archive) as z:
                    self.assertIn(f'__version__ = "{version}"'.encode(), z.read("app/cfos/__init__.py"))
                with patch.object(updater, "PLATFORM", platform):
                    self.assertEqual(updater.read_patch_info(archive)["platform"], platform)
                with patch.object(updater, "PLATFORM", "macos" if platform == "windows" else "windows"):
                    with self.assertRaisesRegex(ValueError, "平台"):
                        updater.install_patch(archive)

    def test_misrouted_manifest_is_rejected(self):
        with patch.object(updater, "PLATFORM", "macos"), patch.object(updater, "_get",
                return_value=b'{"platform":"windows","version":"0.9.99"}'):
            with self.assertRaises(ConnectionError):
                updater.check()
