import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cfos import __version__
from cfos.core import updater


class MajorUpdateChannelTest(unittest.TestCase):
    def test_current_release_uses_its_own_platform_feed(self):
        self.assertEqual(__version__, "1.0.0")
        if updater.PLATFORM:
            for url in updater.MANIFEST_URLS:
                self.assertIn(f"/updates/v1.0/{updater.PLATFORM}/latest.json", url)

    def test_major_upgrade_requests_full_installer(self):
        for current, offered, minimum, expected in [
            ("0.9.22", "1.0.0", "1.0.0", "full"),
            ("1.0.0", "1.0.0", "1.0.0", "latest"),
            ("1.0.0", "1.0.1", "1.0.0", "patch"),
            ("1.0.0", "1.1.0", "1.0.0", "full"),
        ]:
            with self.subTest(current=current, offered=offered):
                manifest = {"version": offered, "min_base": minimum}
                with (patch.object(updater, "fetch_manifest", return_value=(manifest, "test")),
                      patch.object(updater, "current_version", return_value=current),
                      patch.object(updater, "base_version", return_value=current)):
                    self.assertEqual(updater.check()["status"], expected)

    def test_new_state_is_separate_from_legacy_state(self):
        env = dict(os.environ)
        env.pop("CFOS_UPDATES_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(updater.updates_dir().parts[-2:], ("更新", "v1.0"))
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with patch.dict(os.environ, {"CFOS_UPDATES_DIR": str(state)}):
                self.assertEqual(updater.updates_dir(), state)
