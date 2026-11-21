import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cfos.core import components, updater


class ComponentPaths(unittest.TestCase):
    def test_releases_keep_the_shared_component_manifest(self):
        with patch.dict(os.environ, {"CFOS_COMPONENTS_URL": "", "CFOS_UPDATE_URL": ""}):
            self.assertEqual([u.rsplit("/updates/", 1)[0] + "/updates/components.json"
                              for u in updater.MANIFEST_URLS], components.manifest_urls())

    def test_shared_download_and_explicit_override(self):
        with TemporaryDirectory() as tmp:
            shared = Path(tmp) / "updates"
            shared.mkdir()
            (shared / "components.json").write_text(json.dumps({"source": "shared"}), encoding="utf8")
            channel = shared / ("v" + updater.SERIES) / "latest.json"
            with patch.dict(os.environ, {"CFOS_UPDATE_URL": channel.as_uri(), "CFOS_COMPONENTS_URL": ""}):
                self.assertEqual(components.fetch_manifest(), ({"source": "shared"}, shared.as_uri()))
            with patch.dict(os.environ, {"CFOS_UPDATE_URL": "http://localhost/test/latest.json", "CFOS_COMPONENTS_URL": ""}):
                self.assertEqual(components.manifest_urls(), ["http://localhost/test/components.json"])
            with patch.dict(os.environ, {"CFOS_COMPONENTS_URL": "http://localhost/explicit/"}):
                self.assertEqual(components.manifest_urls(), ["http://localhost/explicit/components.json"])
