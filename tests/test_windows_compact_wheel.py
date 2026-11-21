import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile


script = Path(__file__).resolve().parents[1] / "packaging" / "windows_compact_wheel.py"
spec = spec_from_file_location("windows_compact_wheel", script)
transport = module_from_spec(spec)
spec.loader.exec_module(transport)


class CompactWheelTest(unittest.TestCase):
    def test_restore_exact_zip_and_reject_corruption_atomically(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(transport, "BLOCK", 1024):
            bundle = Path(directory)
            (bundle / "wheels").mkdir()
            wheel = bundle / "wheels" / "torch-for-test.whl"
            with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                archive.writestr("A.dll", random.Random(7).randbytes(32 * 1024))
                archive.writestr("metadata.txt", b"untouched")
                archive.writestr("B.dll", random.Random(8).randbytes(32 * 1024))
                archive.comment = b"retain the original ZIP directory"
            original = wheel.read_bytes()
            entry = {"file": wheel.name, "size": len(original), "sha256": hashlib.sha256(original).hexdigest()}
            (bundle / "manifest.json").write_text(json.dumps({"wheels": [entry]}), encoding="utf-8")
            media = bundle / "media"
            transport.pack(bundle, media)
            wheel.unlink()
            transport.restore(bundle, media)
            self.assertEqual(wheel.read_bytes(), original)
            wheel.write_bytes(b"existing data")
            (media / "part-0.bin").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                transport.restore(bundle, media)
            self.assertEqual(wheel.read_bytes(), b"existing data")
            self.assertEqual(list(wheel.parent.iterdir()), [wheel])
