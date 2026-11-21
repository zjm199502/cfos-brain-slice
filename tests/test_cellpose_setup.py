import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cfos.core import cellpose_setup as setup


class Response(io.BytesIO):
    def __init__(self, data, offset, size):
        super().__init__(data)
        self.status = 206
        self.headers = {"Content-Range": f"bytes {offset}-{size - 1}/{size}"}


class SetupTests(unittest.TestCase):
    def test_resumed_weights_are_verified_before_activation(self):
        data = b"example model weights"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpsam_v2"
            path.with_suffix(".part").write_bytes(data[:5])
            reply = Response(data[5:], 5, len(data))
            with patch.object(setup, "weights_path", return_value=path), \
                    patch.object(setup, "WEIGHTS_BYTES", len(data)), \
                    patch.object(setup, "WEIGHTS_SHA256", hashlib.sha256(data).hexdigest()), \
                    patch.object(setup.urllib.request, "urlopen", return_value=reply) as fetch:
                setup._weights(lambda _: None, None)
            self.assertEqual(fetch.call_args.args[0].get_header("Range"), "bytes=5-")
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_suffix(".part").exists())

    def test_interrupted_repair_keeps_the_old_active_runtime(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "_root", return_value=Path(tmp)):
            root = Path(tmp)
            (root / "active.txt").write_text("runtime-previous", encoding="utf-8")
            previous = setup.runtime_python()
            previous.parent.mkdir(parents=True)
            previous.touch()
            with patch.object(setup, "_smoke", side_effect=RuntimeError("broken")), \
                    patch.object(setup, "_run", side_effect=InterruptedError("cancelled")):
                with self.assertRaises(InterruptedError):
                    setup._install_runtime(root, lambda _: None, None)
            self.assertEqual(setup.runtime_python(), previous)
            self.assertEqual(list(root.glob("runtime-*")), [root / "runtime-previous"])
