import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from cfos.core import cellpose_setup as setup
from cfos.core import updater


class Response(io.BytesIO):
    def __init__(self, data, offset, size):
        super().__init__(data)
        self.status = 206
        self.headers = {"Content-Range": f"bytes {offset}-{size - 1}/{size}"}


class SetupTests(unittest.TestCase):
    def offline_media(self, stack):
        base = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        bundle = base / "app" / "resources" / "cellpose4"
        (bundle / "wheels").mkdir(parents=True)
        wheel = bundle / "wheels" / "example-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("example.py", "VALUE = 1")
        weights = bundle / "cpsam_v2"
        weights.write_bytes(b"model")
        entry = {"file": wheel.name, "size": wheel.stat().st_size,
                 "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
        manifest = {"format": 1, "platform": setup.platform_tag(), "cellpose": setup.CELLPOSE_VERSION,
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}", "wheels": [entry]}
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        stack.enter_context(patch.dict(os.environ, {"CFOS_BASE_APP": str(bundle.parents[1]),
                                                     "CFOS_COMPONENTS_DIR": str(base / "components")}))
        stack.enter_context(patch.object(setup, "WEIGHTS_BYTES", weights.stat().st_size))
        stack.enter_context(patch.object(setup, "WEIGHTS_SHA256", hashlib.sha256(b"model").hexdigest()))
        stack.enter_context(patch.object(setup, "_weights", side_effect=AssertionError("online weight lookup")))
        return bundle, wheel, manifest

    def test_offline_media_uses_only_local_wheels_and_activates_after_inference(self):
        with ExitStack() as stack:
            bundle, wheel, _ = self.offline_media(stack)
            calls = []
            def run(args, cancel, *, offline):
                self.assertTrue(offline)
                calls.append(args)
                if "venv" in args:
                    python = setup._python(args[-1])
                    python.parent.mkdir(parents=True)
                    python.touch()
                if "--check-env" in args:
                    self.assertEqual(args[-1], bundle / "cpsam_v2")
                    self.assertFalse(setup.runtime_python().is_file())
            stack.enter_context(patch.object(setup, "_run", side_effect=run))
            setup.install(require_offline=True)
            self.assertTrue(setup.offline_ready())
            pip_cmd = next(args for args in calls if "install" in args)
            self.assertIn(wheel, pip_cmd)
            self.assertIn("--no-index", pip_cmd)
            self.assertIn("--no-deps", pip_cmd)
            with patch.object(setup, "_run") as run_again:
                setup.install(require_offline=True)
                self.assertEqual(run_again.call_count, 1)
                self.assertIn("--check-env", run_again.call_args.args[0])

    def test_offline_damage_and_wrong_media_do_not_fall_back_or_replace_active_runtime(self):
        with ExitStack() as stack:
            bundle, wheel, manifest = self.offline_media(stack)
            run = stack.enter_context(patch.object(setup, "_run"))
            root = setup._root()
            root.mkdir(parents=True)
            (root / "active.txt").write_text("runtime-previous", encoding="utf-8")
            existing = setup.runtime_python()
            wheel.write_bytes(b"x" * wheel.stat().st_size)
            with self.assertRaisesRegex(ValueError, "校验失败"):
                setup.install()
            manifest["platform"] = "unsupported"
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "不匹配"):
                setup.install()
            (bundle / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "离线清单"):
                setup.install()
            bundle.rename(bundle.with_name("missing"))
            with self.assertRaisesRegex(FileNotFoundError, "完整离线安装包"):
                setup.install(require_offline=True)
            run.assert_not_called()
            self.assertEqual(setup.runtime_python(), existing)

    def test_fresh_offline_install_skips_automatic_update_lookup(self):
        with ExitStack() as stack:
            self.offline_media(stack)
            state = {}
            stack.enter_context(patch.object(updater, "_load_state", return_value=state))
            self.assertFalse(updater.auto_check_enabled())
            state["auto_check"] = True
            self.assertTrue(updater.auto_check_enabled())

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
