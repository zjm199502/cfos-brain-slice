from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cfos.core import cellpose4, cellpose_worker


class CellposeRuntimeTests(unittest.TestCase):
    def tearDown(self):
        cellpose4.close()
        cellpose4.set_pref("auto")

    def test_mismatched_weights_fail_before_inference(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp)/"cpsam_v2"
            file.write_bytes(b"bad")
            with patch.object(cellpose4, "_paths", return_value=(Path(sys.executable),file)):
                with self.assertRaisesRegex(RuntimeError, "校验失败"):
                    cellpose4.model_identity()

    @unittest.skipIf(os.name == "nt", "POSIX venv interpreters use symlinks")
    def test_runtime_keeps_venv_python_symlink(self):
        with TemporaryDirectory() as tmp:
            link = Path(tmp)/"bin"/"python"
            link.parent.mkdir()
            link.symlink_to(sys.executable)
            weights = Path(tmp)/"cpsam_v2"
            weights.touch()
            with patch.dict(os.environ, CFOS_CELLPOSE_PYTHON=str(link), CFOS_CELLPOSE_WEIGHTS=str(weights)):
                self.assertEqual(cellpose4._paths()[0],link)

    def test_auto_and_manual_devices_and_precision(self):
        with TemporaryDirectory() as tmp:
            weights = Path(tmp)/"cpsam_v2"
            weights.touch()
            for cuda, mps, pref, expected in [(True,False,"auto","cuda"), (False,True,"auto","mps"),
                                                (False,False,"auto","cpu"), (True,False,"cpu","cpu")]:
                with self.subTest(expected=expected):
                    torch = Mock()
                    torch.cuda.is_available.return_value=cuda
                    torch.cuda.is_bf16_supported.return_value=True
                    torch.cuda.get_device_name.return_value="GPU"
                    torch.backends.mps.is_available.return_value=mps
                    torch.device.side_effect=lambda name,*args:SimpleNamespace(type=name)
                    models=Mock()
                    models.CellposeModel.return_value.net.parameters.return_value=iter([
                        SimpleNamespace(device=SimpleNamespace(type=expected))])
                    modules={"torch":torch,"cellpose":SimpleNamespace(models=models)}
                    with patch.dict(sys.modules,modules),patch.object(cellpose_worker,"runtime_versions"):
                        cellpose_worker.load_model(weights,pref)
                    kwargs=models.CellposeModel.call_args.kwargs
                    self.assertEqual(kwargs["device"].type,expected)
                    self.assertEqual(kwargs["use_bfloat16"],expected=="cuda")
                    self.assertEqual(kwargs["pretrained_model"],str(weights))

    def test_reuses_process_returns_instance_ids_and_can_cancel(self):
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake.py"
            script.write_text('''import json, sys
from pathlib import Path
import numpy as np
for line in sys.stdin.buffer:
    msg = json.loads(line)
    root = Path(msg["folder"])
    a = np.load(root/"image.npy", allow_pickle=False)
    np.save(root/"labels.npy", (a > msg["threshold"]).astype(np.int32))
    print(json.dumps({"device":"test device"}), flush=True)
''', encoding="utf8")
            paths = Path(sys.executable), script
            with patch.object(cellpose4,"_WORKER",script), patch.object(cellpose4,"_paths",return_value=paths), \
                    patch.object(cellpose4,"model_identity"):
                a=np.array([[0.,3.],[1.,5.]])
                np.testing.assert_array_equal(cellpose4.predict_labels(a, 1.5), [[0,1],[0,1]])
                first=cellpose4._worker
                np.testing.assert_array_equal(cellpose4.predict_labels(a, 3.5), [[0,0],[0,1]])
                self.assertIs(cellpose4._worker,first)
                self.assertIn("test device",cellpose4.backend_text())
                with self.assertRaises(InterruptedError):
                    cellpose4.predict_labels(a, cancel=lambda:True)
                cellpose4.set_pref("cpu")
                first.proc.wait(timeout=5)
                self.assertIsNone(cellpose4._worker)
