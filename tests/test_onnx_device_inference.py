import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cfos.core import accel
from cfos.core.gpu_devices import GpuAdapter


class OnnxDeviceInferenceTests(unittest.TestCase):
    def setUp(self):
        state = patch.dict(accel._status)
        state.start()
        self.addCleanup(state.stop)
        pref = patch.object(accel, "PREF", "auto")
        pref.start()
        self.addCleanup(pref.stop)

    def test_directml_uses_dxgi_index_and_reports_only_after_validation(self):
        adapters = (GpuAdapter("integrated", 0x8086, 512, 0), GpuAdapter("discrete", 0x10de, 8192, 2))
        with patch.object(accel, "gpu_provider", return_value="DmlExecutionProvider"), \
                patch("cfos.core.gpu_devices.gpu_adapters", return_value=adapters):
            model = accel.stardist_net("model.onnx")
        self.assertEqual(model.ort_net.adapter.dml_index, 2)
        x = np.ones(2)
        with patch.object(model.ort_net, "predict", return_value=[x, x]), \
                patch.object(model.tf_model, "predict", return_value=[x, x]) as cpu:
            model.predict(x)
            model.predict(x)
        self.assertEqual(cpu.call_count, 1)
        self.assertIn("discrete", accel._status["device"])

    def test_wrong_outputs_or_later_device_failure_fall_back_for_current_and_later_tiles(self):
        x = np.ones(3, dtype=np.float32)
        for outputs in ([[np.array([np.nan]*3), x]], [[x, x], RuntimeError("GPU memory")]):
            with self.subTest():
                gpu = accel.OrtNet("model.onnx", "DmlExecutionProvider")
                cpu = Mock()
                cpu.predict.return_value = [x, x]
                net = accel._CheckedNet(gpu, cpu, "model")
                with patch.object(gpu, "predict", side_effect=outputs) as pred:
                    for _ in range(3):
                        np.testing.assert_array_equal(net.predict(x)[0], x)
                self.assertEqual(pred.call_count, len(outputs))
                self.assertEqual(cpu.predict.call_count, 3)
                self.assertTrue(accel._status["failed"])

    def test_provider_rejection_and_cpu_only_graph(self):
        ep = "DmlExecutionProvider"
        opts = SimpleNamespace()
        s = Mock()
        s.get_inputs.return_value = [SimpleNamespace(name="x")]
        ort = SimpleNamespace(SessionOptions=lambda: opts, InferenceSession=Mock(return_value=s),
                              ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0))
        def profile():
            path = Path(opts.profile_file_prefix + ".json")
            path.write_text(json.dumps([{"cat":"Node","args":{"provider":ep,"op_name":"MemcpyToHost"}},
                                        {"cat":"Node","args":{"provider":"CPUExecutionProvider","op_name":"Conv"}}]))
            return str(path)
        s.end_profiling.side_effect = profile
        with patch.object(accel, "_ort", return_value=ort):
            model = accel.OrtNet("model.onnx", ep)
            s.get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "创建失败"):
                model.predict(np.ones((1, 2)))
            s.get_providers.return_value = [ep, "CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "执行计算节点"):
                model.predict(np.ones((1, 2)))
            self.assertFalse(model._sessions)

    def test_cpu_preference_and_model_invalidation(self):
        loaded = {"": object()}
        with patch("cfos.core.cellpose4.set_pref") as set_pref:
            accel.set_pref("auto")
            set_pref.assert_called_once_with("auto")
            accel.set_pref("cpu", loaded)
            set_pref.assert_called_with("cpu")
        self.assertFalse(loaded)
        with patch.object(accel, "gpu_provider", side_effect=AssertionError):
            self.assertEqual(accel.stardist_net("model.onnx").provider, "CPUExecutionProvider")

    def test_probe_and_diagnostics_report_current_detector(self):
        cancel = Mock()
        with patch("cfos.core.cellpose4.probe_device", return_value="probe") as probe, \
                patch("cfos.core.cellpose4.status_text", return_value="status"), \
                patch("cfos.core.cellpose4.backend_text", return_value="backend"):
            self.assertEqual(accel.probe_device(cancel), "probe")
            probe.assert_called_once_with(cancel=cancel)
            self.assertEqual(accel.status_text(), "status")
            self.assertEqual(accel.backend_text(), "backend")
