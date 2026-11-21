import ctypes as ct
import subprocess
import unittest
from unittest.mock import Mock, patch

from cfos.core import gpu_devices as gpu


class AdapterTests(unittest.TestCase):
    def setUp(self):
        gpu.gpu_adapters.cache_clear()
        self.addCleanup(gpu.gpu_adapters.cache_clear)

    def test_dxgi_preserves_hardware_ordinals(self):
        library = Mock()
        def create_factory(iid, target):
            ct.cast(target, ct.POINTER(ct.c_void_p))[0] = 10
            return 0
        library.CreateDXGIFactory1.side_effect = create_factory
        devices = [("Integrated", 0x8086, 2**27, 0),
                   ("Software", 0x1414, 0, 2), ("Discrete", 0x10DE, 2**33, 0)]
        released = []
        def method(obj, slot, *signature):
            if slot == 12:
                def enumerate_one(_factory, index, target):
                    if index == len(devices):
                        return -1
                    ct.cast(target, ct.POINTER(ct.c_void_p))[0] = index + 20
                    return 0
                return enumerate_one
            if slot == 10:
                def describe(adapter, target):
                    desc = ct.cast(target, ct.POINTER(gpu._AdapterDesc1)).contents
                    desc.name, desc.vendor, desc.video_memory, desc.flags = devices[adapter.value - 20]
                    return 0
                return describe
            self.assertEqual(slot, 2)
            return lambda obj: released.append(obj.value)
        with patch.object(ct, "WinDLL", create=True, return_value=library), patch.object(gpu, "_com_call", method):
            adapters = gpu._dxgi_adapters()
        self.assertEqual(adapters, (gpu.GpuAdapter("Integrated", 0x8086, 2**27, 0),
                                    gpu.GpuAdapter("Discrete", 0x10DE, 2**33, 2)))
        self.assertEqual(released, [20, 21, 22, 10])

    def test_display_names_are_cached_and_have_no_inferred_dml_index(self):
        with patch.object(gpu.sys, "platform", "win32"), patch.object(gpu, "_dxgi_adapters", return_value=()), \
                patch.object(gpu, "_output", return_value="Integrated\nDiscrete\n") as probe:
            self.assertEqual(gpu.gpu_names(), ("Integrated", "Discrete"))
            self.assertEqual([a.dml_index for a in gpu.gpu_adapters()], [None, None])
            probe.assert_called_once()

    def test_mac_and_linux(self):
        cases = [
            ("darwin", '{"SPDisplaysDataType":[{"sppci_model":"Apple M2 Pro"}]}', ("Apple M2 Pro",)),
            ("darwin", "incomplete", ()),
            ("linux", "0000:00:02.0 VGA compatible controller [0300]: Intel UHD Graphics [8086:1234]\n"
                      "0000:01:00.0 3D controller [0302]: NVIDIA A5000 [10de:2231]",
             ("Intel UHD Graphics [8086:1234]", "NVIDIA A5000 [10de:2231]")),
        ]
        for platform, output, expected in cases:
            gpu.gpu_adapters.cache_clear()
            with self.subTest(platform=platform), patch.object(gpu.sys, "platform", platform), \
                    patch.object(gpu, "_output", return_value=output):
                self.assertEqual(gpu.gpu_names(), expected)

    def test_display_probe_timeout(self):
        with patch.object(gpu.sys, "platform", "darwin"), patch.object(
                subprocess, "run", side_effect=subprocess.TimeoutExpired("system_profiler", 6)) as run:
            self.assertEqual(gpu.gpu_names(), ())
            self.assertEqual(gpu.gpu_names(), ())
            run.assert_called_once()
