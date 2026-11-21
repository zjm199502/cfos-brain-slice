import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cfos.core import vsi, vsi_meta
from v084_vsi import write_ets


class VsiSceneTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "多扫描区.vsi"
        self.path.write_bytes(b"II*\0" + bytes(64))

    def frame(self, number, value, side, *, time=0, rgb=False):
        folder = vsi.companion_dir(self.path) / f"stack{number}"
        ets = folder / f"frame_t_{time}.ets"
        shape = (side, side, 3) if rgb else (side, side)
        image = np.full(shape, value, np.uint8)
        write_ets(ets, {(0, 0): image}, comp=0, ncomp=3 if rgb else 1)
        return ets

    def test_unparsed_metadata_separate_stacks_and_time_frames(self):
        large = self.frame(30, 37, 96)
        self.frame(30, 80, 96, time=1)
        small = self.frame(6, 12, 32)
        self.frame(9, 200, 128, rgb=True)
        with patch.object(vsi, "read_ets_index", side_effect=AssertionError("Discovery read a tile table")):
            scenes = vsi.list_vsi_scenes(self.path)
        self.assertEqual([(s.scene_id, s.width, s.height) for s in scenes], [(30, 96, 96), (6, 32, 32)])
        for selection, expected, path in [(None, 37, large), (6, 12, small), (30, 37, large)]:
            with patch.object(vsi, "read_ets_index", wraps=vsi.read_ets_index) as read:
                slide = vsi.VsiFile(self.path, scene_id=selection)
                try:
                    self.assertTrue(np.all(slide.read(0, 1, 1, 8, 4) == expected))
                    read.assert_called_once_with(path)
                finally:
                    slide.close()
        with self.assertRaisesRegex(vsi.VsiReadError, "stack18"):
            vsi.VsiFile(self.path, scene_id=18)
        with self.assertRaisesRegex(vsi.VsiReadError, "明场"):
            vsi.VsiFile(self.path, scene_id=9)

    def test_metadata_roles_duplicates_and_partial_parse(self):
        for num, size in [(2, 64), (3, 64), (4, 96), (5, 96), (6, 32)]:
            self.frame(num, num, size)
        def meta(num, size, role):
            return SimpleNamespace(stack_id=num, name="重复的场景名", width=size, height=size,
                                   image_type=role, slots=[], pixel_size_um=None)
        parsed = [meta(2, 64, 32), meta(3, 64, 32), meta(4, 96, 1)]
        # Role metadata still identifies stack5 as a label when its frame metadata
        # is unreadable. A separate raw monochrome ETS in stack6 remains usable.
        def partial_metadata(_tree, *, missing_external):
            missing_external.append(7)  # missing overview, not a scan to analyze
            return parsed, ["incomplete frame"]

        with patch.object(vsi_meta, "load_tree"), patch.object(vsi_meta, "parse_all_images",
                side_effect=partial_metadata), patch.object(vsi_meta, "stack_image_types",
                return_value={5: 256, 7: 1}):
            notes = []
            self.assertEqual([s.scene_id for s in vsi.list_vsi_scenes(self.path, notes=notes)], [2, 3, 6])
            self.assertEqual(len(notes), 1)
            self.assertIn("stack6", notes[0])
            with self.assertRaisesRegex(vsi.VsiReadError, "stack5"):
                vsi.VsiFile(self.path, scene_id=5)
            slide = vsi.VsiFile(self.path, scene_id=3)
            try:
                self.assertEqual(slide.image_name, "重复的场景名")
                self.assertTrue(np.all(slide.read(0, 0, 0, 3, 3) == 3))
            finally:
                slide.close()

    def test_missing_external_and_damaged_header_warn_without_losing_other_stacks(self):
        sample = Path(__file__).parent / "data/vsi/vsi-ets-test-jpg2k.vsi"
        self.path.write_bytes(sample.read_bytes())  # declares an external ETS for stack1
        self.frame(8, 23, 32)
        broken = self.frame(9, 67, 64)
        broken.write_bytes(b"incomplete")
        notes = []
        scenes = vsi.list_vsi_scenes(self.path, notes=notes)
        self.assertEqual([scene.scene_id for scene in scenes], [8])
        self.assertTrue(any("stack1/" in n and "未找到" in n for n in notes), notes)
        self.assertTrue(any("stack9" in n and "文件头" in n for n in notes), notes)
        with self.assertRaisesRegex(vsi.VsiReadError, "stack1/"):
            vsi.VsiFile(self.path, scene_id=1)

    def test_default_skips_larger_stack_with_unreadable_tile_offsets(self):
        largest = self.frame(30, 71, 96)
        self.frame(6, 18, 32)
        data = bytearray(largest.read_bytes())
        main = vsi._MAIN.unpack_from(data)
        ndim, table, count = main[3], main[7], main[8]
        for i in range(count):
            offset = table + i * 4 * (ndim + 5) + 4 * (ndim + 1)
            struct.pack_into("<Q", data, offset, len(data) + 100)
        largest.write_bytes(data)
        slide = vsi.VsiFile(self.path)
        try:
            self.assertEqual(slide.scene_id, 6)
        finally:
            slide.close()
        with self.assertRaisesRegex(vsi.VsiReadError, "stack30"):
            vsi.VsiFile(self.path, scene_id=30)
