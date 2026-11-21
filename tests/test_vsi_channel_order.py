from contextlib import closing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cfos.core import vsi, vsi_meta
from cfos.core.project import ImageEntry, Project
from v084_vsi import write_ets


class VsiChannelOrderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.path = self.folder / "cells.vsi"
        self.path.write_bytes(b"synthetic")
        self.metas = []
        for num, size, names in [(2, 64, ["DAPI", "FITC", "Cy5"]), (5, 96, ["DAPI", "Cy5", "FITC"]),
                                 (8, 32, ["DAPI", "Cy5", "Cy3"]), (9, 32, ["DAPI", "Cy5", "Cy5"])]:
            channels = [SimpleNamespace(index=i, name=name, alt_name=None, emission_nm=None) for i, name in enumerate(names)]
            slots = [SimpleNamespace(kind="Z"), SimpleNamespace(kind="C", channels=channels)]
            self.metas.append(SimpleNamespace(stack_id=num, width=size, height=size, name=f"scene{num}",
                                              image_type=32, slots=slots, pixel_size_um=None))
            path = vsi.companion_dir(self.path) / f"stack{num}/frame_t_0.ets"
            planes = {(0, i): np.full((size, size), 10 * num + i, np.uint16) for i in range(3)}
            write_ets(path, planes, comp=0)
        for mock in (patch.object(vsi_meta, "load_tree"),
                     patch.object(vsi_meta, "parse_all_images", return_value=(self.metas, []))):
            mock.start()
            self.addCleanup(mock.stop)

    def assert_channels(self, entry, labels, values):
        with closing(entry.open_slide()) as slide:
            self.assertEqual([c.name for c in slide.channels], labels)
            self.assertEqual(slide.read(0, (0, 0, 4, 4))[:, 0, 0].tolist(), values)
            self.assertEqual(slide.read(0, (0, 0, 4, 4), [2, 0])[:, 0, 0].tolist(), [values[2], values[0]])
            lazy = slide._arr(0)
            self.assertEqual([int(lazy[i][2, 2]) for i in range(3)], values)

    def test_same_file_reordered_or_incompatible_channels(self):
        project = Project(self.folder / "new")
        entries, notes = project.import_images(self.path)
        self.assertEqual([e.scene_id for e in entries], [2, 5])
        self.assertTrue(all(any(f"stack{n}" in m and "通道" in m for m in notes) for n in (8, 9)), notes)
        labels = ["DAPI", "FITC", "Cy5"]
        self.assert_channels(entries[0], labels, [20, 21, 22])
        self.assert_channels(entries[1], labels, [50, 52, 51])
        project.save()
        loaded = Project.load(project.folder)
        loaded.import_images(self.path)
        self.assertEqual(len(loaded.images), 2)
        self.assert_channels(loaded.images[1], labels, [50, 52, 51])

    def test_existing_scene_sets_order_independent_of_dye_roles(self):
        for previous_id in (None, 5):
            with self.subTest(previous_id=previous_id):
                # Existing records predate channel order, with/without a scene selector.
                old = ImageEntry.from_dict({"path": str(self.path), "scene_id": previous_id, "storage_key": "old-results"})
                project = Project(self.folder / str(previous_id), images=[old], channel_names=["Nuc", "Fos", "EGFP"])
                entries, _ = project.import_images(self.path)
                self.assertEqual([e.scene_id for e in entries], [2, 5])
                self.assertIs(entries[1], old)
                self.assertEqual(old.key, "old-results")
                self.assertEqual(project.channel_names, ["Nuc", "Fos", "EGFP"])
                labels = ["DAPI", "Cy5", "FITC"]
                self.assert_channels(old, labels, [50, 51, 52])
                self.assert_channels(entries[0], labels, [20, 22, 21])
                project.save()
                loaded = Project.load(project.folder)
                self.assert_channels(loaded.images[0], labels, [50, 51, 52])
