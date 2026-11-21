import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cfos.core import pipeline, relocate
from cfos.core.project import ImageEntry, Project
from cfos.core.registration import Registration
from cfos.core.sections import Section
from v084_vsi import write_ets


class VsiProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "input" / "sample.vsi"
        self.path.parent.mkdir()
        self.path.write_bytes(b"II*\0" + bytes(64))
        self.project = Project(self.root / "project")
        self.values = {2: 17, 7: 119}
        for number, side in [(2, 64), (7, 96)]:
            image = np.full((side, side), self.values[number], dtype=np.uint16)
            companion = self.path.parent / "_sample_" / f"stack{number}" / "frame_t_0.ets"
            write_ets(companion, {(0, 0): image}, comp=0)

    def assert_correct_pixels(self, entry):
        with closing(entry.open_slide()) as slide:
            tile = slide.read(0, (8, 7, 6, 5))
        self.assertEqual(tile.shape, (1, 5, 6))
        self.assertTrue(np.all(tile == self.values[entry.scene_id]))

    def test_import_repeat_save_reopen_and_relocate(self):
        images, _ = self.project.import_images(self.path)
        self.assertEqual([im.scene_id for im in images], [2, 7])
        self.assertEqual(len({im.key for im in images}), 2)
        self.assertEqual(len({im.source_id for im in images}), 2)
        for im in images:
            self.assert_correct_pixels(im)
            im.sections = [Section(1, (0, 0, 20, 20), [], "A", enabled=im.scene_id == 7)]
            self.project.save_cells(im, pd.DataFrame({"value": [self.values[im.scene_id]]}))
        keys = [im.key for im in images]
        repeated, _ = self.project.import_images(self.path)
        self.assertEqual(len(self.project.images), 2)
        self.assertTrue(all(a is b for a, b in zip(images, repeated)))
        self.project.save()
        reopened = Project.load(self.project.folder)
        moved = self.root / "new location"
        self.path.parent.rename(moved)
        self.assertEqual(len(relocate.missing(reopened)), 2)
        result = relocate.relocate(reopened, moved)
        self.assertEqual((result["missing"], result["ambiguous"]), ([], []))
        recovered = Project.load(self.project.folder)
        self.assertEqual([im.key for im in recovered.images], keys)
        for im in recovered.images:
            self.assertEqual(Path(im.path), moved / self.path.name)
            self.assert_correct_pixels(im)
            self.assertEqual(im.sections[0].enabled, im.scene_id == 7)
            self.assertEqual(recovered.load_cells(im)["value"].tolist(), [self.values[im.scene_id]])

    def test_readding_old_project_preserves_largest_scene_results(self):
        legacy = ImageEntry(str(self.path))
        legacy.sections = [Section(1, (9, 8, 7, 6), [], "reviewed", enabled=False)]
        self.project.images.append(legacy)
        self.project.save_cells(legacy, pd.DataFrame({"existing": [42]}))
        old_key = legacy.key
        result_file = self.project.cells_path(legacy)
        previous_bytes = result_file.read_bytes()
        self.project.save()
        file = self.project.folder / "cfos_project.json"
        record = json.loads(file.read_text(encoding="utf-8"))
        for field in ("scene_id", "scene_name"):
            record["images"][0].pop(field)
        file.write_text(json.dumps(record), encoding="utf-8")

        loaded = Project.load(self.project.folder)
        with closing(loaded.images[0].open_slide()) as previous:
            self.assertEqual(int(previous.read(0, (0, 0, 1, 1)).item()), self.values[7])
        small, large = loaded.import_images(self.path)[0]
        self.assertEqual([small.scene_id, large.scene_id], [2, 7])
        self.assertIs(large, loaded.images[0])
        self.assertEqual(large.key, old_key)
        self.assertEqual(large.sections[0], legacy.sections[0])
        self.assertEqual(loaded.cells_path(large), result_file)
        self.assertEqual(result_file.read_bytes(), previous_bytes)
        self.assertEqual(loaded.load_cells(large)["existing"].tolist(), [42])
        self.assertEqual(small.sections, [])
        self.assertIsNone(loaded.load_cells(small))
        loaded.save()
        self.assertEqual(Project.load(loaded.folder).images[0].scene_id, 7)

    def test_batch_visits_each_scene_once_with_its_pixels_and_registration_key(self):
        visited = []

        def detect(project, image, slide, **_kwargs):
            value = int(slide.read(0, (0, 0, 1, 1)).item())
            visited.append((image.scene_id, value))
            project.save_cells(image, pd.DataFrame({"value": [value]}))

        def register(project, targets, _nuc, *, only, slides, **_kwargs):
            image, = targets
            self.assertEqual(only, {image.key: {"A"}})
            self.assertEqual(set(slides), {image.source_id})
            self.assertEqual(image.source_id, (str(self.path), image.scene_id))
            value = int(slides[image.source_id].read(0, (0, 0, 1, 1)).item())
            self.assertEqual(value, self.values[image.scene_id])
            return {image.source_id: {"A": Registration(warnings=[f"scene {image.scene_id}"])}}

        with (patch.object(pipeline, "detect_image", side_effect=detect),
              patch.object(pipeline, "find_sections", side_effect=lambda *a, **kw: [Section(1, (0, 0, 8, 8), [], "A")]),
              patch.object(pipeline.Q, "check_image", return_value={}),
              patch("cfos.core.autoreg.run", side_effect=register) as registration):
            files = pipeline.list_images(self.path.parent)
            checklist = pipeline.run_batch(self.project, files + files, do_register=True)
        self.assertEqual(visited, list(self.values.items()))
        self.assertEqual(registration.call_count, 2)
        rows = {row["图片"]: row for row in checklist if row["脑片"] == "A"}
        self.assertEqual(len(rows), 2)
        for im in Project.load(self.project.folder).images:
            self.assertEqual(im.error, "")
            self.assertIn(f"scene {im.scene_id}", rows[im.display_name]["需要检查"])
            self.assertEqual(self.project.load_cells(im)["value"].tolist(), [self.values[im.scene_id]])
