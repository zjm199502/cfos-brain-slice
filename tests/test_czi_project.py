from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cfos.core import accuracy, pipeline, quality, regions, relocate
from cfos.core.export import params_table
from cfos.core.project import ImageEntry, Project
from cfos.core.registration import Registration
from cfos.core.sections import Section
from test_czi_reader import write_czi


class CziProjectTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.path = self.root / "source" / "sample.CZI"
        self.path.parent.mkdir()
        self.project = Project(self.root / "project")

    def write_scenes(self, numbers):
        planes = [(s, c, 0, 43, 92, np.full((32, 48), 10 + 3 * s + c, dtype=np.uint16))
                  for s in numbers for c in range(2)]
        write_czi(self.path, planes)

    def assert_channels(self, entry):
        with closing(entry.open_slide()) as slide:
            self.assertEqual(slide.scene_id, entry.scene_id)
            px = slide.read(0, (10, 10, 3, 4), [1, 0])
            self.assertEqual(px.shape, (2, 4, 3))
            np.testing.assert_array_equal(px[:, 0, 0], [11 + 3 * entry.scene_id, 10 + 3 * entry.scene_id])

    def test_single_scene_and_file_discovery(self):
        self.write_scenes([0])
        (self.path.parent / "~$sample.CZI").touch()
        (self.path.parent / "sample.CZI.txt").touch()
        self.assertEqual(pipeline.list_images(self.path.parent), [self.path])
        image, _ = self.project.import_image(self.path)
        self.assertEqual(image.key, self.path.stem)
        self.assertEqual(image.source_id, (str(self.path), 0))
        self.assertIn(image.scene_name, image.display_name)
        self.assert_channels(image)

    def test_multiscene_idempotent_import_relocation_and_provenance(self):
        self.write_scenes([0, 3])
        images, _ = self.project.import_images(self.path)
        self.assertEqual([im.scene_id for im in images], [0, 3])
        self.assertEqual(len({im.key for im in images}), 2)
        again, _ = self.project.import_images(self.path)
        self.assertIs(self.project.import_image(self.path)[0], images[0])
        self.assertTrue(all(original is repeat for original, repeat in zip(images, again)))
        for im in images:
            self.assert_channels(im)
            im.sections = [Section(1, (0, 0, 20, 20), [], "A")]
            self.project.save_cells(im, pd.DataFrame({"value": [im.scene_id + 70]}))
        keys = [im.key for im in images]
        self.project.save()
        moved = self.root / "moved"
        self.path.parent.rename(moved)
        reopened = Project.load(self.project.folder)
        recovered = relocate.relocate(reopened, moved)
        self.assertEqual(len(recovered["fixed"]), 2)
        reopened = Project.load(self.project.folder)
        self.assertEqual([im.key for im in reopened.images], keys)
        params = params_table(reopened).set_index("参数")["值"]
        for im in reopened.images:
            self.assert_channels(im)
            self.assertEqual(reopened.load_cells(im).value.tolist(), [im.scene_id + 70])
            self.assertIn(f"S{im.scene_id}", params[f"CZI 场景（{im.key}）"])
            result = dict(n_manual=0, n_auto=0, tp=0, fn=0, fp=0, precision=0., recall=0., f1=0., manual_xy=[])
            record = accuracy.record(reopened, im, "A", 0, (0, 0, 8, 8), result, "test")
            self.assertEqual((record["CZI 场景"], record["CZI S"]), (im.scene_name, im.scene_id))
            self.assertNotIn("VSI stack", record)

    def test_reordered_instrument_channels_rejected_without_changing_user_roles(self):
        def write_order(path, names):
            channels = "".join(f'<Channel Id="Channel:{i}" Name="{name}"/>' for i, name in enumerate(names))
            metadata = (f"<ImageDocument><Metadata><Information><Image><SizeC>3</SizeC><Dimensions>"
                        f"<Channels>{channels}</Channels></Dimensions></Image></Information>"
                        '<Scaling><Items><Distance Id="X"><Value>5e-7</Value></Distance>'
                        '<Distance Id="Y"><Value>5e-7</Value></Distance></Items></Scaling></Metadata></ImageDocument>')
            planes = [(0, c, 0, 0, 0, np.full((8, 8), c, np.uint16)) for c in range(len(names))]
            write_czi(path, planes, metadata=metadata)
        write_order(self.path, ["DAPI", "FITC", "Cy5"])
        original, _ = self.project.import_image(self.path)
        self.assertEqual(original.czi_channels, ["DAPI", "FITC", "Cy5"])
        self.project.channel_names = ["细胞核", "EGFP", "Fos"]
        self.project.save()
        reopened = Project.load(self.project.folder)
        other = self.path.with_name("another.czi")
        write_order(other, ["Cy5", "FITC", "DAPI"])
        with self.assertRaisesRegex(ValueError, "顺序"):
            reopened.import_images(other)
        self.assertEqual(len(reopened.images), 1)
        self.assertEqual(reopened.channel_names, ["细胞核", "EGFP", "Fos"])
        write_order(other, [" dapi", "FITC", "CY5"])
        self.assertEqual(len(reopened.import_images(other)[0]), 1)
        write_order(self.path, ["FITC", "DAPI", "Cy5"])
        with self.assertRaisesRegex(ValueError, "顺序"):
            reopened.images[0].open_slide()

    def test_first_time_point_warning_and_export_without_rereading_source(self):
        write_czi(self.path, [(0, 0, 0, 0, 0, np.full((8, 8), t, np.uint16), t) for t in (7, 11)])
        image, notes = self.project.import_image(self.path)
        self.assertTrue(any("时间点" in text for text in notes), notes)
        with closing(image.open_slide()) as slide:
            self.assertEqual(int(slide.read(0, (0, 0, 1, 1)).item()), 7)
        self.project.save()
        reopened = Project.load(self.project.folder)
        saved = reopened.images[0]
        self.assertEqual((saved.czi_time_index, saved.czi_time_count), (7, 2))
        with patch.object(ImageEntry, "open_slide", side_effect=AssertionError("Excel 参数表重新打开了 CZI")):
            row = params_table(reopened).set_index("参数").at[f"CZI 时间点（{image.key}）", "值"]
        self.assertIn("T=7", row)
        self.assertIn("共 2 个", row)
        with (patch.object(pipeline, "find_sections", return_value=[Section(1, (0, 0, 8, 8), [], "A")]),
              patch.object(pipeline, "detect_image"), patch.object(pipeline.Q, "check_image", return_value={})):
            checklist = pipeline.run_batch(reopened, [self.path], do_register=False)
        self.assertTrue(any(r.get("类型") == "导入" and "时间点" in r["需要检查"] for r in checklist))

    def test_batch_registration_and_caches_are_per_scene(self):
        self.write_scenes([0, 3])
        visited, signatures, intensity, qc = [], [], [], []
        def detect(project, im, slide, **kwargs):
            self.assert_channels(im)
            visited.append(im.scene_id)
            signatures.append(slide.scene_source_signature())
            sec = im.sections[0]
            reg = Registration()
            qc.append(quality._signature(project, im, sec))
            intensity.append(regions._intensity_signature(slide, sec, reg, None, []))
        def register(project, targets, nuc, *, only, slides, **kwargs):
            im, = targets
            self.assertEqual(set(slides), {im.source_id})
            self.assertEqual(slides[im.source_id].scene_id, im.scene_id)
            self.assertEqual(only, {im.key: {"A"}})
            return {im.source_id: {"A": Registration(warnings=[f"S{im.scene_id}"])}}
        with (patch.object(pipeline, "detect_image", side_effect=detect),
              patch.object(pipeline, "find_sections", side_effect=lambda *a, **kw: [Section(1, (0, 0, 20, 20), [], "A")]),
              patch.object(pipeline.Q, "check_image", return_value={}),
              patch("cfos.core.autoreg.run", side_effect=register)):
            checklist = pipeline.run_batch(self.project, [self.path, self.path], do_register=True)
        self.assertEqual(visited, [0, 3])
        self.assertNotEqual(signatures[0], signatures[1])
        self.assertNotEqual(intensity[0], intensity[1])
        self.assertNotEqual(qc[0], qc[1])
        rows = {r["图片"]: r["需要检查"] for r in checklist if r["脑片"] == "A"}
        self.assertEqual(len(rows), 2)
        for im in self.project.images:
            self.assertIn(f"S{im.scene_id}", rows[im.display_name])

    def test_cell_detection_cache_reuses_only_its_own_scene(self):
        self.write_scenes([0, 3])
        first, second = self.project.import_images(self.path)[0]
        self.project.unmix = {"enabled": False}
        for im in (first, second):
            im.sections = [Section(1, (0, 0, 16, 16), [], "A")]
        with (patch.object(pipeline.det.cellpose4, "model_identity", return_value="test model"),
              patch.object(pipeline.det, "detect_section", side_effect=lambda *a, **k: pd.DataFrame()) as inference,
              patch.object(pipeline, "_merge_frames"),
              patch.object(self.project, "save_part", wraps=self.project.save_part) as save):
            self.assertEqual(pipeline.detect_image(self.project, first)["done"], 1)
            self.assertEqual(pipeline.detect_image(self.project, second)["done"], 1)
            self.assertNotEqual(save.call_args_list[0].args[2], save.call_args_list[1].args[2])
            self.assertEqual(pipeline.detect_image(self.project, first)["reused"], 1)
            self.assertEqual(inference.call_count, 2)


if __name__ == "__main__":
    unittest.main()
