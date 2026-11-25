import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from cfos.core import export, pipeline, quality
from cfos.core.detection import ChannelParams
from cfos.core.project import ImageEntry, Project
from cfos.core.scope import selected_images, selected_sections, scoped_project
from cfos.core.sections import Section


class ProcessingScopeTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        source = self.root / "sections.tif"
        tifffile.imwrite(source, np.ones((64, 256), np.uint8))
        secs = [Section(i, (i * 64, 0, 60, 60), [], f"S{i}") for i in range(3)]
        self.im = ImageEntry(str(source), secs)
        self.other = ImageEntry(str(self.root / "unopened.tif"), [Section(1, (0,0,1,1), [], "S1")])
        self.p = Project(self.root, images=[self.im, self.other], channel_names=["DAPI"])
        self.selection = {self.im.key: ("S1",)}
        df = pd.DataFrame([dict(section=s.name, channel=0, x=64.*i+8, y=8., area_um2=20., mean_ch0=150.,
                                contour=[], cell_id=i, merged_id=i, combo="0") for i,s in enumerate(secs)])
        df.attrs["processed_sections"] = ["S0", "S1", "S2"]
        self.p.save_cells(self.im, df)
        self.df = df

    def test_export_metadata_scope_and_caches(self):
        k = self.im.key
        self.im.rois = [{"name": "选中", "polygon": [(65,1),(120,1),(120,42),(65,42)]},
                        {"name": "未选", "polygon": [(0,1),(40,1),(40,42),(0,42)]}]
        self.p.accuracy = [{"图片": k, "脑片":"S0"}, {"图片": k, "脑片":"S1"}]
        self.p.samples = {f"{k}/S0": {"animal":"a"}, f"{k}/S1":{"animal":"b"}}
        self.p.behavior = {"a":1, "b":2}
        self.im.field_reviews = [{"脑片":"S0"}, {"脑片":"S1"}]
        self.im.figure_rois = [dict(name=name, x=x, y=0, w=20, h=20)
                               for name, x in (("框0", 5), ("跨入框1", 58), ("框外", 220))]
        view = scoped_project(self.p, self.selection)
        self.assertEqual(view.load_cells(self.im).section.tolist(), ["S1"])
        self.assertIsNone(view.load_cells(self.other))
        self.assertEqual(view.sample_rows()[0]["动物"], "b")
        self.assertEqual(view.behavior, {"b":2})
        self.assertEqual(view.accuracy, [{"图片":k, "脑片":"S1"}])
        self.assertEqual(view.images[0].field_reviews, [{"脑片":"S1"}])
        self.assertEqual([r["name"] for r in view.images[0].figure_rois], ["跨入框1"])
        self.assertEqual(len(self.im.figure_rois), 3)
        self.assertEqual(scoped_project(self.p, {k: None}).images[0].figure_rois, self.im.figure_rois)
        self.assertNotEqual(export.companion_dir(view, "report.xlsx", "质控图"),
                            export.companion_dir(self.p, "report.xlsx", "质控图"))
        with self.assertRaises(RuntimeError):
            view.save()
        tables = export.build_tables(view, for_excel=True)
        self.assertEqual(list(tables["按脑片汇总"]["脑片"]), ["S1"])
        self.assertEqual(list(tables["按自定义区域"]["区域"]), ["选中"])
        summary = quality._summary_path(self.p)
        summary.parent.mkdir(exist_ok=True)
        summary.write_text(json.dumps({"unselected/S1":{"why":"keep"}}), encoding="utf-8")
        with patch.object(quality, "check_image", return_value={"S1":{}}):
            quality.quality_table(view)
        self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["unselected/S1"], {"why":"keep"})

    def test_subset_coloc_preserves_only_selected_legacy_cells_without_duplication(self):
        self.im.sections[2].enabled = False
        for _ in range(2):
            with patch.object(self.p, "load_part", wraps=self.p.load_part) as load:
                self.assertEqual(pipeline.refresh_coloc(self.p, selection=self.selection), 1)
                load.assert_not_called()
            df = self.p.load_cells(self.im)
            self.assertCountEqual(df.section, ["S0","S1","S2"])
            self.assertEqual(df.attrs["processed_sections"], ["S0","S1","S2"])
            self.assertEqual(df.attrs["carried_forward_sections"], [])
            originals = df.set_index("section").loc[["S0","S2"]]
            self.assertEqual(originals["merged_id"].tolist(), [0,2])

    def test_selected_raw_recompute_preserves_other_raw_and_completed_zero(self):
        self.p.detection.channels[0] = ChannelParams(min_intensity=1000)
        cells = self.df[self.df.section != "S2"].copy()
        cells.attrs["section_devices"] = {"S0": {0: "kept-device"}, "S2": {0: "empty-device"}}
        self.p.save_cells(self.im, cells)
        for section in self.im.sections:
            raw = self.df[self.df.section == section.name].copy()
            self.p.save_part(self.im, section, section.name, raw)
        paths = [self.p.part_path(self.im, s.name) for s in self.im.sections]
        bytes_before = [p.read_bytes() for p in paths]
        self.assertEqual(pipeline.refresh_coloc(self.p, selection=self.selection), 1)
        result = self.p.load_cells(self.im)
        self.assertEqual(result.section.tolist(), ["S0", "S1"])
        self.assertEqual(result.merged_id.tolist(), [0, 1])
        self.assertEqual(result.attrs["processed_sections"], ["S0", "S1", "S2"])
        self.assertEqual(result.attrs["section_devices"], cells.attrs["section_devices"])
        self.assertEqual([p.read_bytes() for p in paths], bytes_before)
        cell_file = self.p.cells_path(self.im)
        before = cell_file.read_bytes()
        for empty in ({}, {self.im.key: ()}, {self.im.key: ("absent",)}):
            self.assertEqual(pipeline.refresh_coloc(self.p, selection=empty), 0)
            self.assertEqual(scoped_project(self.p, empty).images, [])
        self.assertEqual(cell_file.read_bytes(), before)

    def test_selected_batch_skips_import_and_resplit_even_on_redo(self):
        expected = [self.im.sections[1]]
        with (patch.object(self.p, "import_images", side_effect=AssertionError("must not import siblings")),
              patch.object(pipeline, "find_sections", side_effect=AssertionError("must not re-split full image")),
              patch.object(pipeline, "detect_image") as detect,
              patch("cfos.core.autoreg.run", return_value={}) as register,
              patch.object(quality, "check_image", return_value={}) as qc):
            rows = pipeline.run_batch(self.p, [], selection=self.selection, redo=True)
        self.assertEqual(list(detect.call_args.kwargs["section_names"]), ["S1"])
        self.assertEqual(register.call_args.kwargs["only"], {self.im.key:{"S1"}})
        self.assertEqual(list(qc.call_args.kwargs["section_names"]), ["S1"])
        self.assertEqual([r["脑片"] for r in rows], ["S1"])
        self.assertEqual(selected_sections(self.im,self.selection), expected)
        self.assertEqual(selected_images(self.p,{}), [])
        self.assertEqual(pipeline.run_batch(self.p,[],selection={}), [])
        self.im.sections[0].enabled = False
        with (patch.object(self.p, "save", side_effect=AssertionError("empty batch wrote project")),
              patch.object(self.im, "open_slide", side_effect=AssertionError("empty batch opened pixels"))):
            self.assertEqual(pipeline.run_batch(self.p, [], selection={self.im.key: ("S0", "absent")}), [])
