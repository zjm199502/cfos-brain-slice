from contextlib import closing
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from cfos.core import region_sets as RS
from cfos.core import regions
from cfos.core.export import build_tables, export_excel
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section


class CustomRegionExport(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        atlas_dir = self.root / "atlas"
        atlas_dir.mkdir()
        st = pd.DataFrame([
            (1, 997, "root", "/997/"), (2, 382, "CA1", "/997/382/"),
            (3, 10, "leaf1", "/997/382/10/"), (4, 20, "leaf2", "/997/382/20/"),
            (5, 30, "other", "/997/30/"), (6, 40, "absent", "/997/40/"),
        ], columns=["index", "id", "acronym", "path"])
        st["name"] = st["acronym"] + " name"
        st.to_csv(atlas_dir / "structures.csv", index=False)
        RS.info.cache_clear()
        self.addCleanup(RS.info.cache_clear)
        for mock in (patch.object(RS, "resources_dir", return_value=self.root),
                     patch.object(regions, "load_atlas", return_value=SimpleNamespace(structures=st))):
            mock.start()
            self.addCleanup(mock.stop)
        sec = Section(1, (0, 0, 16, 16), [], "S1")
        sec.registration = {"plane": [1, 2, 3, 1, 0, 0, 0, 1, 0]}
        im = ImageEntry("input.tif", sections=[sec])
        self.project = Project(self.root, images=[im], channel_names=["Fos"], export_region_set=RS.CUSTOM)
        self.project.coloc.require_nucleus = False
        self.project.export_region_ids = [382]
        cells = pd.DataFrame(dict(section=["S1"] * 3, merged_id=[0, 1, 2], channel=[0] * 3,
                                  combo=["0"] * 3, x=[1., 2., 3.], y=[1., 2., 3.]))
        self.project.save_cells(im, cells)
        areas = pd.DataFrame(dict(section=["S1"] * 4, region_index=[3, 3, 4, 5],
                                  hemisphere=[1, 2, 1, 2], area_mm2=[1., 2., 3., 4.],
                                  n_px=[10., 20., 30., 40.], sum_ch0=[10., 100., 120., 1000.]))
        def assign(frame, *args, **kwargs):
            if frame.empty:
                return frame.assign(region_index=pd.Series(dtype=int), hemisphere=pd.Series(dtype=int)), areas.copy()
            return frame.assign(region_index=[3, 3, 5], hemisphere=[1, 2, 2]), areas.copy()
        self.project.export_intensity = True
        for mock in (patch.object(ImageEntry, "open_slide", return_value=SimpleNamespace(pixel_size_um=1., close=lambda: None)),
                     patch.object(regions, "assign_regions", side_effect=assign)):
            mock.start()
            self.addCleanup(mock.stop)

    def workbook(self):
        path = export_excel(self.project, self.root / "export.xlsx", masks=False, qc=False, quality=False)
        return pd.read_excel(path, sheet_name=None)

    def test_parent_aggregates_children_while_only_explicit_choices_appear(self):
        sheets = self.workbook()
        t = sheets["按脑区×脑片"]
        self.assertEqual(t["Allen ID"].tolist(), [382])
        self.assertEqual(t.loc[0, "Fos+"], 2)
        self.assertEqual(t.loc[0, "面积 (mm²)"], 6)
        self.assertAlmostEqual(t.loc[0, "Fos 平均荧光强度"], 3.8)
        self.assertEqual(sheets["按脑片汇总"].loc[0, "Fos+"], 3)
        self.project.export_region_ids = [382, 20, 20]
        self.project.export_hemisphere = "仅左半脑"
        sheets = self.workbook()
        t = sheets["按脑区×脑片"].set_index("Allen ID")
        self.assertEqual(t.index.tolist(), [382, 20])
        self.assertEqual(t["Fos+"].tolist(), [1, 0])
        self.assertEqual(t["面积 (mm²)"].tolist(), [4, 3])
        self.assertEqual(t["半脑"].tolist(), ["左", "左"])
        params = sheets["分析参数"].set_index("参数")["值"]
        self.assertIn("CA1", params["自选脑区（Allen ID 382）"])
        self.assertIn("leaf2", params["自选脑区（Allen ID 20）"])

    def test_absent_region_preserves_headers_and_selection_in_excel(self):
        self.project.export_region_ids = [40]
        sheets = self.workbook()
        self.assertTrue(sheets["按脑区×脑片"].empty)
        self.assertEqual(list(sheets["按脑区×脑片"].columns[:4]), ["图片", "脑片", "组别", "动物"])
        self.assertIn("Allen ID", sheets["按脑区×脑片"])
        params = sheets["分析参数"].set_index("参数")["值"]
        self.assertIn("没有覆盖", params["自选脑区导出结果"])
        self.assertIn("absent", params["自选脑区（Allen ID 40）"])
        self.project.images[0].sections[0].registration = {}
        self.assertTrue(self.workbook()["按脑区×脑片"].empty)

    def test_region_rows_carry_each_images_sample_and_remain_filterable(self):
        first = self.project.images[0]
        labels = [(first.key, "对照组", "001"), ("second", "=实验组", "002"), ("third", None, None)]
        for key, group, animal in labels:
            if key != first.key:
                image = deepcopy(first)
                image.storage_key = key
                self.project.images.append(image)
                self.project.save_cells(image, self.project.load_cells(first))
            if group:
                self.project.set_sample(f"{key}/S1", animal=animal, group=group)
        self.project.export_region_ids = [382, 20]
        self.project.export_hemisphere = "左右分别导出并附合计"
        path = export_excel(self.project, self.root / "groups.xlsx", masks=False, qc=False, quality=False)
        with closing(load_workbook(path, data_only=True)) as workbook:
            sheet = workbook["按脑区×脑片"]
            header, *rows = sheet.values
            self.assertEqual(header[:4], ("图片", "脑片", "组别", "动物"))
            self.assertEqual(sheet.auto_filter.ref, sheet.dimensions)
            samples = {r[:4] for r in rows}
            self.assertEqual(samples, {(key, "S1", group, animal) for key, group, animal in labels})
            sides = header.index("半脑")
            self.assertEqual({row[sides] for row in rows}, {"左", "右", "左右合计"})
            region = header.index("Allen ID")
            identities = {(r[0], r[1], r[region], r[sides]) for r in rows}
            self.assertEqual(len(identities), len(rows))

    def test_empty_or_invalid_selection_is_not_all_regions(self):
        for ids in ([], [99999]):
            self.project.export_region_ids = ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "Allen"):
                self.workbook()
            full = build_tables(self.project, all_regions=True)["按脑区×脑片"]
            self.assertSetEqual(set(full["脑区缩写"]), {"root", "CA1", "leaf1", "leaf2", "other"})

    def test_zero_cells_across_all_channels_keep_covered_regions(self):
        im = self.project.images[0]
        empty = self.project.load_cells(im).iloc[:0].copy()
        empty.attrs["processed_sections"] = ["S1"]
        self.project.save_cells(im, empty)
        self.project.channel_names = ["DAPI", "EGFP", "Fos"]
        self.project.coloc.require_nucleus = True
        self.project.export_region_ids = [382, 20, 40]
        sheets = self.workbook()
        regions = sheets["按脑区×脑片"].set_index("Allen ID")
        self.assertEqual(regions.index.tolist(), [382, 20])
        self.assertEqual(regions["面积 (mm²)"].tolist(), [6, 3])
        for table in (regions, sheets["按脑片汇总"]):
            for col in ("DAPI+", "EGFP+", "Fos+", "DAPI+Fos 共标", "DAPI+EGFP+Fos 共标"):
                self.assertTrue((table[col] == 0).all(), col)
            self.assertTrue(table["Fos+ 占 DAPI+ 的比例 %"].isna().all())
