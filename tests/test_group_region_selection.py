from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from cfos.core import region_sets as RS
from cfos.core.behavior import BehaviorParams, correlate
from cfos.core.project import Project
from cfos.core.stats import StatsParams, per_animal, run_group_stats


class GroupRegionSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = Project(self.root, channel_names=["Fos"], export_region_set=RS.CUSTOM,
                               export_region_ids=[295])
        atlas = pd.DataFrame([(382, "CA1", "Field CA1", "海马 CA1 区"),
                              (295, "BLA", "BLA", "基底外侧杏仁核"),
                              (463, "CA3", "Field CA3", "海马 CA3 区")],
                             columns=["id", "acronym", "name", "zh"])
        info = patch.object(RS, "info", return_value=atlas)
        info.start()
        self.addCleanup(info.stop)
        rows = []
        self.scores = {}
        for i in range(6):
            image = f"image{i}"
            self.project.set_sample(f"{image}/S01", animal=image, group="A" if i < 3 else "B")
            self.scores[image] = i
            for region in ("CA1", "BLA"):
                rows.append([image, "S01", region, region, 7, "左右合计", 1., i + 1])
        self.regions = pd.DataFrame(rows, columns=["图片", "脑片", "脑区缩写", "脑区名称", "层级",
                                                   "半脑", "面积 (mm²)", "Fos+"])
        self.params = StatsParams(metric="Fos+ 密度 (个/mm²)", region_set=RS.CUSTOM, region_ids=[382, 463])

    def test_statistics_workbook_includes_only_selected_covered_regions(self):
        with patch("cfos.core.stats.bar_plot"), patch("cfos.core.stats.volcano_plot"), \
             patch("cfos.core.heatmap.heatmap_figure"), patch("cfos.core.heatmap.default_ap_levels"):
            run_group_stats(self.project, self.root, self.params, self.regions)
        workbook = pd.read_excel(self.root / "组间统计.xlsx", sheet_name=None)
        for name in ("组间比较", "按组汇总", "按动物×脑区"):
            self.assertEqual(set(workbook[name]["脑区缩写"]), {"CA1"}, name)
        metadata = dict(workbook["统计方法"].itertuples(index=False, name=None))
        self.assertIn("2 个", metadata["脑区范围"])
        self.assertIn("CA3", metadata["自选脑区（Allen ID 463）"])
        self.assertIn("CA1", metadata["自选脑区（Allen ID 382）"])

    def test_uncovered_or_stale_selection_reports_reason(self):
        for ids, message in (([463], "未覆盖"), ([382, 1000000], "找不到"), ([], "至少勾选")):
            with self.subTest(region_ids=ids), self.assertRaisesRegex(ValueError, message):
                self.params.region_ids = ids
                run_group_stats(self.project, self.root, self.params, self.regions)

    def test_behavior_reuses_selected_regions(self):
        behavior = BehaviorParams(metric=self.params.metric, region_set=RS.CUSTOM, region_ids=[382])
        animal = per_animal(self.regions, self.project.samples)
        results, wide = correlate(animal, self.scores, behavior)
        self.assertEqual(results["脑区缩写"].tolist(), ["CA1"])
        self.assertIn("CA1", wide.columns)
        self.assertNotIn("BLA", wide.columns)

    def test_no_animal_meets_area_threshold(self):
        self.params.min_area_mm2 = 2.
        with self.assertRaisesRegex(ValueError, "最小面积"):
            run_group_stats(self.project, self.root, self.params, self.regions)
        self.assertFalse((self.root / "组间统计.xlsx").exists())
