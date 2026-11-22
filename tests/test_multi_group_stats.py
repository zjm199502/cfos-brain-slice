from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats as st

from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section
from cfos.core import stats


def make_project(folder, groups):
    project = Project(folder, channel_names=["Fos"])
    rows = []
    for g, group in enumerate(groups):
        for i in range(5):
            image = ImageEntry(f"image_{g}_{i}.tif", sections=[Section(1, (0, 0, 5, 5), [], "S01")])
            project.images.append(image)
            project.set_sample(f"{image.key}/S01", animal=f"M{g}_{i}", group=group)
            for region in ("A", "B"):
                if g >= 2 and region == "B":
                    continue
                rows.append({"图片": image.key, "脑片": "S01", "脑区缩写": region, "脑区名称": region,
                             "层级": 4, "半脑": "左右合计", "面积 (mm²)": 1.,
                             "Fos+": i + 3 * g if region == "A" else i + g})
    return project, pd.DataFrame(rows)


class MultiGroupStats(unittest.TestCase):
    metric = "Fos+ 密度 (个/mm²)"

    def test_auto_group_count_and_methods(self):
        for num in (2, 3, 4):
            for method in ("welch", "mannwhitney"):
                with self.subTest(num=num, method=method), TemporaryDirectory() as tmp:
                    groups = ["CTRL", "T vs X:/?", "T<2", "T|3"][:num]
                    p, rows = make_project(Path(tmp), groups)
                    params = stats.StatsParams(metric=self.metric, test=method, control="CTRL")
                    with (patch.object(stats, "volcano_plot"), patch.object(stats, "bar_plot") as bars,
                          patch("cfos.core.heatmap.heatmap_figure") as heatmap):
                        out, summary, figures = stats.run_group_stats(p, Path(tmp), params, region_sec=rows)
                    workbook = pd.read_excel(out / "组间统计.xlsx", sheet_name=None)
                    self.assertEqual("多组总体检验" in workbook, num >= 3)
                    self.assertIn(f"{num} 个组", summary)
                    pages = [set(call.args[0]["组别"]) for call in heatmap.call_args_list]
                    self.assertEqual(set.union(*pages), set(groups))
                    self.assertTrue(all("CTRL" in page and len(page) <= 3 for page in pages))
                    pairwise = workbook["组间比较"]
                    self.assertEqual(set(pairwise["比较组"]), set(groups[1:]))
                    pvals = pairwise["p 值"].dropna()
                    np.testing.assert_allclose(pairwise.loc[pvals.index, "q 值 (FDR 校正)"],
                                               st.false_discovery_control(pvals))
                    for fig in figures:
                        self.assertEqual(fig.parent, Path(tmp))
                        self.assertFalse(set(fig.name) & set('<>:"/\\|?*'))
                    self.assertEqual(set(bars.call_args_list[0].args[0]["组别"]), set(groups[:2]))
                    if num < 3:
                        continue
                    overall = workbook["多组总体检验"].set_index("脑区缩写")
                    arrays = [np.arange(5) + g * 3 for g in range(num)]
                    expected = (st.f_oneway(*arrays, equal_var=False) if method == "welch"
                                else st.kruskal(*arrays))
                    self.assertAlmostEqual(overall.loc["A", "p 值"], expected.pvalue)
                    self.assertAlmostEqual(overall.loc["A", "统计量"], expected.statistic)
                    self.assertTrue(pd.isna(overall.loc["B", "p 值"]))
                    self.assertIn("不足", overall.loc["B", "说明"])
                    self.assertEqual(overall.loc["B", f"{groups[-1]} 动物数"], 0)

    def test_scoped_group_with_no_region_data_keeps_omnibus(self):
        with TemporaryDirectory() as tmp:
            p, rows = make_project(Path(tmp), ["Control", "A", "B"])
            rows = rows[~rows["图片"].str.startswith("image_2_")]
            with (patch.object(stats, "volcano_plot"), patch.object(stats, "bar_plot"),
                  patch("cfos.core.heatmap.heatmap_figure")):
                out, _, _ = stats.run_group_stats(p, tmp, stats.StatsParams(metric=self.metric), region_sec=rows)
            overall = pd.read_excel(out / "组间统计.xlsx", sheet_name="多组总体检验")
            self.assertTrue(overall["p 值"].isna().all())
            self.assertTrue(overall["B 动物数"].eq(0).all())

    def test_section_count_distinguishes_images_but_animal_is_one_sample(self):
        with TemporaryDirectory() as tmp:
            p, rows = make_project(Path(tmp), ["A"])
            rows = rows[rows["图片"].isin(rows["图片"].unique()[:2])]
            for im in rows["图片"].unique():
                p.set_sample(f"{im}/S01", animal="same-animal", group="A")
            animal = stats.per_animal(rows, p.samples)
            self.assertTrue(animal["脑片数"].eq(2).all())
            self.assertEqual(animal["动物"].nunique(), 1)
            self.assertTrue(animal["面积 (mm²)"].eq(2).all())
