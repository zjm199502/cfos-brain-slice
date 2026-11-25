from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from cfos.core.coloc import summarize
from cfos.core.export import params_table
from cfos.core.project import Project
from cfos.core.stats import add_metrics, metric_options


class RemovedIntensityGrades(TestCase):
    def test_historical_grades_do_not_generate_new_counts_or_metrics(self):
        cells = pd.DataFrame({"section": ["S01", "S01"], "pos_ch0": [True, True],
                              "pos_ch1": [True, False], "grade_ch1": [3, 0]})
        row = summarize(cells, ["DAPI", "Green"]).iloc[0]
        self.assertEqual((row["DAPI+"], row["Green+"], row["DAPI+Green 共标"]), (2, 1, 1))
        self.assertNotIn("Green+ 强", row.index)
        metrics = add_metrics(pd.DataFrame({"面积 (mm²)": [2.], "Green+": [4], "Green+ 强": [1]}))
        self.assertEqual(metrics["Green+ 密度 (个/mm²)"][0], 2)
        self.assertFalse(any("强" in name for name in metric_options(["DAPI", "Green"])))
        self.assertNotIn("Green+ 强 密度 (个/mm²)", metrics)

    def test_old_project_opens_without_graded_export_parameters(self):
        with TemporaryDirectory() as folder:
            p = Project(Path(folder), channel_names=["DAPI", "Green"], grading={"Green": [2., 6.]})
            p.save()
            opened = Project.load(p.folder)
            self.assertEqual(opened.grading, p.grading)
            self.assertNotIn("强度分级", params_table(opened).to_string(index=False))
