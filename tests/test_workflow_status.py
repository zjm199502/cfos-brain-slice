from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from cfos.core import figure
from cfos.core.export import export_excel
from cfos.core.project import ImageEntry, Project
from cfos.core.scope import scoped_project
from cfos.core.sections import Section
from cfos.core.workflow_status import record_table_output, section_progress, source_folders


class WorkflowStatusTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.project = Project(Path(tmp.name), channel_names=["Cy5"])
        self.image = ImageEntry("A/sample.tif", storage_key="sample-A",
                                sections=[Section(i, (i * 48, 0, 32, 32), [], name)
                                          for i, name in enumerate(["S1", "S2", "S3"])])
        self.other = ImageEntry("B/sample.tif", storage_key="sample-B",
                                sections=[Section(0, (0, 0, 32, 32), [], "S1")])
        self.project.images = [self.image, self.other]
        self.df = pd.DataFrame(dict(section=["S1"], merged_id=[0], combo=["0"], channel=[0],
                                    x=[5.], y=[3.]))

    def statuses(self, image=None):
        return dict(section_progress(self.project, image or self.image))

    def test_sources_disambiguate_even_if_parent_folder_has_same_name(self):
        paths = [r"C:\A\scans\01.tif", r"C:\B\scans\01.tif", "/shared/unique/02.tif", r"C:\A\scans\02.tif"]
        result = source_folders([ImageEntry(path) for path in paths])
        self.assertEqual(result[0], (r"A\scans", r"C:\A\scans"))
        self.assertEqual(result[1], (r"B\scans", r"C:\B\scans"))
        self.assertEqual(result[2], ("unique", "/shared/unique"))
        self.assertEqual(result[0], result[3])

    def test_partial_results_zero_and_unknown_do_not_collapse_to_latest_action(self):
        self.df.attrs["processed_sections"] = ["S1", "S2"]
        self.project.save_cells(self.image, self.df)
        self.image.sections[1].registration = {"plane": [1] * 9, "warnings": ["请检查"]}
        self.image.sections[2].enabled = False
        self.image.status = "导出完成"
        self.image.display_overrides = {"0": {"gamma": 2.0}}
        self.image.section_display = {"S3": {"0": {"gamma": 3.0}}}
        with patch("pandas.read_pickle", side_effect=AssertionError("项目页不应载入大型细胞轮廓")):
            rows = self.statuses()
        self.assertEqual(rows["S1"][2:6], ("图片设置", "已识别（有细胞）", "已有共标结果", "待配准"))
        self.assertEqual(rows["S2"][3:6], ("已识别（零细胞）", "零细胞", "已配准（待复核）"))
        self.assertEqual(rows["S3"][1:5], ("已拆分（停用）", "脑片设置", "待识别", "待识别"))
        self.assertEqual(rows["S1"][7:], ("无导出记录", "无制图记录"))

    def test_legacy_history_and_partial_checkpoint(self):
        self.project.save_part(self.image, self.image.sections[2], "sig", self.df)
        self.assertEqual(self.statuses()["S3"][3], "已检测，待汇总")
        self.image.sections[2].bbox0 = (22, 0, 32, 32)
        self.assertEqual(self.statuses()["S3"][3], "待识别")
        self.df.drop(columns="combo").to_pickle(self.project._old_cells_path(self.image))
        self.image.sections[2].enabled = False
        rows = self.statuses()
        self.assertEqual(rows["S1"][4], "待计算")
        self.assertEqual(rows["S2"][3], "已识别（零细胞）")
        self.assertEqual(rows["S3"][3], "待识别")

    def test_invalid_legacy_cells_have_stable_read_failure_and_no_repeated_reads(self):
        pd.DataFrame({"cell_id": [1]}).to_pickle(self.project._old_cells_path(self.image))
        with patch("pandas.read_pickle", wraps=pd.read_pickle) as read:
            for _ in range(2):
                self.assertEqual(self.statuses()["S1"][3], "结果读取失败")
            self.assertEqual(read.call_count, 1)

    def test_scoped_excel_and_group_history_survive_reopen_and_respect_geometry(self):
        self.project.save_cells(self.image, self.df)
        self.project.set_sample("sample-A/S1", animal="A1", group="C")
        view = scoped_project(self.project, {"sample-A": ["S1"]})
        p = self.project.folder
        with patch("cfos.core.export._write_xlsx", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                export_excel(view, p / "bad.xlsx", masks=False, qc=False, quality=False)
        self.assertEqual(self.statuses()["S1"][7], "无导出记录")
        export_excel(view, p / "one.xlsx", masks=False, qc=False, quality=False)
        table = pd.DataFrame({"图片": ["sample-A", "sample-A", "sample-B"], "脑片": ["S1", "S2", "S1"]})
        record_table_output(self.project, "group", table)
        self.project.save()
        reopened = Project.load(p)
        first, second = reopened.images
        self.assertEqual(dict(section_progress(reopened, first))["S1"][6:8], ("已分组 · 曾参与统计", "曾导出"))
        self.assertEqual(dict(section_progress(reopened, first))["S2"][6:8], ("待分组", "无导出记录"))
        self.assertEqual(dict(section_progress(reopened, second))["S1"][6:8], ("待分组", "无导出记录"))
        first.sections[0].bbox0 = (10, 0, 32, 32)
        self.assertEqual(dict(section_progress(reopened, first))["S1"][7], "无导出记录")

    def test_figures_record_only_successful_overlapping_sections(self):
        self.image.figure_rois = [{"name": "first", "x": 2, "y": 0, "w": 8, "h": 8},
                                  {"name": "failed", "x": 54, "y": 0, "w": 8, "h": 8}]
        slide = Mock(level_shapes=[(40, 150)])
        st = figure.FigureSettings()
        with patch.object(self.image, "open_slide", return_value=slide), patch.object(
                figure, "export_roi", side_effect=[[Path("success.png")], OSError("failed")]):
            files, errors = figure.export_images(self.project, [self.image], st)
        self.assertEqual(len(files), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual([state[8] for state in self.statuses().values()],
                         ["曾制图", "无制图记录", "无制图记录"])
