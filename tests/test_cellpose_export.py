import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cfos.core import cellpose4
from cfos.core.detection import ChannelParams
from cfos.core.export import params_table
from cfos.core.project import Project


class CellposeExportTests(unittest.TestCase):
    def test_export_uses_the_selected_model_and_its_own_threshold_after_reopening(self):
        with TemporaryDirectory() as tmp:
            project = Project(Path(tmp), channel_names=["DAPI", "EGFP"])
            project.detection.channels = {
                0: ChannelParams(),
                1: ChannelParams(model=cellpose4.MODEL_KEY, cellprob_threshold=-0.8),
            }
            project.save()
            project = Project.load(tmp)
            rows = params_table(project).set_index("参数")["值"]
            self.assertIn("Cellpose 4", rows["细胞识别神经网络"])
            self.assertEqual(rows["DAPI Cellpose 掩膜阈值"], 0.0)
            self.assertEqual(rows["EGFP Cellpose 掩膜阈值"], -0.8)
            self.assertIn("cpsam_v2", rows["EGFP 识别模型"])
            self.assertNotIn("EGFP 置信度阈值", rows)
            project.unused_channels = [0, 1]
            rows = params_table(project).set_index("参数")["值"]
            self.assertEqual(rows["细胞识别神经网络"], "未启用识别通道")
