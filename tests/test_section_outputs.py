import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
from PIL import ImageDraw

from cfos.core.masks import export_masks
from cfos.core.project import Project, section_result_states
from cfos.core.qc import export_qc
from cfos.core.sections import Section


class SectionOutputTests(unittest.TestCase):
    def test_positive_zero_and_unprocessed_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "image.tif"
            tifffile.imwrite(path, np.ones((64, 256), np.uint8))
            project = Project(root / "project")
            image, _ = project.import_image(path)
            image.sections = [Section(i, (i*64, 0, 64, 64), [], f"S{i}") for i in range(4)]
            image.sections[3].enabled = False
            cells = pd.DataFrame([dict(section="S0", channel=0, x=16., y=16., merged_id=2,
                                       contour=[[10.,10.],[22.,10.],[22.,22.]])])
            cells.attrs["processed_sections"] = ["S0", "S1", "S3"]
            project.save_cells(image, cells)
            self.assertEqual(section_result_states(image, cells), {
                "S0": "已分析（有细胞）", "S1": "已分析（零细胞）", "S2": "未分析"})
            out = export_masks(project, root / "masks")
            masks = sorted(out.glob("*.tif"))
            self.assertEqual([f.name.split("_")[1] for f in masks], ["S0", "S1"])
            self.assertEqual(int(tifffile.imread(masks[0]).max()), 3)
            self.assertFalse(tifffile.imread(masks[1]).any())
            headers = []
            original = ImageDraw.ImageDraw.text
            def record_text(draw, xy, text, *args, **kwargs):
                if xy == (8, 6):
                    headers.append(text)
                return original(draw, xy, text, *args, **kwargs)
            with patch.object(ImageDraw.ImageDraw, "text", new=record_text), patch.object(
                    project, "load_cells", wraps=project.load_cells) as read:
                directory = export_qc(project, root / "qc")
                self.assertEqual(read.call_count, 1)
                self.assertEqual(len(list(directory.glob("*.png"))), 3)
            channel = project.channel_names[0]
            self.assertIn(f"{channel} 1", headers[0])
            self.assertIn(f"{channel} 0", headers[1])
            self.assertRegex(headers[2], "未分析|not analyzed")
            zero = cells.iloc[:0].copy()
            zero.attrs["processed_sections"] = ["S1"]
            project.save_cells(image, zero)
            existing = out / "人工圈选.tif"
            existing.write_bytes(b"user file")
            export_masks(project, out)
            self.assertEqual(existing.read_bytes(), b"user file")
            masks = list(out.glob("image_*.tif"))
            self.assertEqual(len(masks), 1)
            self.assertFalse(tifffile.imread(masks[0]).any())
