import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from cfos.core.pipeline import detect_image
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section


class SectionSelectionTests(unittest.TestCase):
    def test_incremental_selection_preserves_results_and_tracks_empty_sections(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "slide.tif"
            source.touch()
            sections = [Section(i, (100*i, 0, 100, 100), [], f"S{i}") for i in range(4)]
            sections[3].enabled = False
            image = ImageEntry(str(source), sections=sections)
            project = Project(Path(folder), images=[image], channel_names=["Cy5"])
            slide = SimpleNamespace(pixel_size_um=1)
            cell = pd.DataFrame([dict(section="S0", channel=0, x=30., y=40., area_um2=50., mean_ch0=100.)])
            with patch("cfos.core.detection.detect_section", side_effect=[
                    cell, cell.iloc[:0].copy(), cell.assign(section="S2", x=220.)]) as infer:
                def run(*names):
                    return detect_image(project, image, sl=slide, section_names=names)
                with self.assertRaises(ValueError):
                    run("S3")
                run("S0")
                first = project.load_cells(image)
                self.assertEqual(first.attrs["processed_sections"], ["S0"])
                run("S1", "S2")
                result = project.load_cells(image)
                self.assertEqual(result.attrs["processed_sections"], ["S0", "S1", "S2"])
                self.assertEqual(list(result.section), ["S0", "S2"])
                pd.testing.assert_frame_equal(result.iloc[:1], first)
                self.assertTrue(result.merged_id.is_unique)
                self.assertEqual(run("S1")["reused"], 1)
                self.assertEqual([c.args[1].name for c in infer.call_args_list], ["S0", "S1", "S2"])
