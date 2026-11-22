from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from cfos.core.project import Project
from cfos.core.sections import Section


class WorkflowStatusGuiTests(unittest.TestCase):
    def test_project_master_selection_shows_each_step_and_full_source(self):
        from napari.qt import get_qapp
        from cfos.gui.main_window import MainWindow, STEPS
        app = get_qapp()
        with TemporaryDirectory() as temp:
            base = Path(temp)
            project = Project(base / "project")
            for folder in ["A", "B"]:
                source = base / folder / "sample.tif"
                source.parent.mkdir()
                tifffile.imwrite(source, np.zeros((64, 64), np.uint16))
                im, _ = project.import_image(source)
                im.sections = [Section(1, (0, 0, 32, 32), [], "S01")]
            df = pd.DataFrame(dict(section=["S01"], combo=["0"], merged_id=[0],
                                   channel=[0], x=[4.], y=[4.]))
            project.save_cells(project.images[0], df)
            project.save()
            window = MainWindow()
            try:
                with patch("cfos.core.log.attach_project"), patch(
                        "cfos.core.slide_io.Slide._arr", lambda sl, level: sl._tif.series[0].levels[level].asarray()):
                    window._load_project(project)
                app.processEvents()
                self.assertEqual(window.images_table.item(0, 0).text(), "sample.tif")
                self.assertEqual(window.images_table.item(1, 0).text(), "sample.tif")
                self.assertEqual(window.images_table.item(0, 3).text(), "A")
                self.assertEqual(window.images_table.item(1, 3).text(), "B")
                self.assertEqual(window.progress_table.rowCount(), len(STEPS))
                self.assertEqual(window.progress_table.item(3, 0).text(), "已识别（有细胞）")
                window.images_table.setCurrentCell(1, 0)
                self.assertIn(str((base / "B").resolve()), window.source_folder_label.text())
                self.assertEqual(window.progress_table.item(3, 0).text(), "待识别")
                self.assertIs(window.image, project.images[0])
                project.images[1].sections = []
                window._refresh_images()
                self.assertEqual(window.progress_table.item(1, 0).text(), "待拆分")
            finally:
                window.close()
