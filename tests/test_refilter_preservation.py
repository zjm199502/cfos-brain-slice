from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import tifffile

from cfos.core.detection import ChannelParams
from cfos.core.manual import save_manual
from cfos.core.pipeline import refresh_coloc
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section


class RefilterPreservationTests(TestCase):
    def test_mixed_legacy_repeated_coloc_and_new_raw_cache(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "slide.tif"
            tifffile.imwrite(path, np.zeros((64, 512), np.uint8))
            sections = [Section(i, (i*100, 0, 80, 60), [], name)
                        for i, name in enumerate(("A", "B", "Zero", "Pending"))]
            im = ImageEntry(str(path), sections=sections)
            project = Project(root, images=[im], channel_names=["Cy5", "FITC"])
            project.detection.channels[0] = ChannelParams(min_intensity=1000)
            previous = pd.DataFrame([
                dict(section=s, channel=c, x=x, y=10., area_um2=25., mean_ch0=200.,
                     mean_ch1=200., contour=[], manual=0, combo="stale")
                for s, c, x in [("A", 0, 20.), ("B", 0, 120.), ("B", 0, 121.), ("B", 1, 121.)]
            ])
            previous.attrs["carried_forward_sections"] = ["B", "Zero"]
            previous.attrs["processed_sections"] = ["A", "B", "Zero"]
            project.save_cells(im, previous)
            project.save_part(im, sections[0], "A", previous[previous.section == "A"])
            save_manual(project, im, {
                "added": [dict(section="B", channel=0, x=155., y=30., means=[150., 0.], contour=[])],
                "removed": [dict(section="B", channel=0, x=120., y=10.)],
            })

            for _ in range(2):
                self.assertEqual(refresh_coloc(project), 1)
                rebuilt = project.load_cells(im)
                self.assertEqual(rebuilt.attrs["processed_sections"], ["A", "B", "Zero"])
                self.assertEqual(rebuilt.attrs["carried_forward_sections"], ["B", "Zero"])
                self.assertEqual(rebuilt.section.unique().tolist(), ["A", "B"])
                self.assertEqual(sorted(rebuilt.x), [20., 121., 121., 155.])
                self.assertEqual(rebuilt[rebuilt.x == 121].combo.tolist(), ["0+1", "0+1"])
                self.assertEqual(rebuilt.manual.sum(), 1)

            project.save_part(im, sections[1], "B", previous[previous.section == "B"])
            refresh_coloc(project)
            updated = project.load_cells(im)
            self.assertEqual(updated.attrs["carried_forward_sections"], ["B", "Zero"])
            self.assertEqual(updated.combo.tolist(), ["0", "0+1", "0+1", "0"])
