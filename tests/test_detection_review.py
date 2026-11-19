import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from cfos.core import accuracy, detection
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section
from cfos.core.unmix import apply
from types import SimpleNamespace


class Slide:
    channels = [0, 1]
    pixel_size_um = 1

    def __init__(self):
        self.reads = []
        self.image = np.zeros((2, 16, 48), np.uint16)

    def level_for_pixel_size(self, um):
        return int(um == 2)

    def downsample(self, level):
        return 1

    def read(self, level, box, channels=None):
        self.reads.append((level, box))
        x, y, w, h = box
        return self.image[:, y:y+h, x:x+w][channels if channels is not None else slice(None)]


class DetectionReview(unittest.TestCase):
    def setUp(self):
        self.slide = Slide()
        self.sec = Section(1, (0, 0, 48, 16), [(1, 1), (14, 1), (14, 14), (1, 14)], "s")
        self.channels = {c: detection.ChannelParams() for c in range(2)}

    def test_scores(self):
        cells = pd.DataFrame({"x": [2], "y": [2], "channel": [0]})
        for auto, manual, expected in [(cells, [(2, 2)], 1), (cells, [], 0),
                                       (cells.iloc[:0], [(2, 2)], 0), (cells, [(24, 24)], 0)]:
            self.assertEqual(accuracy.check(auto, 0, (0, 0, 30, 30), manual, 1)["f1"], expected)
        self.assertTrue(np.isnan(accuracy.check(cells.iloc[:0], 0, (0, 0, 30, 30), [], 1)["f1"]))

    def test_norms(self):
        self.slide.image[:] = np.arange(self.slide.image.size).reshape(self.slide.image.shape)
        model = SimpleNamespace(matrix=[[1, .1], [.02, 1]], offsets=[1, 2])
        actual = detection.section_norms(self.slide, self.sec, self.channels, model)
        mask = detection.section_mask(self.sec, (16, 48), 1)
        for c in self.channels:
            plane = apply(self.slide.image, model, [c])[0]
            np.testing.assert_allclose(actual[c], np.percentile(plane[mask], [1, 99.8, 50]))
        self.assertEqual(len(self.slide.reads), 1)

    def test_dark_tile_trace_survives_project_cache(self):
        settings = detection.DetectionSettings(tile=16, overlap=0, channels=self.channels)
        with patch.object(detection, "_pool", return_value=None):
            raw = detection.detect_section(self.slide, self.sec, settings, prefilter=False)
        self.assertEqual(self.slide.reads, [(1, self.sec.bbox0), (0, (0, 0, 16, 16))])
        with TemporaryDirectory() as folder:
            project = Project(Path(folder), detection=settings, channel_names=["DAPI", "Green"])
            image = ImageEntry("01.tif", sections=[self.sec])
            project.save_part(image, self.sec, detection.raw_signature(settings), raw)
            reason = accuracy.miss_details(project, image, 1, [[8, 8]], 1)[0]["线索"]
            self.assertIn("亮度过低", reason)
            sig = detection.raw_signature(settings)
            settings.channels[1].skip_dark_tiles = False
            self.assertNotEqual(sig, detection.raw_signature(settings))

    def test_candidate_thresholds_and_config_groups(self):
        settings = detection.DetectionSettings(channels=self.channels)
        with TemporaryDirectory() as folder:
            project = Project(Path(folder), detection=settings, channel_names=["DAPI", "Green"])
            image = ImageEntry("01.tif", sections=[self.sec])
            cells = pd.DataFrame([dict(channel=1, x=4., y=4., area_um2=12., mean_ch1=2., contrast=.5)])
            self.channels[1].min_intensity = 3
            self.channels[1].min_contrast = 1
            project.save_part(image, self.sec, "test", cells)
            clues = accuracy.miss_details(project, image, 1, [(4, 4)], 1)
            self.assertIn("面积、强度、对比度", clues[0]["线索"])
            box = (0, 0, 16, 16)
            res = accuracy.check(cells.iloc[:0], 1, box, [(4, 4)], 1)
            for threshold in (3, 4):
                self.channels[1].min_intensity = threshold
                accuracy.record(project, image, "s", 1, box, res, "")
            summary = accuracy.summary(project)
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary["F1"] == 0).all())
