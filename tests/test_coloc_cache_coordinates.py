import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cfos.core.coloc import ColocParams, colocalize, merged_cells
from cfos.core.detection import ChannelParams, DetectionSettings, detect_section
from cfos.core.sections import Section


class CachedCoordinatesTests(unittest.TestCase):
    def test_mixed_cached_coordinates_preserve_cross_channel_matches(self):
        standard = pd.DataFrame({"section": ["S01"] * 3, "channel": [0, 1, 1],
                                 "x": [12.5, 13.5, 32.5], "y": [17.5, 17.5, 17.5]})
        legacy = standard.assign(section="S03").astype({"x": object, "y": object})
        cached = pd.concat([standard, legacy], ignore_index=True)
        result = colocalize(cached, 1., ColocParams())
        self.assertEqual(result["merged_id"].tolist(), [0, 0, 1, 2, 2, 3])
        self.assertEqual(result["combo"].tolist(), ["0+1", "0+1", "1"] * 2)
        self.assertEqual(result["x"].dtype, np.float64)
        self.assertEqual(result["y"].dtype, np.float64)
        self.assertEqual(merged_cells(result, 2)["x"].tolist(), [13., 32.5] * 2)

    def test_empty_resolution_result_keeps_numeric_columns(self):
        slide = SimpleNamespace(channels=["DAPI"], pixel_size_um=1.,
                                level_for_pixel_size=lambda _: 0, downsample=lambda _: 1.)
        section = Section(1, (0, 0, 32, 32), [], "S01")
        empty = detect_section(slide, section, DetectionSettings(channels={0: ChannelParams()}),
                               selected_boxes=[])
        self.assertTrue(empty.empty)
        self.assertEqual(empty["channel"].dtype, np.int64)
        for name in ("x", "y", "area_um2", "mean_ch0"):
            self.assertEqual(empty[name].dtype, np.float64)
