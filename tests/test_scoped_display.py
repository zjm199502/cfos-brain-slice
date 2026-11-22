import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cfos.core.figure import FigureSettings, render_roi
from cfos.core.project import DisplayChannel, ImageEntry, Project
from cfos.core.sections import Section


class ScopedDisplay(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s1 = Section(1, (0, 0, 50, 50), [], "S01")
        s2 = Section(2, (50, 0, 50, 50), [], "S02")
        self.im = ImageEntry("A.tif", [s1, s2])
        self.other = ImageEntry("B.tif", [Section(1, s1.bbox0, [], s1.name)])
        self.project = Project(Path(tmp.name), [self.im, self.other], ["Green"],
                               {0: DisplayChannel(low=0, high=100, color=[0, 255, 0])})

    def test_field_inheritance_and_round_trip(self):
        p = self.project
        a, b = self.im.sections
        self.assertEqual(p.channel_display(0, self.im, a), p.display[0])
        p.update_display({"A": ["S01"], "B": None}, 0, low=10, high=60)
        p.update_display({"A": ["S01"]}, 0, gamma=.5, color=[255, 0, 0])
        p.update_display({"A": None}, 0, gamma=1.5)
        p.save()
        saved = Project.load(p.folder)
        im, other = saved.images
        self.assertEqual(saved.channel_display(0, im, im.sections[0]),
                         DisplayChannel(low=10, high=60, gamma=1.5, color=[255, 0, 0]))
        self.assertEqual(saved.channel_display(0, im, im.sections[1]),
                         DisplayChannel(low=0, high=100, gamma=1.5, color=[0, 255, 0]))
        self.assertEqual(saved.channel_display(0, other, other.sections[0]),
                         DisplayChannel(low=10, high=60, gamma=1, color=[0, 255, 0]))
        self.assertEqual(saved.display[0], DisplayChannel(low=0, high=100, color=[0, 255, 0]))
        p.update_display({"A": ["S01"]}, 0, color=None)
        self.assertIsNone(p.custom_color(0, self.im, a))
        self.assertEqual(p.custom_color(0, self.im, b), (0, 255, 0))

    def test_figure_resolves_settings_at_roi_center(self):
        p = self.project
        p.update_display({"A": ["S01"]}, 0, color=[255, 0, 0], gamma=2.0)
        sl = SimpleNamespace(channels=[object()], level_shapes=[(50, 100)], n_levels=1,
                             pixel_size_um=1, downsample=lambda lv: 1,
                             read=lambda lv, bbox, channels: np.full((1, bbox[3], bbox[2]), 50, np.uint8))
        for x, expected in ((3, (63, 0, 0)), (53, (0, 127, 0))):
            image, _ = render_roi(p, self.im, sl, {"x": x, "y": 3, "w": 8, "h": 8},
                                  FigureSettings(), with_scale=False)
            self.assertEqual(image.getpixel((4, 4)), expected)


if __name__ == "__main__":
    unittest.main()
