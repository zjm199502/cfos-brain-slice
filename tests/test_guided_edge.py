import unittest

import numpy as np

from cfos.core.guided_edge import contour_edits, fluorescence_edge
from cfos.core.edge_drag import deform_paths, snap_path


class Slide:
    pixel_size_um = 1.
    level_shapes = [(180, 230)]

    def __init__(self):
        y, x = np.indices(self.level_shapes[0])
        self.image = 10. + 90.*(y > 76.+.1*x)

    def read(self, level, roi, channels):
        x, y, w, h = roi
        assert channels == [1]
        return self.image[None, y:y+h, x:x+w]

    def level_for_pixel_size(self, um):
        return 0

    def downsample(self, level):
        return 1.


class GuidedEdge(unittest.TestCase):
    def setUp(self):
        self.slide = Slide()
        self.clicks = [(25, 87), (92, 96), (155, 100), (207, 104)]

    def trace(self):
        return fluorescence_edge(self.slide, self.clicks, 1, (0, 0, 230, 180))

    def test_seek_between_clicks_in_both_directions(self):
        xy = self.trace()
        self.assertLess(np.max(np.abs(xy[:, 1] - 76. - .1*xy[:, 0])), 3.)
        self.clicks.reverse()
        np.testing.assert_allclose(xy, self.trace()[::-1], atol=3.1)

    def test_uniform_image_is_rejected(self):
        self.slide.image.fill(25.)
        with self.assertRaisesRegex(ValueError, "缺少可辨边缘"):
            self.trace()

    def test_group_and_rotated_label_geometry(self):
        target = self.trace()
        xy = np.array([[8., 64.], [222., 85.]])
        paths = [xy[:, ::-1]]
        normal = contour_edits(paths, target, 230, 180, 1., 100, False, [])
        rotated = contour_edits(paths, target, 230, 180, 1., 100, True, [])
        self.assertTrue(all(p[7] == 1 for p in normal))
        np.testing.assert_allclose(np.asarray(normal)[:, :4], 1.-np.asarray(rotated)[:, :4])
        for edit in normal:
            paths = deform_paths(paths, edit, 230, 180)
        for x, y in target[3:-3:3]:
            self.assertIsNotNone(snap_path(paths, x, y, 4.))

    def test_unrelated_fragment_does_not_follow_guide(self):
        with self.assertRaisesRegex(ValueError, "贯穿"):
            contour_edits([np.array([[85., 80.], [85., 110.]])], self.trace(), 230, 180, 1., 50., False, [])
