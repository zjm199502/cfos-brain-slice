import unittest

import numpy as np

from cfos.core.atlas_contours import boundary_paths


class PreviewContours(unittest.TestCase):
    def test_shared_interface_and_crop_coordinates(self):
        labels = np.ones((23, 30), np.uint16)
        labels[:, 10:] = 42
        original = labels.copy()
        (path,) = boundary_paths(labels)
        self.assertEqual(path.dtype, np.float32)
        self.assertEqual({tuple(path[0]), tuple(path[-1])}, {(0, 10), (23, 10)})
        np.testing.assert_array_equal(path[:, 1], 10)
        np.testing.assert_array_equal(labels, original)
        self.assertEqual(boundary_paths(np.ones((4, 5))), [])
        self.assertEqual(boundary_paths(np.zeros((0, 5))), [])

    def test_junctions_do_not_split_or_duplicate(self):
        labels = np.full((20, 20), 3)
        labels[:9, :7] = 1
        labels[:9, 7:] = 2
        paths = boundary_paths(labels, 7)
        self.assertEqual(len(paths), 3)
        ends = [tuple(p[i]) for p in paths for i in (0, -1)]
        self.assertCountEqual(ends, [(9, 7)] * 3 + [(0, 7), (9, 0), (9, 20)])
        # Sharing only a corner must not join previously disjoint labels.
        paths = boundary_paths(np.array([[1, 0], [0, 1]]))
        self.assertEqual(sum(tuple(p[i]) == (1, 1) for p in paths for i in (0, -1)), 4)

    def test_single_pixel_island_and_thin_ring_remain_distinct(self):
        labels = np.zeros((32, 32), np.uint16)
        labels[4:28, 4:28] = 5
        labels[5:27, 5:27] = 0
        labels[14, 14] = 7
        rings = sorted(boundary_paths(labels, 10), key=lambda p: np.ptp(p[:, 0]))
        self.assertEqual(len(rings), 3)
        for p in rings:
            np.testing.assert_equal(p[0], p[-1])
        island, inner, outer = rings
        self.assertTrue(np.all(island.min(axis=0) < (14.5, 14.5)))
        self.assertTrue(np.all(island.max(axis=0) > (14.5, 14.5)))
        self.assertTrue(np.all(outer.min(axis=0) < inner.min(axis=0)))
        self.assertTrue(np.all(outer.max(axis=0) > inner.max(axis=0)))
        # Even in the curved corners the two outlines may not touch or cross.
        from matplotlib.path import Path
        self.assertTrue(Path(outer).contains_points(inner).all())

    def test_voxel_size_controls_staircase_smoothing(self):
        y, x = np.indices((100, 100))
        cut = 10 + (x // 6) * 4
        labels = y >= cut
        (path,) = boundary_paths(labels, voxel_px=4)
        if path[0, 1] > path[-1, 1]:
            path = path[::-1]
        samples = np.arange(10.5, 90, 1)
        actual = np.interp(samples, path[:, 1], path[:, 0])
        ideal = 8 + samples * (4 / 6)
        raw = cut[0, samples.astype(int)]
        self.assertLess(np.std(actual - ideal), np.std(raw - ideal) / 3)
        self.assertLess(np.max(np.abs(actual - ideal)), 0.4)


if __name__ == "__main__":
    unittest.main()
