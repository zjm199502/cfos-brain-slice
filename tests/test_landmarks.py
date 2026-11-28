"""Landmarks: instant interpolated preview, single deletion and no stale state."""
import tempfile
import unittest

import numpy as np

from cfos.core import registration as R

W, H, PX = 120, 90, 8.0   # thumbnail size and µm per pixel


def make_reg(results):
    # Coronal plane: atlas X follows the section's x, Y the section's y (1 voxel = 1 thumbnail pixel).
    reg = R.Registration(plane=[100., 200., 300., W, 0., 0., 0., H, 0.], status="已配准")
    base = np.zeros((2, H, W), np.float32)
    base[0] += 1.5   # an automatic nonlinear result that is not the identity
    reg.warp_file = reg.base_warp_file = R.save_warp(results, base)
    reg.nonlinear = True
    return reg, base


def atlas_at(reg, field, x, y):
    return R.landmark_atlas_xyz(reg, W, H, field, x, y)


def load(results, reg):
    return np.load(R.warp_path(results, "", "", reg))


class LandmarkTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = self._tmp.name
        self.reg, self.base = make_reg(self.results)

    def tearDown(self):
        self._tmp.cleanup()

    def landmark(self, atlas_xy, section_xy):
        """The atlas content shown at atlas_xy belongs at section_xy."""
        xyz = atlas_at(self.reg, self.base, *atlas_xy)
        return [section_xy[0] / W, section_xy[1] / H, *xyz]

    def test_field_is_exact_at_landmarks_and_untouched_far_away(self):
        L = [self.landmark((40, 40), (46, 50)), self.landmark((80, 30), (78, 34))]
        field = R.landmark_field(self.reg, self.base, W, H, PX, L, radius_um=200)
        self.assertTrue(max(R.landmark_offsets_um(self.reg, field, W, H, PX, L)) < 2)
        before = R.landmark_offsets_um(self.reg, self.base, W, H, PX, L)
        self.assertAlmostEqual(before[0], np.hypot(6, 10) * PX, places=3)
        self.assertTrue(np.allclose(field[:, 85:, :10], self.base[:, 85:, :10]))   # > 200 µm away

    def test_add_delete_and_clear_have_no_stale_state(self):
        a, b = self.landmark((40, 40), (46, 50)), self.landmark((80, 30), (78, 34))
        R.set_landmarks(self.reg, [a, b], self.results, W, H, PX)
        self.assertEqual(self.reg.landmark_fit, "preview")
        self.assertNotEqual(self.reg.warp_file, self.reg.base_warp_file)
        both = load(self.results, self.reg)
        self.assertTrue(max(R.landmark_offsets_um(self.reg, both, W, H, PX)) < 2)
        # Deleting one landmark immediately drops its correction, without a refit.
        R.set_landmarks(self.reg, [b], self.results, W, H, PX)
        one = load(self.results, self.reg)
        self.assertGreater(R.landmark_offsets_um(self.reg, one, W, H, PX, [a])[0], 50)
        self.assertLess(R.landmark_offsets_um(self.reg, one, W, H, PX, [b])[0], 2)
        # Clearing returns exactly to the automatic result.
        R.set_landmarks(self.reg, [], self.results, W, H, PX)
        self.assertEqual(self.reg.warp_file, self.reg.base_warp_file)
        self.assertEqual(self.reg.landmark_fit, "")
        self.assertTrue(np.array_equal(load(self.results, self.reg), self.base))

    def test_contradicting_landmark_is_refused_and_record_unchanged(self):
        a = self.landmark((40, 40), (40, 52))
        R.set_landmarks(self.reg, [a], self.results, W, H, PX)
        saved = dict(self.reg.__dict__)
        b = self.landmark((40, 44), (40, 30))   # neighbouring atlas point pulled the opposite way
        with self.assertRaises(R.LandmarkError):
            R.set_landmarks(self.reg, [a, b], self.results, W, H, PX)
        self.assertEqual(self.reg.__dict__, saved)

    def test_landmarks_follow_plane_changes(self):
        a = self.landmark((40, 40), (46, 50))
        R.set_landmarks(self.reg, [a], self.results, W, H, PX)
        self.reg.shift_x = .05   # slider moved: the atlas plane changed, the landmark did not
        R.set_landmarks(self.reg, self.reg.landmarks, self.results, W, H, PX)
        field = load(self.results, self.reg)
        self.assertLess(R.landmark_offsets_um(self.reg, field, W, H, PX)[0], 2)

    def test_legacy_edits_must_be_removed_first(self):
        self.reg.control_points = [[.3, .3, .32, .3, .05, 2]]
        with self.assertRaises(R.LandmarkError):
            R.set_landmarks(self.reg, [self.landmark((40, 40), (46, 50))], self.results, W, H, PX)

    def test_old_project_without_base_keeps_current_warp_as_base(self):
        self.reg.base_warp_file = ""
        auto = self.reg.warp_file
        R.set_landmarks(self.reg, [self.landmark((40, 40), (46, 50))], self.results, W, H, PX)
        self.assertEqual(self.reg.base_warp_file, auto)
        R.set_landmarks(self.reg, [], self.results, W, H, PX)
        self.assertEqual(self.reg.warp_file, auto)

    def test_record_round_trip(self):
        R.set_landmarks(self.reg, [self.landmark((40, 40), (46, 50))], self.results, W, H, PX)
        again = R.Registration.from_dict(self.reg.to_dict())
        self.assertEqual((again.landmarks, again.landmark_fit, again.base_warp_file),
                         (self.reg.landmarks, "preview", self.reg.base_warp_file))


if __name__ == "__main__":
    unittest.main()
