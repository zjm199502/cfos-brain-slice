import unittest
from unittest.mock import patch

import numpy as np
from cfos.core import registration as R
from cfos.core.edge_drag import landmark_radius, transport


class RegistrationLandmarks(unittest.TestCase):
    def test_limit_swept_radius_at_previously_placed_target(self):
        size = np.array([200, 100])
        first = [0.38, 0.4, 0.4, 0.4, 0.3, 2, 1]
        start, end = np.array([70., 48.]), np.array([90., 48.])
        r = landmark_radius([first], start, end, *size, 40.)
        self.assertEqual(r, 8.)
        new = [*(start / size), *(end / size), r / size[0], 2, 1]
        old_target = np.array([[80.], [40.]])
        np.testing.assert_allclose(transport(*old_target, new, *size), old_target)
        np.testing.assert_allclose(transport(*start[:, None], new, *size), end[:, None])

    def test_prior_drag_and_duplicate_target(self):
        points = [[.3, .4, .4, .4, .1, 2, 1], [.4, .4, .5, .4, .1, 2]]
        self.assertAlmostEqual(landmark_radius(points, [50.,40.], [51.,40.], 100,100,6.), 0.)
        self.assertEqual(landmark_radius([points[1]], [50.,40.], [51.,40.], 100,100,6.), 6.)


class ApSearch(unittest.TestCase):
    def setUp(self):
        self.reg = R.Registration(plane=[420.,250.,280.,-400.,0.,0.,0.,0.,-240.])
        self.mask = np.ones((30, 40), bool)
        self.image = np.arange(1200, dtype=np.float32).reshape(30,40)

    def test_no_signal_and_cancellation_leave_initial_geometry_intact(self):
        out = R.refine_ap(self.reg,self.mask,self.image*0)
        self.assertFalse(out["has_signal"])
        self.assertEqual(out["shift_um"], 0)
        self.reg.ap_auto = 3
        with self.assertRaises(InterruptedError):
            R.refine_ap(self.reg,self.mask,self.image,cancelled=lambda: True)
        self.assertEqual(self.reg.ap_auto, 3)

    def search(self, scores):
        def sample(plane, w, h, which, **kwargs):
            um = round((plane.o[1]-250)*25)
            return {k: np.full((h,w),1 if k=="annotation" else 1+scores.get(um,0),np.float32) for k in which}

        with (patch.object(R,"_ap_candidates",return_value=list(scores)),
              patch.object(R,"sample_plane",side_effect=sample),
              patch.object(R,"_elastix",return_value=None),
              patch.object(R,"similarity",side_effect=lambda ref,img,mask: float(ref[0,0])-1.)):
            return R.refine_ap(self.reg,self.mask,self.image)

    def test_both_limits_and_separated_competing_peaks(self):
        for limit in (-10_000,10_000):
            out = self.search({0:1., limit:2.})
            self.assertEqual(out["shift_um"],limit)
            self.assertTrue(out["edge"])
            self.assertTrue(all(abs(s)<=10_000 for s in out["scores"]))
        out = self.search({0:1., 200:1.3, -3000:1.9, 5000:2.})
        self.assertEqual(out["shift_um"],200)
        self.assertEqual(out["ambiguous_um"],5000)

if __name__=="__main__":
    unittest.main()
