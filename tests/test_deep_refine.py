"""Deep-structure texture refinement and the transformix output directory."""
import os
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

from cfos.core import deep_refine as D


def bands(shift=0):
    """Two bright 'cell layers' in a dim tissue disc, as in a DAPI hippocampus."""
    yy, xx = np.mgrid[:160, :200]
    mask = (yy - 80) ** 2 / 70 ** 2 + (xx - 100) ** 2 / 90 ** 2 < 1
    labels = np.where(mask, 1, 0)
    labels[(np.abs(yy - 60 - shift) < 4) & mask] = 2
    labels[(np.abs(yy - 100 - shift) < 4) & mask] = 3
    return labels, mask


class DeepRefineTest(unittest.TestCase):
    def test_paint_uses_region_medians(self):
        labels, mask = bands()
        density = np.where(labels >= 2, .9, .2).astype(np.float32)
        synth = D.paint_labels(labels, density, mask)
        self.assertAlmostEqual(float(synth[60, 100]), .9, places=5)
        self.assertAlmostEqual(float(synth[80, 100]), .2, places=5)
        self.assertEqual(float(synth[0, 0]), 0.)

    def test_compose_adds_prior_at_displaced_point(self):
        shape = (20, 30)
        residual = (np.full(shape, 2., np.float32), np.zeros(shape, np.float32))
        gy, gx = np.mgrid[:20, :30].astype(np.float32)
        prior = (gx * .1, np.zeros(shape, np.float32))
        total = D.compose(residual, prior)
        self.assertAlmostEqual(float(total[0][5, 10]), 2. + 1.2, places=4)
        self.assertIs(D.compose(residual, None)[0].dtype, np.dtype(np.float32))

    def test_bspline_recovers_band_shift_from_readonly_cwd(self):
        try:
            import itk  # noqa: F401
        except ImportError:
            self.skipTest("elastix not installed")
        labels, mask = bands()
        moved, _ = bands(shift=5)
        density = ndi.gaussian_filter(np.where(labels >= 2, 1., .2) * mask, 1.5).astype(np.float32)
        synth = ndi.gaussian_filter(np.where(moved >= 2, 1., .2) * mask, 1.5).astype(np.float32)
        outline = ndi.gaussian_filter(mask.astype(np.float32), 2)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o500)   # like "/" for an app started from Finder
            os.chdir(tmp)
            try:
                dx, dy = D._bspline(density, synth, outline, outline, 24)
            finally:
                os.chdir(cwd)
                os.chmod(tmp, 0o700)
        # Sampling the atlas 5 px further down puts its bands on the section's bands.
        self.assertGreater(float(np.median(dy[55:65, 60:140])), 3.)
        self.assertGreater(float(np.median(dy[95:105, 60:140])), 3.)


if __name__ == "__main__":
    unittest.main()
