"""Joint AP positioning of a slice series from slice order and spacing (series.fit_order / position_series)."""
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from cfos.core import deep_refine as D
from cfos.core import registration as R
from cfos.core import series as S
from cfos.core.project import Project

STEP = 160 / 25.   # 160 µm spacing in 25 µm voxels


def coronal(y, tilt_ml_deg=0.0):
    """A QuickNII-style coronal plane through AP voxel y (normal along +y)."""
    a = np.deg2rad(tilt_ml_deg)
    u = np.array([456. * np.cos(a), 456. * np.sin(a), 0.])
    v = np.array([0., 0., -320.])
    c = np.array([228., y, 160.])
    return R.Registration(plane=list(np.concatenate([c - .5 * u - .5 * v, u, v])), method="giRAff",
                          status="CCFv3 · giRAff")


def project(names_y, animal=None, registered=()):
    """Fake project: one image whose sections (in listing order) were located at the given AP voxels."""
    secs = [SimpleNamespace(name=n, enabled=True, registration=(coronal(y).to_dict() if n in registered else {}))
            for n, y in names_y]
    im = SimpleNamespace(key="img1", source_id="src1", sections=secs)
    samples = {f"img1/{n}": {"animal": animal} for n, _ in names_y} if animal else {}
    return SimpleNamespace(images=[im], samples=samples, registration_atlas="CCFv3", section_spacing_um=0.0)


class FitOrderTest(unittest.TestCase):
    def test_known_spacing_finds_the_jumping_slice(self):
        # S03 placed 400 µm (16 voxels) too far, as giRAff's widened search did on A海马3-4.
        fit = S.fit_order([300., 300 + STEP, 300 + 2 * STEP + 16], STEP)
        self.assertEqual(fit["outliers"].tolist(), [False, False, True])
        self.assertTrue(fit["consistent"])
        self.assertAlmostEqual(fit["pred"][2], 300 + 2 * STEP, places=6)

    def test_reverse_cutting_direction(self):
        fit = S.fit_order([300., 300 - STEP - 10, 300 - 2 * STEP, 300 - 3 * STEP], STEP)
        self.assertEqual(fit["outliers"].tolist(), [False, True, False, False])
        self.assertLess(fit["slope"], 0)

    def test_unknown_spacing_estimates_it(self):
        y = 300 + 5.0 * np.arange(6)
        y[3] -= 14
        fit = S.fit_order(y)
        self.assertEqual(np.flatnonzero(fit["outliers"]).tolist(), [3])
        self.assertAlmostEqual(fit["slope"], 5.0, places=6)

    def test_small_jitter_is_not_flagged(self):
        fit = S.fit_order([300., 300 + STEP + 2, 300 + 2 * STEP - 3, 300 + 3 * STEP + 1], STEP)
        self.assertFalse(fit["outliers"].any())   # ≤ 75 µm stays inside the 100 µm floor

    def test_too_few_slices_to_tell(self):
        self.assertIsNone(S.fit_order([0., 6., 30.]))          # unknown spacing needs 4
        self.assertIsNone(S.fit_order([0., STEP], STEP))

    def test_order_not_matching_positions_is_inconsistent(self):
        fit = S.fit_order([300., 300 + 3 * STEP, 300 + STEP, 300 + 4 * STEP, 300 + 2 * STEP], STEP)
        self.assertFalse(fit["consistent"])


class PositionSeriesTest(unittest.TestCase):
    def test_outlier_moved_and_flagged_neighbours_bounded(self):
        names = [("S01", 300.), ("S02", 300 + STEP), ("S03", 300 + 2 * STEP + 16)]
        proj = project(names)
        proj.section_spacing_um = 160.
        new = {("src1", n): coronal(y) for n, y in names}
        out = S.position_series(proj, new)
        s03 = new[("src1", "S03")]
        self.assertTrue(s03.plane_raw)
        self.assertAlmostEqual(R.adjusted_plane(s03).center()[1], 300 + 2 * STEP, places=4)
        self.assertIn("切片顺序", out["notes"][("src1", "S03")][0])
        self.assertNotIn(("src1", "S01"), out["notes"])
        self.assertFalse(new[("src1", "S01")].plane_raw)
        tol = 100 / 25.
        for key, (lo, hi) in out["bounds"].items():
            self.assertLessEqual(lo, 0.)
            self.assertGreaterEqual(hi, 0.)
            self.assertLessEqual(hi - lo, 2 * tol + 1e-6)
        # Undo through the series dialog restores giRAff's own answer.
        self.assertTrue(S.undo(s03))
        self.assertAlmostEqual(R.adjusted_plane(s03).center()[1], 300 + 2 * STEP + 16, places=4)

    def test_previous_sections_are_references_only(self):
        names = [("S01", 300.), ("S02", 300 + 5.), ("S03", 310.), ("S04", 315. - 14), ("S05", 320.)]
        proj = project(names, registered={"S01", "S02", "S03"})
        new = {("src1", n): coronal(y) for n, y in names[3:]}
        out = S.position_series(proj, new)
        self.assertEqual(set(out["bounds"]), {("src1", "S04"), ("src1", "S05")})
        self.assertAlmostEqual(R.adjusted_plane(new[("src1", "S04")]).center()[1], 315., places=4)
        self.assertEqual(out["groups"][0]["moved"][0]["section"], "S04")

    def test_inconsistent_series_is_not_moved(self):
        ys = [300., 300 + 3 * STEP, 300 + STEP, 300 + 4 * STEP, 300 + 2 * STEP]
        names = [(f"S0{i + 1}", y) for i, y in enumerate(ys)]
        proj = project(names)
        proj.section_spacing_um = 160.
        new = {("src1", n): coronal(y) for n, y in names}
        out = S.position_series(proj, new)
        self.assertTrue(all(not r.plane_raw for r in new.values()))
        self.assertTrue(out["notes"])
        self.assertIn("未自动挪动", next(iter(out["notes"].values()))[0])

    def test_animals_are_separate_series(self):
        names = [("S01", 300.), ("S02", 300 + STEP), ("S03", 300 + 2 * STEP + 16)]
        proj = project(names)
        proj.samples = {"img1/S01": {"animal": "A"}, "img1/S02": {"animal": "A"}, "img1/S03": {"animal": "B"}}
        proj.section_spacing_um = 160.
        new = {("src1", n): coronal(y) for n, y in names}
        out = S.position_series(proj, new)
        self.assertFalse(out["notes"])                       # 2 + 1 slices: nobody can be judged
        self.assertTrue(all(not r.plane_raw for r in new.values()))
        self.assertEqual(set(out["bounds"]), {("src1", "S01"), ("src1", "S02")})


class FewSlicesTest(unittest.TestCase):
    def test_three_slices_unknown_spacing_only_bound_the_search(self):
        names = [("S01", 300.), ("S02", 306.), ("S03", 314.)]
        new = {("src1", n): coronal(y) for n, y in names}
        out = S.position_series(project(names), new)
        self.assertFalse(out["notes"])
        self.assertTrue(all(not r.plane_raw for r in new.values()))
        self.assertEqual(out["bounds"][("src1", "S03")], (-4., 4.))   # half of a 7-voxel gap < 100 µm floor


class SearchBoundsTest(unittest.TestCase):
    def test_ap_search_respects_series_bounds(self):
        orig, tex = D.texture_score, D.TEXTURE
        D.TEXTURE = "paint"
        D.texture_score = lambda reg, *a, **k: float(reg.ap_auto)   # texture keeps pulling posterior/anterior
        try:
            img = np.random.default_rng(0).random((40, 50)).astype(np.float32)
            mask = np.zeros((40, 50), bool); mask[5:35, 5:45] = True
            changed, rep = D.search_plane(coronal(300.), img, mask, None, 25., ap_bounds=(-2., 3.))
            self.assertEqual(changed["ap_auto"], 3.)
            self.assertEqual(rep["ap_bounds"], [-2., 3.])
            changed, _ = D.search_plane(coronal(300.), img, mask, None, 25.)
            self.assertEqual(changed["ap_auto"], 16.)
        finally:
            D.texture_score, D.TEXTURE = orig, tex


class AutoregRunTest(unittest.TestCase):
    def test_run_positions_series_before_the_texture_fit(self):
        from cfos.core import autoreg
        ys = {"S01": 300., "S02": 300 + STEP, "S03": 300 + 2 * STEP + 16}
        proj = project(list(ys.items()))
        proj.section_spacing_um, proj.z_mode, proj.results_dir = 160., "max", "/nonexistent"
        slide = SimpleNamespace(pixel_size_um=.5)
        seen = {}

        def refit(reg, image, mask, results_dir, pixel_um, search_plane=False, ap_bounds=None):
            seen[round(R.adjusted_plane(reg).center()[1], 3)] = ap_bounds
            reg.warnings = []          # the real refit resets warnings; series notes must survive it
            reg.nonlinear = True
        saved = autoreg.fit_section, R.section_thumbnail, R.refit_outline
        autoreg.fit_section = lambda sl, sec, *a, **k: coronal(ys[sec.name])
        R.section_thumbnail = lambda *a: (np.zeros((4, 4), np.uint8), np.ones((4, 4), bool), 16.)
        R.refit_outline = refit
        try:
            out = autoreg.run(proj, proj.images, 0, nonlinear=True, slides={"src1": slide})
        finally:
            autoreg.fit_section, R.section_thumbnail, R.refit_outline = saved
        regs = out["src1"]
        self.assertEqual(sorted(seen), [300., round(300 + STEP, 3), round(300 + 2 * STEP, 3)])
        self.assertTrue(all(b is not None for b in seen.values()))
        self.assertTrue(regs["S03"].warnings and not regs["S01"].warnings)
        self.assertIn("series", regs["S03"].metrics)


class ProjectSpacingTest(unittest.TestCase):
    def test_spacing_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Project(folder=__import__("pathlib").Path(d))
            p.section_spacing_um = 160.
            p.save()
            self.assertEqual(Project.load(d).section_spacing_um, 160.)


if __name__ == "__main__":
    unittest.main()
