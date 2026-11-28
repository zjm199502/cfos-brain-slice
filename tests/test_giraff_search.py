"""Coarse-to-fine coronal plane search in giRAff (giraff.refine_planes)."""
import unittest

from cfos.core import giraff as G


def cand(z, o, s):
    return {"atlas_z_index": z, "orientation": o, "score": s}


class RefinePlanesTest(unittest.TestCase):
    def test_neighbours_of_the_best_coarse_planes_in_their_orientation(self):
        coarse = [cand(z, o, 1.0 + (0.02 if (z, o) == (280, "r180") else 0.01 if (z, o) == (16, "r0") else 0.))
                  for z in range(0, 528, 8) for o in ("r0", "r180")]
        fine = dict(G.refine_planes(coarse, 528, 8, 2, 3))
        self.assertEqual(sorted(fine), [13, 14, 15, 17, 18, 19, 277, 278, 279, 281, 282, 283])
        self.assertEqual(fine[278], ("r180",))
        self.assertEqual(fine[14], ("r0",))

    def test_stays_inside_the_atlas_and_skips_planes_already_searched(self):
        coarse = [cand(0, "r0", 2.), cand(8, "r0", 1.), cand(520, "r180", 1.5)]
        fine = dict(G.refine_planes(coarse, 524, 8, 3, 4))
        self.assertNotIn(0, fine)
        self.assertNotIn(8, fine)
        self.assertEqual(max(fine), 523)
        self.assertTrue(all(z >= 0 for z in fine))


if __name__ == "__main__":
    unittest.main()
