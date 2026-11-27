import unittest
from unittest.mock import patch

import numpy as np
from napari.layers import Points

from cfos.gui.cell_rings import _CellPoints


class CellRingNavigationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(88)
        yx = rng.uniform(-5000, 5000, (5000, 2))
        yx[-2:] = [[300, 511], [300, 515]]
        sizes = rng.uniform(2, 12, len(yx))
        sizes[-2:] = [20, 16]
        self.layer = _CellPoints(yx, size=sizes, scale=(.5, 1.5))
        self.layer.editable = False

    def test_overlap_scale_visibility_and_edit_mode_match_napari(self):
        layer = self.layer
        for hidden in [False, True]:
            shown = np.ones(len(layer.data), bool)
            shown[-1] = not hidden
            layer.shown = shown
            for yx in [[300, 511], [299, 512], [314, 512], [0, 9000]]:
                self.assertEqual(layer._get_value(yx), Points._get_value(layer, yx))
        layer.editable = True
        self.assertEqual(layer._get_value([300, 512]), Points._get_value(layer, [300, 512]))

    def test_hit_index_follows_size_and_position_changes(self):
        layer = self.layer
        self.assertIsNotNone(layer._hit_tree)
        layer.size = 100
        self.assertEqual(layer._get_value([300, 540]), len(layer.data) - 1)
        moved = layer.data.copy()
        moved[-1] = [8000, 9000]
        layer.data = moved
        self.assertEqual(layer._get_value([8000, 9000]), len(layer.data) - 1)
        layer.data = moved[:3]
        self.assertIsNone(layer._hit_tree)

    def test_zoom_skips_empty_highlight_on_read_only_layer(self):
        layer = self.layer
        with patch.object(layer, "_set_highlight", wraps=layer._set_highlight) as highlight:
            layer._update_draw(.5, np.array([[10., 10.], [500., 500.]]), (200, 200))
            highlight.assert_not_called()
            layer.editable = True
            layer._update_draw(.3, np.array([[10., 10.], [400., 400.]]), (200, 200))
            self.assertTrue(highlight.called)
