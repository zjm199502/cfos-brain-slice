import unittest
from types import SimpleNamespace

import numpy as np
from napari.layers import Image
from vispy.scene import Node

from cfos.gui.atlas_outline import AtlasOutline, set_outline_paths


class AtlasOutlineLayer(unittest.TestCase):
    def test_scene_attachment_and_separate_paths(self):
        layer = Image(np.zeros((7, 7, 4), np.uint8), rgb=True, opacity=.8)
        image_node = Node()
        canvas = SimpleNamespace(layer_to_visual={layer: SimpleNamespace(node=image_node)})
        viewer = SimpleNamespace(canvas=canvas)
        paths = [np.array([[0, 1], [3, 4]]), np.array([[3, 4], [7, 5]])]
        set_outline_paths(viewer, layer, paths)
        (outline,) = image_node.children
        self.assertIsInstance(outline, AtlasOutline)
        np.testing.assert_equal(outline.stroke.connect, [True, False, True, False])
        # Input corners map to the same local image coordinates used by the RGBA texture.
        np.testing.assert_allclose(outline.transform.map(outline.stroke.pos)[:, :2],
                                   [[1, 0], [4, 3], [4, 3], [5, 7]])
        layer.opacity = .35
        self.assertAlmostEqual(outline.opacity, .35)
        set_outline_paths(viewer, layer, [])
        self.assertFalse(outline.visible)
        set_outline_paths(viewer, layer, paths[:1])
        self.assertTrue(outline.visible)
        self.assertEqual(image_node.children, [outline])


if __name__ == "__main__":
    unittest.main()
