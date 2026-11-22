"""Draw the registration preview in screen-width strokes on the napari canvas."""
import numpy as np
from vispy.scene.visuals import Compound
from vispy.visuals import LineVisual
from vispy.visuals.transforms import MatrixTransform


class AtlasOutline(Compound):
    def __init__(self, parent, layer):
        self.halo = LineVisual(color=(0.06, 0.05, 0.02, 0.70), width=3.0, antialias=True)
        self.stroke = LineVisual(color=(1.0, 0.91, 0.16, 1.0), width=1.5, antialias=True)
        super().__init__([self.halo, self.stroke], parent=parent)
        # napari places children of 2D images in pixel-centre coordinates,
        # shifting them by half a pixel relative to the image texture.
        self.transform = MatrixTransform()
        self.transform.translate((0.5, 0.5, 0))
        self.opacity = layer.opacity
        layer.events.opacity.connect(self._on_layer_opacity)

    def _on_layer_opacity(self, event):
        self.opacity = event.source.opacity

    def set_paths(self, paths):
        self.visible = bool(len(paths))
        if not self.visible:
            return
        # Input is (row, col) on cell corners; VisPy expects (x, y) about
        # image pixel centres. Break connectivity at every branch endpoint.
        vertices = np.ascontiguousarray(np.concatenate(paths)[:, ::-1], dtype=np.float32) - 0.5
        ends = np.cumsum([len(path) for path in paths]) - 1
        connect = np.ones(len(vertices), dtype=bool)
        connect[ends] = False
        for line in (self.halo, self.stroke):
            line.set_data(pos=vertices, connect=connect)


def set_outline_paths(qt_viewer, layer, paths):
    parent = qt_viewer.canvas.layer_to_visual[layer].node
    outline = next((child for child in parent.children if isinstance(child, AtlasOutline)), None)
    if outline is None:
        outline = AtlasOutline(parent, layer)
    outline.set_paths(paths)
