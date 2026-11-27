"""Display equivalent-area cell rings in the same GPU scene as the microscope image."""
from functools import partial

import numpy as np
from napari.layers import Layer, Points
from scipy.spatial import cKDTree
from qtpy.QtCore import QEvent, QObject, QSettings
from qtpy.QtGui import QColor
from vispy import gloo
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from ..core.manual import ADD_RADIUS_UM
from ..core.cell_markers import diameters


# Use quads so very large circles at high zoom are not clipped to the driver's
# maximum GL point size. Only the camera transform changes during navigation.
_VERTEX = """
attribute vec2 a_center;
attribute vec2 a_corner;
attribute float a_radius;
uniform float u_width;
varying vec2 v_offset;
varying float v_radius;
void main() {
    vec4 center = $to_fb(vec4(a_center, 0., 1.));
    vec4 edge = $to_fb(vec4(a_center + vec2(a_radius, 0.), 0., 1.));
    v_radius = length(edge.xy / edge.w - center.xy / center.w);
    v_offset = a_corner * (v_radius + 0.5 * u_width + 1.);
    center.xy += v_offset * center.w;
    gl_Position = $to_render(center);
}
"""
_FRAGMENT = """
uniform vec4 u_color;
uniform float u_width;
varying vec2 v_offset;
varying float v_radius;
void main() {
    float dist = length(v_offset);
    float coverage;
    if (v_radius < 1.) {
        coverage = clamp(1.25 - dist, 0., 1.);
    } else {
        float width = min(u_width, 2.*v_radius);
        coverage = clamp(0.5*width + 0.5 - abs(dist-v_radius), 0., 1.);
        if (coverage <= 0.) discard;
        // Qt DashLine has a four-pen-width mark followed by a two-width gap.
        // Fit whole periods to the circle so the seam closes cleanly.
        float circumference = 6.2831853 * v_radius;
        float periods = max(1., floor(circumference / (6.*u_width) + 0.5));
        float phase = fract(atan(v_offset.y, v_offset.x) * periods / 6.2831853 + 0.5);
        coverage *= clamp((0.3333333 - abs(phase-0.5)) * circumference/periods + 0.5, 0., 1.);
    }
    if (coverage <= 0.) discard;
    gl_FragColor = vec4(u_color.rgb, coverage*u_color.a);
}
"""


class _CellPoints(Points):
    """Keep hover and zoom independent of total cell count for read-only overlays."""

    def __init__(self, *args, **kwargs):
        self._hit_tree = None
        self._hit_radius = 0.
        super().__init__(*args, **kwargs)
        self.events.data.connect(self._index_positions)
        self.events.size.connect(self._index_sizes)
        self._index_positions()

    @property
    def _type_string(self):
        return "points"

    def _index_positions(self, event=None):
        self._hit_tree = None
        self._index_sizes()
        if len(self.data) >= 4096 and self.ndim == 2 and np.isfinite(self.data).all():
            self._hit_tree = cKDTree(self.data, copy_data=True)

    def _index_sizes(self, event=None):
        self._hit_radius = self.size.max() / 2 if len(self.size) else 0.

    def _update_draw(self, scale_factor, corner_pixels_displayed, shape_threshold):
        if self.editable or self.selected_data:
            return super()._update_draw(scale_factor, corner_pixels_displayed, shape_threshold)
        # Points re-uploads selection markers on every zoom even when this
        # read-only results layer has no selected cells.
        return Layer._update_draw(self, scale_factor, corner_pixels_displayed, shape_threshold)

    def _get_value(self, position):
        if (self.editable or self._hit_tree is None or self._slice_input.ndisplay != 2
                or len(self._view_indices) != len(self.data) or not np.isfinite(self._hit_radius)):
            return super()._get_value(position)
        # Match napari's rectangular hit area and last matching row rule,
        # while only reading nearby cells when the cursor status is refreshed.
        yx = np.asarray(position)[:2]
        scale = np.abs(self.scale / self.scale[-1])
        if not np.isfinite(yx).all():
            return None
        rows = self._hit_tree.query_ball_point(yx, self._hit_radius / scale.min(), p=np.inf)
        if not rows:
            return None
        rows = np.asarray(rows)
        hits = np.all(np.abs(self.data[rows] - yx) <= self.size[rows, None] / (2 * scale), axis=1)
        return np.max(rows[hits]) if np.any(hits) else None


class _CellRingVisual(Visual):
    def __init__(self):
        super().__init__(_VERTEX, _FRAGMENT)
        self._draw_mode = "triangles"
        self.count = 0
        self.vertices = gloo.VertexBuffer()
        self.on_opacity = self.on_geometry = self.on_visibility = None
        self.width_px = 1.0
        self.color_rgba = (1., 1., 1., 1.)
        self.set_gl_state("translucent", depth_test=False, cull_face=False)

    def set_geometry(self, positions, sizes):
        self.count = len(positions)
        if self.count:
            verts = np.empty((self.count, 6), dtype=[("a_center","f4",2),
                             ("a_corner","f4",2),("a_radius","f4")])
            verts["a_center"] = np.asarray(positions)[:, -2:][:, ::-1][:, None, :]
            verts["a_corner"] = [(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)]
            verts["a_radius"] = np.reshape(sizes, (-1, 1)) * .5
            self.vertices.set_data(verts.ravel())
            self.shared_program.bind(self.vertices)
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert["to_fb"] = view.get_transform("visual", "framebuffer")
        view.view_program.vert["to_render"] = view.get_transform("framebuffer", "render")

    def _prepare_draw(self, view):
        if not self.count:
            return False
        view.view_program["u_width"] = self.width_px * view.transforms.pixel_scale


CellRingNode = create_visual_node(_CellRingVisual)


class CellRings(QObject):
    def __init__(self, canvas, viewer):
        super().__init__(canvas.native)
        self.canvas = canvas
        self.layers = {}
        self.suppressed = False
        self.pending_geometry = set()
        self.settings = QSettings("cfos", "cfos脑片分析")
        self.width_pt = self.settings.value("cellRingWidthPt", 0.75, type=float)
        viewer.layers.events.removed.connect(self._removed)
        canvas.native.installEventFilter(self)

    def attach(self, layer, color):
        host = self.canvas.layer_to_visual[layer].node
        node = CellRingNode(parent=host)
        self.layers[layer] = node
        # Keep the points layer for normal visibility, ordering and metadata.
        # Its transparent stock markers would otherwise draw the same data twice.
        stock_markers = getattr(host, "points_markers", None)
        if stock_markers is not None:
            stock_markers.visible = False
        node.width_px = self.width_pt * self.canvas.native.logicalDpiX()/72.
        self._geometry(layer)
        self._visibility(layer)
        node.on_visibility = partial(self._visibility, layer)
        layer.events.visible.connect(node.on_visibility)
        self.set_color(layer, color)
        node.on_opacity = partial(self._set_opacity, layer)
        node.on_geometry = partial(self._geometry, layer)
        layer.events.opacity.connect(node.on_opacity)
        for emitter in (layer.events.data, layer.events.size):
            emitter.connect(node.on_geometry)

    def _visibility(self, layer, event=None):
        node = self.layers[layer]
        node.parent.visible = layer.visible and not self.suppressed
        node.visible = not self.suppressed

    def set_suppressed(self, suppressed):
        if self.suppressed == suppressed:
            return
        self.suppressed = suppressed
        for layer in self.layers:
            self._visibility(layer)
        if not suppressed:
            for layer in tuple(self.pending_geometry):
                self._geometry(layer)
            self.pending_geometry.clear()

    def _geometry(self, layer, event=None):
        if self.suppressed:
            self.pending_geometry.add(layer)
            return
        if layer in self.layers and len(layer.data) == len(layer.size):
            self.layers[layer].set_geometry(layer.data, layer.size)

    def _set_opacity(self, layer, event=None):
        node = self.layers.get(layer)
        if node is not None:
            r, g, b, a = node.color_rgba
            node.shared_program["u_color"] = (r, g, b, a * layer.opacity)
            node.update()

    def set_color(self, layer, color):
        node = self.layers[layer]
        node.color_rgba = QColor(color).getRgbF()
        self._set_opacity(layer)

    def set_width(self, width):
        self.width_pt = width
        self.settings.setValue("cellRingWidthPt", width)
        self._refresh_width()

    def _refresh_width(self):
        for node in self.layers.values():
            node.width_px = self.width_pt * self.canvas.native.logicalDpiX()/72.
            node.update()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, getattr(QEvent, "DevicePixelRatioChange", None)):
            self._refresh_width()
        return False

    def _removed(self, event):
        layer = event.value
        self.pending_geometry.discard(layer)
        node = self.layers.pop(layer, None)
        if node is not None:
            for emitter, callback in ((layer.events.visible,node.on_visibility), (layer.events.opacity,node.on_opacity),
                                      (layer.events.data,node.on_geometry),(layer.events.size,node.on_geometry)):
                emitter.disconnect(callback)
            node.parent = None


def add_cell_layer(window, frame, channel, name, editable):
    """Sizing follows mask area. Manual additions use their measurement radius."""
    px = window.slide.pixel_size_um
    default_size = 2 * ADD_RADIUS_UM / px
    position = frame[["y", "x"]].to_numpy(float) if frame is not None else np.empty((0, 2))
    size = diameters(frame, px) if frame is not None and len(frame) else default_size
    ids = frame.index.to_numpy() if frame is not None else np.empty(0, int)
    layer = window.viewer.add_layer(_CellPoints(
        position, name=name, size=size, face_color="transparent", border_color="transparent",
        features={"cid": ids}, metadata={"cell_channel": channel}))
    layer.current_size = default_size
    layer.feature_defaults = {"cid": -1}
    layer.editable = editable
    window.cell_rings.attach(layer, window.ch_point_color(channel))
    return layer
