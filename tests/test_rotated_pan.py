"""Exercise the shipped napari camera and VisPy viewbox mouse event path."""
import unittest
from types import SimpleNamespace

import numpy as np
from napari._vispy.camera import MouseToggledPanZoomCamera
from vispy.scene.widgets import ViewBox
from vispy.util.event import Event

from cfos.gui.canvas_transform import pan_rotated_view
from cfos.gui.tile_selection import TileSelectionMixin


def make_view(angle):
    view = ViewBox(size=(640, 480))
    view.camera = MouseToggledPanZoomCamera(
        aspect=1, rect=(250, 40, 400, 300), flip=(False, True, False)
    )
    window = SimpleNamespace(
        qt_viewer=SimpleNamespace(canvas=SimpleNamespace(view=view)),
        view_angle=SimpleNamespace(value=lambda: angle),
        tile_overlay=SimpleNamespace(update=lambda: None),
    )
    TileSelectionMixin._rotate_preview(window)
    view.events.mouse_move.connect(pan_rotated_view, position="first")
    return view


def mouse_move(view, start, previous, current, modifiers=()):
    press = SimpleNamespace(pos=start)
    event = Event(
        "mouse_move", pos=current, last_event=SimpleNamespace(pos=previous),
        press_event=press, buttons=[1], modifiers=modifiers,
        mouse_event=SimpleNamespace(press_event=press, pos=current, modifiers=modifiers),
    )
    view.events.mouse_move(event)
    return event


class RotatedPanTest(unittest.TestCase):
    def test_point_under_cursor_tracks_successive_drags(self):
        for angle in (0, 37.5, -90, 90, 180):
            with self.subTest(angle=angle):
                view = make_view(angle)
                view.camera.zoom(0.37, center=(334, 92))
                previous = start = (127, 203)
                anchor = view.scene.transform.imap(start)
                for current in ((199, 203), (199, 122), (125, 267)):
                    event = mouse_move(view, start, previous, current)
                    self.assertTrue(event.handled)
                    np.testing.assert_allclose(view.scene.transform.map(anchor)[:2], current, atol=1e-3)
                    previous = current

    def test_respects_edit_modes_and_modified_drags(self):
        view = make_view(35)
        start, end = (200, 125), (240, 210)
        center = view.camera.center
        self.assertFalse(mouse_move(view, start, start, end, modifiers=("Shift",)).handled)
        view.camera.mouse_pan = False
        self.assertFalse(mouse_move(view, start, start, end).handled)
        self.assertEqual(view.camera.center, center)
