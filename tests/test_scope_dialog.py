import unittest
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from cfos.gui.processing_scope import ScopeDialog


class ScopeDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selecting_last_section_does_not_request_resplitting_image(self):
        section = SimpleNamespace(name="S01", enabled=True)
        image = SimpleNamespace(display_name="sample.tif", key="sample",
                                path="D:/day1/sample.tif", sections=[section])
        dlg = ScopeDialog([image], {})
        self.addCleanup(dlg.close)
        root = dlg.tree.topLevelItem(0)
        self.assertFalse(dlg.ok.isEnabled())
        root.child(0).setCheckState(0, Qt.Checked)
        self.assertEqual(dlg.selection(), {"sample": ("S01",)})
        root.setCheckState(0, Qt.Checked)
        self.assertEqual(dlg.selection(), {"sample": None})
        dlg._set_all(Qt.Unchecked)
        self.assertEqual(dlg.selection(), {})
        self.assertFalse(dlg.ok.isEnabled())

    def test_reopen_restores_named_sections_and_distinguishes_sources(self):
        images = [
            SimpleNamespace(display_name="same.tif", key=folder, path=f"/{folder}/same.tif",
                            sections=[SimpleNamespace(name="S01", enabled=True)])
            for folder in ("day1", "day2")
        ]
        scope = {"day1": ("S01",), "day2": None}
        dlg = ScopeDialog(images, scope)
        self.addCleanup(dlg.close)
        self.assertEqual(dlg.selection(), scope)
        self.assertNotEqual(dlg.tree.topLevelItem(0).text(1), dlg.tree.topLevelItem(1).text(1))
