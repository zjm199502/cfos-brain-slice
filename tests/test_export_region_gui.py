"""Excel 脑区自选的 Qt 交互回归；仅需要 Qt，无需 OpenGL 窗口。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import MethodType
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QMainWindow,
                            QProgressBar, QPushButton, QStackedWidget, QWidget)

from cfos.core import region_sets as RS
from cfos.gui.region_scope import RegionPreviewDialog, RegionScopeBox


class ExportRegionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        table = pd.DataFrame([
            (997, "root", "", "Root", "整个脑", 0),
            (1089, "HPF", "root", "Hippocampal formation", "海马结构", 1),
            (382, "CA1", "HPF", "Field CA1", "海马 CA1 区", 2),
            (463, "CA3", "HPF", "Field CA3", "海马 CA3 区", 2),
            (295, "BLA", "root", "Basolateral amygdalar nucleus", "基底外侧杏仁核", 1),
        ], columns=["id", "acronym", "parent", "name", "zh", "depth"])
        table["脑区类别"] = "测试"
        atlas = patch.object(RS, "info", return_value=table)
        atlas.start()
        self.addCleanup(atlas.stop)

    def test_search_and_edit_are_independent_of_parent_and_filter(self):
        dialog = RegionPreviewDialog(RS.CUSTOM, region_ids=[1089, 295])
        self.addCleanup(dialog.close)
        for term in ("field ca1", "CA1", "海马 CA1", "382"):
            dialog.search.setText(term)
            self.assertFalse(dialog._by_id[382].isHidden())
            self.assertFalse(dialog._by_id[1089].isHidden())
            self.assertTrue(dialog._by_id[295].isHidden())
        self.assertEqual(dialog._by_id[382].checkState(0), Qt.Unchecked)
        dialog._by_id[382].setCheckState(0, Qt.Checked)
        self.assertEqual(set(dialog.selected_ids()), {1089, 382, 295})
        self.assertEqual(dialog._by_id[463].checkState(0), Qt.Unchecked)
        dialog.search.setText("杏仁核")
        self.assertFalse(dialog._by_id[295].isHidden())
        self.assertTrue(dialog._by_id[382].isHidden())
        dialog.tabs.setCurrentIndex(1)
        for row in range(dialog.selection.count()):
            item = dialog.selection.item(row)
            item.setSelected(item.data(Qt.UserRole) in {1089, 295})
        dialog.remove_button.click()
        self.assertEqual(dialog.selected_ids(), [382])
        dialog.clear_button.click()
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.selection.count(), 0)

    def test_cancel_rolls_back_and_presets_keep_the_previous_choices(self):
        export = RegionScopeBox(allow_custom=True)
        self.addCleanup(export.close)
        export.set_values(RS.DEPTH, 6, True, [382])
        with patch.object(RegionPreviewDialog, "exec", return_value=QDialog.Rejected):
            export.combo.setCurrentIndex(export.combo.findData(RS.CUSTOM))
        self.assertEqual((export.mode(), export.max_depth(), export.skip_layers()), (RS.DEPTH, 6, True))
        self.assertEqual(export.region_ids(), [382])

        def choose(dialog):
            dialog._by_id[1089].setCheckState(0, Qt.Checked)
            dialog._by_id[463].setCheckState(0, Qt.Checked)
            dialog.apply_button.click()
            return dialog.result()

        with patch.object(RegionPreviewDialog, "exec", choose):
            export.combo.setCurrentIndex(export.combo.findData(RS.CUSTOM))
        selected = export.region_ids()
        self.assertEqual(set(selected), {1089, 382, 463})
        self.assertFalse(export.skip_layers())
        self.assertIn("3", export.desc.text())

        def cancel(dialog):
            dialog.clear_button.click()
            dialog.reject()
            return dialog.result()

        with patch.object(RegionPreviewDialog, "exec", cancel):
            export.btn.click()
        self.assertEqual(export.mode(), RS.CUSTOM)
        self.assertEqual(export.region_ids(), selected)
        export.combo.setCurrentIndex(export.combo.findData(RS.ALL))
        self.assertEqual(export.region_ids(), selected)
        export.set_values(RS.ALL)
        self.assertEqual(export.region_ids(), [])

    def test_group_region_choice_survives_refresh_and_project_reopen(self):
        from cfos.core.project import Project
        from cfos.gui.groups_panel import GroupsMixin

        class Window(QMainWindow, GroupsMixin):
            _processing_scope = None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = Window()
            self.addCleanup(window.close)
            window.project = Project(root, export_region_set=RS.CUSTOM, export_region_ids=[295])
            panel = window._panel_groups()
            window.setCentralWidget(panel)
            window._sync_groups_panel()
            scope = window.st_scope
            def choose(dialog):
                dialog._by_id[382].setCheckState(0, Qt.Checked)
                dialog.apply_button.click()
                return dialog.result()
            with patch.object(RegionPreviewDialog, "exec", choose):
                scope.combo.setCurrentIndex(scope.combo.findData(RS.CUSTOM))
            window.project = Project.load(root)
            window._sync_groups_panel()
            self.assertEqual((scope.mode(), scope.region_ids()), (RS.CUSTOM, [382]))
            self.assertEqual(window._stats_params().region_ids, [382])
            self.assertEqual(window.project.export_region_ids, [295])

    def test_manual_and_batch_exports_keep_one_scope_and_release_it_on_exit(self):
        from cfos.core.project import ImageEntry, Project
        from cfos.gui import main_window
        window = QMainWindow()
        self.addCleanup(window.close)
        window.stack = QStackedWidget(window)
        window.setCentralWidget(window.stack)
        for _ in range(7):
            window.stack.addWidget(QWidget())
        window.exp_scope = RegionScopeBox(allow_custom=True)
        window.stack.addWidget(window.exp_scope)
        window.exp_scope.set_values(RS.CUSTOM, region_ids=[382])
        window.exp_scope.changed.connect(lambda: main_window.MainWindow._save_export_scope(window))
        window.exp_hemisphere = QComboBox()
        window.exp_hemisphere.addItem("左右合并")
        window.exp_intensity, window.exp_masks, window.exp_qc = QCheckBox(), QCheckBox(), QCheckBox()
        window.device_box = Mock()
        window.progress, window.cancel_btn = QProgressBar(), QPushButton()
        window.info, window.error = Mock(), Mock()
        window.task = window._after_task = None
        window._finish_transition = lambda: None
        window._cells_cache = {}
        for method in ("_refresh_images", "_refresh_channels_form", "_refresh_enabled", "_refresh_section_table"):
            setattr(window, method, Mock())
        window.run_task = MethodType(main_window.MainWindow.run_task, window)
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        project = Project(Path(temp.name), export_region_set=RS.CUSTOM, export_region_ids=[382])
        window.project = project

        def wait_until(condition):
            end = time.monotonic() + 5
            while not condition() and time.monotonic() < end:
                self.app.processEvents()
                time.sleep(.005)
            self.assertTrue(condition())

        for batch, outcome in [(False, "ok"), (True, "ok"), (False, "error"), (True, "cancel")]:
            with self.subTest(batch=batch, outcome=outcome):
                started, release = Event(), Event()
                read = []
                fresh = ImageEntry(str(project.folder / f"fresh-{len(project.images)}.tif"))
                def exporter(p, *args, **kwargs):
                    read.append(p.export_region_ids.copy())
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("test timeout")
                    read.append(p.export_region_ids.copy())
                    if outcome == "error":
                        raise ValueError("test export error")
                    return project.folder / "selected.xlsx"
                def pipeline(p, *args, **kwargs):
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("test timeout")
                    if kwargs["cancel"]():
                        raise InterruptedError
                    p.images.append(fresh)
                    return []
                with patch.object(main_window, "export_excel", side_effect=exporter) as writer, \
                     patch.object(QFileDialog, "getSaveFileName", return_value=(str(project.folder / "selected.xlsx"), "")), \
                     patch("cfos.core.pipeline.list_images", return_value=[Path("fresh.tif")]), \
                     patch("cfos.core.pipeline.run_batch", side_effect=pipeline):
                    if batch:
                        main_window.MainWindow.run_batch(window, str(project.folder))
                    else:
                        main_window.MainWindow.export(window)
                    try:
                        wait_until(started.is_set)
                        self.assertFalse(window.exp_scope.combo.isEnabled())
                        self.assertFalse(window.exp_scope.btn.isEnabled())
                        QTest.keyClick(window.exp_scope.combo, Qt.Key_Home)
                        self.assertEqual(window.exp_scope.mode(), RS.CUSTOM)
                        main_window.MainWindow.export(window)
                        self.assertEqual(window.info.call_args.args[0], "上一个任务还在运行，请等待完成或点击“取消”。")
                        if outcome == "cancel":
                            window.task.cancel()
                    finally:
                        release.set()
                        window.task.wait(5000)
                    wait_until(window.exp_scope.isEnabled)
                    if batch and outcome == "ok":
                        self.assertIn(fresh, writer.call_args.args[0].images)
                    if outcome == "cancel":
                        writer.assert_not_called()
                    else:
                        self.assertEqual(read, [[382], [382]])


if __name__ == "__main__":
    unittest.main()
