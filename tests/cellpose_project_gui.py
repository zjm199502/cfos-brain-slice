"""Test the Cellpose-only detection page, legacy project settings and task transitions."""
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import tifffile
from napari.qt import get_qapp
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import QDoubleSpinBox, QFormLayout, QMessageBox, QPushButton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfos.core import cellpose4
from cfos.core.project import Project
from cfos.gui.main_window import MainWindow


def main():
    app = get_qapp()
    def wait_for(condition):
        deadline = time.monotonic() + 12
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.015)
        assert condition()

    with TemporaryDirectory() as tmp, patch("cfos.core.log.attach_project"):
        root = Path(tmp)
        source = root / "input.tif"
        tifffile.imwrite(source, np.ones((80, 96), np.uint16))
        project = Project(root / "project")
        project.import_image(source)
        project.save()
        config = project.folder / "cfos_project.json"
        data = json.loads(config.read_text("utf-8"))
        channel = data["detection"]["channels"]["0"]
        channel.update(model="previous-custom", prob_thresh=.8)
        del channel["cellprob_threshold"]
        config.write_text(json.dumps(data), "utf-8")
        prefs = QSettings(str(root / "prefs.ini"), QSettings.IniFormat)
        with patch("cfos.gui.device_panel.QSettings", return_value=prefs):
            win = MainWindow()
        win.setAttribute(Qt.WA_DontShowOnScreen)
        win._load_project(Project.load(project.folder))
        win.steps.setCurrentRow(3)
        win.show()
        app.processEvents()
        form = win.det_rows.itemAt(0).widget().layout()
        params = win.project.detection.channels[0]
        assert params.model == cellpose4.MODEL_KEY
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            if label and label.widget().text() == "Cellpose 掩膜阈值":
                cutoff = form.itemAt(row, QFormLayout.FieldRole).widget()
                break
        assert isinstance(cutoff, QDoubleSpinBox) and cutoff.value() == 0
        cutoff.setValue(-1.5)
        wait_for(lambda: Project.load(project.folder).detection.channels[0].cellprob_threshold == -1.5)
        assert all(b.text() != "开始训练" for b in win.findChildren(QPushButton))

        with patch.object(QMessageBox, "question", return_value=QMessageBox.No), patch(
                "cfos.core.cellpose_setup.install") as install:
            win.device_box.install.click()
            install.assert_not_called()
        def installing(report, cancel):
            report("正在下载测试")
            while not cancel():
                time.sleep(.01)
            raise InterruptedError
        win.info = lambda _: None
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), patch(
                "cfos.core.cellpose_setup.install", side_effect=installing) as install:
            win.device_box.install.click()
            assert not win.device_box.install.isEnabled() and not win.device_box.mode.isEnabled()
            win.cancel_btn.click()
            wait_for(win.device_box.mode.isEnabled)
            install.assert_called_once()

        def slow(task):
            while not task.is_cancelled():
                time.sleep(.01)
            raise InterruptedError
        replacement = Project(root / "replacement")
        replacement.save()
        win.run_task(slow, lambda _: None)
        win._open_saved_project(str(replacement.folder))
        wait_for(lambda: win.project.folder == replacement.folder)
        assert win.centralWidget().isEnabled()
        assert win.slide is None
        win.run_task(slow, lambda _: None)
        win.close()
        wait_for(lambda: not win.isVisible())
        assert win.task is None
    print("Cellpose GUI legacy threshold migration, persistence, switch and close passed")


if __name__ == "__main__":
    main()
