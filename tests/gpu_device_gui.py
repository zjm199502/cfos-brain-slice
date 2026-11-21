"""Exercise the device controls using temporary preferences and the host's real GPU inventory."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy.QtCore import QEventLoop, QSettings, QTimer
from qtpy.QtWidgets import QApplication

from cfos.core import cellpose4
from cfos.core.gpu_devices import gpu_names
from cfos.gui.device_panel import DetectionDeviceBox


def main():
    app = QApplication.instance() or QApplication([])
    with TemporaryDirectory() as tmp:
        prefs = QSettings(str(Path(tmp) / "device.ini"), QSettings.IniFormat)
        def run_task(job, done, label):
            done(job(SimpleNamespace(is_cancelled=lambda: False)))
        with patch("cfos.gui.device_panel.QSettings", return_value=prefs):
            box = DetectionDeviceBox(run_task)
            loop = QEventLoop()
            box.inventory_ready.connect(lambda _: loop.quit())
            QTimer.singleShot(15000, loop.quit)
            box.show()
            loop.exec_()
            app.processEvents()
            assert box.hardware.text() == "系统显卡：" + ("、".join(gpu_names()) or "未读取到型号")
            print(box.hardware.text())
            print(box.runtime.text())
            box.mode.setCurrentIndex(box.mode.findData("cpu"))
            assert prefs.value("detectionDevice") == "cpu"
            assert box.runtime.text() == cellpose4.backend_text()
            second = DetectionDeviceBox(run_task)
            assert second.mode.currentData() == "cpu"
            box.mode.setCurrentIndex(box.mode.findData("auto"))
            with patch.object(cellpose4, "probe_device", wraps=cellpose4.probe_device) as probe:
                box.probe.click()
                probe.assert_called_once()
            assert box.runtime.text() == cellpose4.backend_text()
            assert "尚未运行" not in box.runtime.text()
            print(box.runtime.text())
            box.close()
            second.close()
            cellpose4.close()


if __name__ == "__main__":
    main()
