"""细胞识别设备偏好、显卡型号和推理状态。"""
from qtpy.QtCore import QSettings, QThreadPool, QTimer, Qt, Signal
from qtpy.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ..core import cellpose4


class DetectionDeviceBox(QGroupBox):
    inventory_ready = Signal(str)

    def __init__(self, run_task):
        super().__init__("细胞识别设备")
        self.run_task = run_task
        self.settings = QSettings("cfos", "cfos脑片分析")
        self.mode = QComboBox()
        self.mode.addItem("自动（优先 GPU）", "auto")
        self.mode.addItem("仅 CPU", "cpu")
        pref = self.settings.value("detectionDevice", "auto")
        self.mode.setCurrentIndex(max(0, self.mode.findData(pref)))
        cellpose4.set_pref(self.mode.currentData())
        self.mode.currentIndexChanged.connect(self._change_mode)
        self.probe = QPushButton("测试 Cellpose 设备")
        self.probe.clicked.connect(self._probe_device)
        line = QHBoxLayout()
        line.addWidget(self.mode)
        line.addWidget(self.probe)
        layout = QVBoxLayout(self)
        layout.addLayout(line)
        self.install = QPushButton("安装 / 修复 Cellpose 4 运行环境")
        self.install.clicked.connect(self._install_runtime)
        layout.addWidget(self.install)
        self.hardware = QLabel("正在检测显卡型号…")
        self.runtime = QLabel()
        for label in (self.hardware, self.runtime):
            label.setWordWrap(True)
            layout.addWidget(label)
        self.inventory_started = False
        self.inventory_ready.connect(self.hardware.setText, Qt.QueuedConnection)
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self._refresh_status)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.inventory_started:
            self.inventory_started = True
            QThreadPool.globalInstance().start(self._read_hardware)
        self.timer.start()
        self._refresh_status()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def _read_hardware(self):
        from ..core.gpu_devices import gpu_names
        names = gpu_names()
        self.inventory_ready.emit("系统显卡：" + ("、".join(names) if names else "未读取到型号"))

    def _refresh_status(self):
        status = cellpose4.backend_text()
        if status != self.runtime.text():
            self.runtime.setText(status)

    def _change_mode(self):
        pref = self.mode.currentData()
        cellpose4.set_pref(pref)
        self.settings.setValue("detectionDevice", pref)
        self._refresh_status()

    def _probe_device(self):
        self.run_task(lambda task: cellpose4.probe_device(cancel=task.is_cancelled),
                      lambda _: self._refresh_status(), label="正在用 Cellpose-SAM 测试推理设备…")

    def _install_runtime(self):
        from ..core import cellpose_setup
        question = "将下载 Cellpose 运行环境及模型，首次安装约需数 GB。现在开始安装？"
        if QMessageBox.question(self, "安装 Cellpose 4", question, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        def install(task):
            cellpose4.close()
            return cellpose_setup.install(report=lambda message: task.report(0, 0, message),
                                          cancel=task.is_cancelled)
        def installed(_):
            self._refresh_status()
            QMessageBox.information(self, "Cellpose 4", "运行环境已安装。可点“测试 Cellpose 设备”核对推理设备。")
        self.run_task(install, installed, label="正在安装 Cellpose 4 运行环境…")

    def set_busy(self, busy):
        self.mode.setEnabled(not busy)
        self.probe.setEnabled(not busy)
        self.install.setEnabled(not busy)
