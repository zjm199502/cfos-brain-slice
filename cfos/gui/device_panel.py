"""细胞识别设备偏好、显卡型号和推理状态。"""
from qtpy.QtCore import QSettings, QThreadPool, QTimer, Qt, Signal
from qtpy.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ..core import cellpose4, cellpose_setup
from .worker import Task


class DetectionDeviceBox(QGroupBox):
    inventory_ready = Signal(str)
    warm_message = Signal(str)

    def __init__(self, run_task):
        super().__init__("细胞识别设备")
        self.run_task = run_task
        self.settings = QSettings("cfos", "cfos脑片分析")
        self.mode = QComboBox()
        self.mode.addItem("自动（优先 GPU）", "auto")
        self.mode.addItem("仅 CPU", "cpu")
        pref = self.settings.value("detectionDevice", "auto")
        self.mode.setCurrentIndex(max(0, self.mode.findData(pref)))
        self.preloader = None
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
        self.offline = cellpose_setup.offline_bundle().is_dir()
        if self.offline:
            self.install.setText("安装 / 修复 Cellpose 4 运行环境（离线）")
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
        self.stop_preload()
        cellpose4.set_pref(pref)
        self.preload()
        self.settings.setValue("detectionDevice", pref)
        self._refresh_status()

    def stop_preload(self):
        if self.preloader and self.preloader.isRunning():
            self.preloader.cancel()
            self.preloader.wait()

    def preload(self):
        if self.preloader and self.preloader.isRunning():
            return
        def warm(task):
            if task.is_cancelled():
                return
            self.warm_message.emit("正在加载 Cellpose 4…")
            try:
                cellpose4.probe_device(cancel=task.is_cancelled)
            except Exception as exc:
                if not task.is_cancelled():
                    self.warm_message.emit(f"Cellpose 4 加载未完成 · {exc}")
            else:
                if not task.is_cancelled():
                    self.warm_message.emit("Cellpose 4 模型已就绪")
        self.preloader = Task(warm)
        self.preloader.start()

    def _probe_device(self):
        self.run_task(lambda task: cellpose4.probe_device(cancel=task.is_cancelled),
                      lambda _: self._refresh_status(), label="正在用 Cellpose-SAM 测试推理设备…")

    def offer_offline_setup(self):
        if (self.offline and not cellpose_setup.offline_ready()
                and not self.settings.value("cellposeOfflineSetupOffered", False, type=bool)):
            self.settings.setValue("cellposeOfflineSetupOffered", True)
            self._install_runtime()

    def _install_runtime(self):
        if self.offline:
            question = ("本安装包已包含 Cellpose 4、PyTorch 和模型。现在从本地文件配置并测试识别环境吗？"
                        "\n也可以稍后在「④ 细胞识别」中安装。")
        else:
            question = "将下载 Cellpose 运行环境及模型，首次安装约需数 GB。现在开始安装？"
        if QMessageBox.question(self.window(), "安装 Cellpose 4", question, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes if self.offline else QMessageBox.No) != QMessageBox.Yes:
            return
        self.stop_preload()
        def install(task):
            cellpose4.close()
            return cellpose_setup.install(report=lambda message: task.report(0, 0, message),
                                          cancel=task.is_cancelled, require_offline=self.offline)
        def installed(_):
            self.preload()
            self._refresh_status()
            QMessageBox.information(self, "Cellpose 4", "运行环境已安装。可点“测试 Cellpose 设备”核对推理设备。")
        self.run_task(install, installed, label="正在安装 Cellpose 4 运行环境…")

    def set_busy(self, busy):
        self.mode.setEnabled(not busy)
        self.probe.setEnabled(not busy)
        self.install.setEnabled(not busy)
