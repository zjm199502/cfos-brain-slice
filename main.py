"""cFos 脑片分析软件入口。"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # 见 cfos/__init__.py

# 最多用一半的 CPU 核，并分配好线程（必须在导入 numpy / TensorFlow 之前）。
# 后台测量进程也会导入本文件（多进程 spawn），它们只用 1 个线程。
if __name__ == "__main__":
    from cfos.core.cpu import limit as _limit_cpu
    _limit_cpu(main_process=True)

# 用 pythonw（无控制台窗口）启动时 sys.stdout / sys.stderr 是 None，
# 一些组件（进度条、DeepSlice 的打印）往里写会直接报错，所以指向空设备。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _raise_priority():
    """Windows 上把软件（以及它启动的后台计算进程，会继承）设为“高于正常”优先级。
    不用“高”或“实时”：满载时会让鼠标键盘和系统卡住。"""
    if sys.platform != "win32":
        return
    try:
        import psutil
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def _splash():
    """第一次启动要加载深度学习组件，可能要十几秒；先显示一个小窗口，免得以为没反应又点一次。"""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(sys.argv)
    lb = QLabel("cFos 脑片分析\n\n正在启动，请稍候…\n（第一次启动可能需要半分钟）")
    lb.setAlignment(Qt.AlignCenter)
    lb.setWindowFlags(Qt.SplashScreen | Qt.WindowStaysOnTopHint)
    lb.setStyleSheet("background:#262930; color:white; font-size:16px; padding:30px; border:1px solid #4a7ab8;")
    lb.resize(380, 180); lb.show(); app.processEvents()
    return lb


def main():
    _raise_priority()
    splash = _splash()
    from napari.qt import get_qapp

    from cfos.core.log import setup_logging, log, log_dir
    setup_logging()
    app = get_qapp()

    def excepthook(t, e, tb):
        import traceback
        log.error("未处理的错误\n%s", "".join(traceback.format_exception(t, e, tb)))
        try:
            from qtpy.QtWidgets import QMessageBox
            from cfos.core.log import friendly
            QMessageBox.critical(None, "cFos 脑片分析",
                                 f"{friendly(e)}\n\n软件没有关闭，已保存的结果不受影响。\n"
                                 f"详细记录在：{log_dir()}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = excepthook
    app.setApplicationName("cFos 脑片分析")
    from cfos.gui.i18n import install_chinese
    install_chinese(app)
    from cfos.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    splash.close()
    if len(sys.argv) > 1:
        from cfos.core.project import Project
        if Project.is_project(sys.argv[1]):
            w._load_project(Project.load(sys.argv[1]))
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()
