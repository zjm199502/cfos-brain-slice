"""软件主程序（由 main.py 启动）。放在 cfos 包里，这样补丁包更新代码时启动流程也能一起更新。"""
import os
import sys

from cfos import APP_NAME

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "assets")


def app_icon():
    """程序图标（FosMap 活动热图点阵）：小尺寸用点更少的简化版，缩小后不糊。"""
    from qtpy.QtGui import QIcon
    ic = QIcon()
    for n in (16, 32, 64, 128, 256):
        f = os.path.join(ASSETS, f"fosmap_{n}.png")
        if os.path.exists(f):
            ic.addFile(f)
    return ic


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
    logo = os.path.join(ASSETS, "fosmap_128.png").replace("\\", "/")
    img = f"<img src='{logo}' width='96' height='96'><br>" if os.path.exists(logo) else ""
    lb = QLabel(f"{img}<b style='font-size:20px'>{APP_NAME}</b><br><br>正在启动，请稍候…<br>（第一次启动可能需要半分钟）")
    lb.setTextFormat(Qt.RichText)
    lb.setAlignment(Qt.AlignCenter)
    lb.setWindowFlags(Qt.SplashScreen | Qt.WindowStaysOnTopHint)
    lb.setStyleSheet("background:#262930; color:white; font-size:16px; padding:30px; border:1px solid #4a7ab8;")
    lb.setWindowIcon(app_icon()); lb.resize(400, 300); lb.show(); app.processEvents()
    return lb


def selftest_cli():
    """命令行自检：main.py --selftest。结果打印出来并写到日志文件夹的“自检结果.txt”。退出码 0 = 全部正常。"""
    from cfos.core.log import setup_logging
    setup_logging()
    from cfos.core import selftest
    ok, text, _ = selftest.run(lambda i, n, t: print(t, flush=True))
    print(text, flush=True)
    sys.exit(0 if ok else 1)


def main():
    if "--version" in sys.argv:          # 诊断用：打印实际运行的版本和代码位置
        import cfos
        from cfos.core import updater
        updater.mark_started_ok()
        print(updater.version_text(), cfos.__file__, flush=True)
        sys.exit(0)
    if "--selftest" in sys.argv:
        selftest_cli()
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
            QMessageBox.critical(None, APP_NAME,
                                 f"{friendly(e)}\n\n软件没有关闭，已保存的结果不受影响。\n"
                                 f"详细记录在：{log_dir()}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = excepthook
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    from cfos.gui.i18n import install_chinese
    install_chinese(app)
    from cfos.gui.main_window import MainWindow

    w = MainWindow()
    w.setWindowIcon(app_icon())
    w.show()
    splash.close()
    _after_start(w)
    if len(sys.argv) > 1:
        from cfos.core.project import Project
        if Project.is_project(sys.argv[1]):
            w._load_project(Project.load(sys.argv[1]))
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())



def _after_start(w):
    """主窗口打开以后：记录本次启动成功（补丁不会被自动停用）；告知补丁被停用；后台检查更新。"""
    try:
        from cfos.core import updater
        updater.mark_started_ok()
        msg = os.environ.get("CFOS_PATCH_DISABLED")
        if msg:
            w.info(msg + "。\n现在使用的是安装时的版本。\n请点“① 项目与图片 → 导出诊断包”，把 zip 文件发给开发者。")
        if updater.auto_check_enabled() and "--no-update-check" not in sys.argv:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(4000, lambda: w.check_updates(silent=True))
    except Exception:  # noqa: BLE001
        from cfos.core.log import log
        import traceback
        log.warning("更新检查出错\n%s", traceback.format_exc())
