"""cFos 脑片分析软件入口（启动器）。

这个文件尽量保持简单、以后不再改动：真正的启动流程在 cfos/app.py。
有补丁（软件更新）时，先把补丁的代码放到最前面（见 cfos/core/updater.py），再导入 cfos。"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # 见 cfos/__init__.py

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["CFOS_BASE_APP"] = _HERE                   # 安装目录（资源文件在这里）


def _boot_patch(main_process: bool):
    """补丁：直接按文件路径执行安装目录里的 updater.py（只用标准库），决定用哪一份程序代码。
    后台计算进程（多进程 spawn 会重新执行本文件）也走这里，保证和主进程用同一份代码。"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_cfos_boot_updater", os.path.join(_HERE, "cfos", "core", "updater.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.boot_path(main_process)
    except Exception:  # noqa: BLE001  更新机制出问题时照常用安装时的版本
        return None


_boot_patch(__name__ == "__main__")

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


if __name__ == "__main__":
    from cfos.app import main
    main()
