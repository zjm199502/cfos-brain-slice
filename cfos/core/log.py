"""运行日志：所有错误的详细信息写到日志文件，界面上只显示简短的中文说明。"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger("cfos")
_project_handler = None


def log_dir() -> Path:
    import sys
    if sys.platform == "darwin":
        d = Path(os.path.expanduser("~/Library/Logs")) / "cfos脑片分析"
    else:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        d = Path(base) / "cfos脑片分析" / "日志"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging():
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        h = RotatingFileHandler(log_dir() / "运行日志.txt", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        h.setFormatter(fmt)
        log.addHandler(h)
    except OSError:
        pass
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt)
    log.addHandler(sh)


def attach_project(folder):
    """项目打开后，日志同时写一份到项目文件夹，方便把整个文件夹发给别人排查问题。"""
    global _project_handler
    setup_logging()
    if _project_handler is not None:
        log.removeHandler(_project_handler); _project_handler.close(); _project_handler = None
    try:
        h = RotatingFileHandler(Path(folder) / "运行日志.txt", maxBytes=5_000_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h); _project_handler = h
    except OSError:
        pass


def friendly(e: BaseException) -> str:
    """把常见的异常翻译成学生能看懂的中文说明。"""
    if isinstance(e, MemoryError):
        return "内存不足。请关闭其他程序后重试；或在“④ 细胞识别”中关闭去卷积。"
    if isinstance(e, FileNotFoundError):
        return f"找不到文件：{e.filename or e}。图片可能被移动、改名，或移动硬盘没有插上。"
    if isinstance(e, PermissionError):
        return f"没有写入权限：{e.filename or e}。如果 Excel 文件正在打开，请先关闭它。"
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return "磁盘空间不足，请清理磁盘后重试。"
    msg = str(e)
    if any(ch > "一" for ch in msg):  # 本软件自己抛出的中文说明
        return msg
    return f"程序出错（{type(e).__name__}: {msg[:200]}）。"
