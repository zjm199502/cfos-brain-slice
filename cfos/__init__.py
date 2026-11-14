"""FosMap 脑片活动图谱（原名 cFos 脑片分析）。

TensorFlow 的 oneDNN 优化在某些 Windows 多核电脑上会让 CPU 计算慢几百倍
（2026-11-12 在一台 32 核 Windows 电脑上实测：一块图从 1 秒变成 5 分钟算不完）。
必须在导入 TensorFlow 之前关掉，所以放在包的最顶层。"""
import os as _os

APP_NAME = "FosMap 脑片活动图谱"   # 显示名称；安装目录、用户数据目录仍沿用 cfos脑片分析，不受改名影响

_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "0.8.2"
