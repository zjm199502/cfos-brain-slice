"""定位随软件打包的资源（模型、图谱）。开发时回退到用户目录缓存。"""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # 用补丁运行时，代码在用户目录的“更新”文件夹里，资源（模型、图谱）仍在安装目录（main.py 设置 CFOS_BASE_APP）
    import os
    if os.environ.get("CFOS_BASE_APP"):
        return Path(os.environ["CFOS_BASE_APP"])
    return Path(__file__).resolve().parents[2]


def resources_dir() -> Path:
    return app_root() / "resources"


def stardist_model_dir() -> Path | None:
    d = resources_dir() / "models" / "2D_versatile_fluo"
    return d if (d / "config.json").exists() else None
