"""限制软件使用的 CPU：最多用一半的核（Lucas 2026-11-12 的要求），并合理分配线程。

必须在导入 numpy / TensorFlow 之前调用 limit()，环境变量才会生效。

分配方式（预算 B = 总核数的一半）：
- 后台测量进程：每个只用 1 个线程，最多 6 个，约占预算的一半；
- 主进程（TensorFlow 神经网络、numpy）：用剩下的线程。
32 核电脑：B=16，后台 6 个进程 + 主进程 10 个线程。
4 核电脑：B=2，后台 1 个进程 + 主进程 1 个线程。
"""
from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS")


def total_cores() -> int:
    return os.cpu_count() or 2


def budget() -> int:
    """允许使用的核数：总核数的一半（至少 1）。可用环境变量 CFOS_MAX_CORES 改。"""
    env = os.environ.get("CFOS_MAX_CORES")
    if env:
        return max(1, int(env))
    return max(1, total_cores() // 2)


def n_workers() -> int:
    """后台测量进程数。"""
    env = os.environ.get("CFOS_WORKERS")
    if env is not None:
        return max(0, int(env))
    b = budget()
    if b < 2:
        return 0
    return max(1, min(6, b // 2))


def main_threads() -> int:
    """主进程（神经网络）的线程数。"""
    return max(1, budget() - n_workers())


def limit(main_process: bool = True):
    """设置线程数上限；Windows / Linux 上再把进程绑定到一半的核上（子进程会继承），保证不超过一半。"""
    n = main_threads() if main_process else 1
    for k in _THREAD_VARS:
        os.environ[k] = str(n)
    os.environ["TF_NUM_INTEROP_THREADS"] = "2" if main_process and n >= 4 else "1"
    if main_process and sys.platform != "darwin":
        try:
            import psutil
            p = psutil.Process()
            cores = sorted(p.cpu_affinity())
            b = budget()
            if len(cores) > b:
                p.cpu_affinity(cores[:b])
        except Exception:  # noqa: BLE001
            pass


def describe() -> str:
    return (f"CPU：共 {total_cores()} 核，本软件最多用 {budget()} 核"
            f"（神经网络 {main_threads()} 线程 + 后台测量 {n_workers()} 个进程）")
