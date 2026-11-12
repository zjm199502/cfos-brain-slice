#!/bin/bash
# cFos 脑片分析（Mac）启动脚本：用程序包里自带的 Python 运行主程序。
R="$(cd "$(dirname "$0")/../Resources" && pwd)"
export PYTHONNOUSERSITE=1
exec "$R/python/bin/python3" "$R/app/main.py" "$@"
