#!/bin/bash
# 软件打不开时双击运行：在终端里启动，并显示出错信息。请把窗口里的文字截图发给开发者。
H="$(cd "$(dirname "$0")" && pwd)"
for A in "$H/FosMap脑片活动图谱.app" "/Applications/FosMap脑片活动图谱.app" "$H/cFos脑片分析.app" "/Applications/cFos脑片分析.app"; do
  [ -d "$A" ] && break
done
echo "程序位置：$A"
R="$A/Contents/Resources"
export PYTHONNOUSERSITE=1
"$R/python/bin/python3" -X faulthandler -c "import sys; print(sys.version)"
"$R/python/bin/python3" -X faulthandler "$R/app/main.py" "$@"
echo; echo "程序已退出。按回车键关闭窗口。"; read
