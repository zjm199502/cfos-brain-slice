#!/usr/bin/env bash
# 在 Linux 上制作 Windows 离线安装包（不需要 Windows 机器）。
# 步骤：下载 Windows 版 Python（python-build-standalone）→ 按开发环境的版本下载 win_amd64 组件 →
#       预装到 python/Lib/site-packages → 去掉用不到的部分 → 拷贝程序和资源 → NSIS 编译。
# 用法：bash packaging/build_windows.sh <工作目录>     需要：makensis、开发用 venv（/home/user/venv）
set -euo pipefail
W=${1:?工作目录}; APP=$(cd "$(dirname "$0")/.." && pwd); PIP=/home/user/venv/bin/pip
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261020/cpython-3.13.16+20261020-x86_64-pc-windows-msvc-install_only.tar.gz"
mkdir -p "$W" && cd "$W"
[ -f py.tar.gz ] || curl -sL -o py.tar.gz "$PY_URL"
$PIP freeze | grep -viE "^(-e|nvidia|triton|tensorflow==|onnx|tf2onnx|sympy|mpmath|pyflakes|typing==)" > constraints.txt
grep -vE "^tensorflow; " "$APP/requirements.txt" | sed 's/; sys_platform.*//' > req_win.txt
PLAT=(--platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 --only-binary=:all:)
$PIP download "${PLAT[@]}" -r req_win.txt -c constraints.txt -d wheels
while read -r r; do $PIP download -q "${PLAT[@]}" --no-deps "$r" -d wheels || echo "跳过 $r"; done < constraints.txt
rm -f wheels/typing-3*.whl
rm -rf stage && mkdir -p stage && tar -xzf py.tar.gz -C stage
find stage/python -name "*.pdb" -delete
$PIP install -q --no-deps --no-compile --target stage/python/Lib/site-packages "${PLAT[@]}" wheels/*.whl
SP=stage/python/Lib/site-packages
rm -rf $SP/clang $SP/tensorflow/include $SP/include $SP/PySide6/Qt6WebEngine* $SP/PySide6/QtWebEngine* $SP/PySide6/resources \
       $SP/PySide6/qml $SP/PySide6/metatypes $SP/PySide6/*.exe $SP/PySide6/examples $SP/pyside6_webengine*
find $SP/PySide6/translations -type f ! -name "*zh_CN*" -delete
mkdir -p stage/app/resources
cp -r "$APP"/cfos "$APP"/main.py "$APP"/README.md "$APP"/requirements.txt stage/app/
cp -r "$APP"/resources/{atlas,deepslice,models,demo,teaching} stage/app/resources/
cp "$APP"/packaging/cfos.ico stage/app/
cp "$APP"/packaging/诊断启动（出问题时用）.bat stage/
find stage -name "__pycache__" -type d -prune -exec rm -rf {} +
# 注意：NSIS 的数据块不能超过 2 GB，所以不能用 /SOLID 压缩
LANG=C.UTF-8 LC_ALL=C.UTF-8 makensis -V2 -INPUTCHARSET UTF8 "-DSTAGE=$PWD/stage" "-DOUTFILE=$PWD/cFos脑片分析_v0.5.1_Windows安装包.exe" "$APP/packaging/installer.nsi"
