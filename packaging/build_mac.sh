#!/usr/bin/env bash
# 在 Linux 上制作 Mac（Apple M 系列芯片）版程序包（不需要 Mac）。
# 步骤：下载 Mac 版 Python（python-build-standalone，aarch64-apple-darwin）→ 按开发环境的版本下载 macOS arm64 组件 →
#       预装到 python/lib/python3.13/site-packages → 去掉用不到的部分 → 拷贝程序和资源 → 预编译 → 组装 .app → zip。
# 用法：bash packaging/build_mac.sh <工作目录>     需要：zip、开发用 venv（/home/user/venv，Python 3.13）
set -euo pipefail
W=${1:?工作目录}; APP=$(cd "$(dirname "$0")/.." && pwd); PY=/home/user/venv/bin/python; PIP=/home/user/venv/bin/pip
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261020/cpython-3.13.16+20261020-aarch64-apple-darwin-install_only.tar.gz"
NAME="cFos脑片分析"; VER=0.5.1
mkdir -p "$W" && cd "$W"
[ -f py.tar.gz ] || curl -sL -o py.tar.gz "$PY_URL"
# Mac 上用完整版 tensorflow（没有 tensorflow-cpu）
$PIP freeze | grep -viE "^(-e|nvidia|triton|tensorflow|tensorflow_cpu|onnx|tf2onnx|sympy|mpmath|pyflakes|typing==)" > constraints.txt
echo "tensorflow==2.21.0" >> constraints.txt
PLAT=(--platform macosx_15_0_arm64 --python-version 3.13 --implementation cp --abi cp313 --only-binary=:all:)
mkdir -p wheels; : > fail.txt
while read -r r; do $PIP download -q "${PLAT[@]}" --no-deps "$r" -d wheels >/dev/null 2>&1 || echo "$r" >> fail.txt; done < constraints.txt
[ -s fail.txt ] && { echo "以下组件没有 Mac 版："; cat fail.txt; }
# 同一组件既有通用版又有 Mac 专用版时，只留 Mac 专用版
$PY - <<'PYEOF'
import collections, pathlib, re
fs = collections.defaultdict(list)
for f in pathlib.Path("wheels").glob("*.whl"):
    fs[re.sub(r"[-_.]+", "-", f.name.split("-")[0].lower()) + "-" + f.name.split("-")[1]].append(f)
for k, v in fs.items():
    if len(v) > 1:
        for f in v:
            if f.name.endswith("-none-any.whl"):
                f.unlink(); print("去掉重复", f.name)
PYEOF
rm -rf stage && mkdir -p stage && tar -xzf py.tar.gz -C stage
SP=stage/python/lib/python3.13/site-packages
$PIP install -q --no-deps --no-compile --target $SP "${PLAT[@]}" wheels/*.whl
rm -rf $SP/clang $SP/tensorflow/include $SP/PySide6/Qt/lib/QtWebEngine* $SP/PySide6/QtWebEngine* $SP/PySide6/Qt/qml \
       $SP/PySide6/Qt/metatypes $SP/PySide6/examples $SP/pyside6_webengine* $SP/PySide6/*.app \
       $SP/PySide6/Designer.app $SP/PySide6/Assistant.app $SP/PySide6/Linguist.app
find $SP/PySide6 -path "*translations*" -type f ! -name "*zh_CN*" -delete
# 本软件用不到的 Qt 模块（3D、多媒体、PDF、图表、蓝牙等）
for m in 3DAnimation 3DCore 3DExtras 3DInput 3DLogic 3DRender Quick3D Quick3DAssetImport Quick3DAssetUtils \
         Quick3DEffects Quick3DGlslParser Quick3DHelpers Quick3DHelpersImpl Quick3DIblBaker Quick3DParticleEffects \
         Quick3DParticles Quick3DPhysics Quick3DPhysicsHelpers Quick3DRuntimeRender Quick3DSpatialAudio Quick3DUtils \
         Quick3DXr Multimedia MultimediaWidgets MultimediaQuick SpatialAudio Pdf PdfWidgets PdfQuick Designer \
         DesignerComponents Charts ChartsQml DataVisualization DataVisualizationQml Graphs GraphsWidgets Location \
         Positioning PositioningQuick Sensors SensorsQuick RemoteObjects RemoteObjectsQml Scxml ScxmlQml StateMachine \
         StateMachineQml TextToSpeech WebChannel WebChannelQuick WebSockets Bluetooth Nfc SerialPort SerialBus \
         HttpServer WebView WebViewQuick VirtualKeyboard VirtualKeyboardSettings; do
  rm -rf $SP/PySide6/Qt$m.abi3.so $SP/PySide6/Qt$m.pyi $SP/PySide6/Qt/lib/Qt$m.framework
done
rm -rf $SP/PySide6/Qt/lib/libav* $SP/PySide6/Qt/lib/libsw* $SP/PySide6/Qt/plugins/multimedia \
       $SP/PySide6/Qt/plugins/sensors $SP/PySide6/Qt/plugins/position $SP/PySide6/Qt/plugins/sceneparsers \
       $SP/PySide6/Qt/plugins/renderers $SP/PySide6/Qt/plugins/geometryloaders $SP/PySide6/Qt/plugins/designer \
       $SP/PySide6/qmlls $SP/PySide6/qmlformat $SP/PySide6/qmllint $SP/PySide6/lupdate $SP/PySide6/lrelease
# universal2 文件只留 arm64（M 系列芯片）
$PY "$APP/packaging/mac/thin_arm64.py" stage/python
B="stage/$NAME.app/Contents"
mkdir -p "$B/MacOS" "$B/Resources/app/resources"
mv stage/python "$B/Resources/python"
cp -r "$APP"/cfos "$APP"/main.py "$APP"/README.md "$APP"/requirements.txt "$B/Resources/app/"
cp -r "$APP"/resources/{atlas,deepslice,models,demo,teaching} "$B/Resources/app/resources/"
cp "$APP"/packaging/mac/Info.plist "$B/"
cp "$APP"/packaging/mac/cfos.icns "$B/Resources/"
cp "$APP"/packaging/mac/launcher.sh "$B/MacOS/cfos"
cp "$APP"/packaging/mac/诊断启动.command stage/
chmod +x "$B/MacOS/cfos" stage/诊断启动.command
printf "APPL????" > "$B/PkgInfo"
find stage -name "__pycache__" -type d -prune -exec rm -rf {} +
# 不预编译 .pyc（会让压缩包大 100 多 MB）；第一次启动时 Python 自己编译并保存，以后启动就快了
cat > stage/安装说明.txt <<'EOT'
1. 把“cFos脑片分析”拖到“应用程序”文件夹。
2. 第一次打开：双击后如果提示“无法验证开发者”，点“完成”，再打开“系统设置 → 隐私与安全性”，
   在页面下方找到“仍要打开”，点它，输入开机密码确认。以后就可以直接双击打开。
3. 如果打不开，双击“诊断启动.command”，把终端窗口里的文字截图发给开发者。
EOT
rm -f "${NAME}_Mac版_v${VER}.zip"
(cd stage && zip -qry -9 "../${NAME}_Mac版_v${VER}.zip" "$NAME.app" 诊断启动.command 安装说明.txt)
ls -la "${NAME}_Mac版_v${VER}.zip"
