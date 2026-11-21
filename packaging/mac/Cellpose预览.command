#!/bin/bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"

fail() {
  echo "$1"
  read -r -p "按回车键关闭。" _ || true
  exit 1
}

app="${1:-}"
if [[ -z "$app" ]]; then
  for folder in /Applications "$HOME/Applications" "$here" "$here/.."; do
    for name in "FosFolio脑片图册" "FosFolio 脑片图册" "cFos脑片分析"; do
      if [[ -x "$folder/$name.app/Contents/Resources/python/bin/python3" ]]; then
        app="$folder/$name.app"
        break 2
      fi
    done
  done
fi
base="$app/Contents/Resources"
python="$base/python/bin/python3"
[[ -x "$python" && -f "$base/app/main.py" ]] || fail "请把已安装的 FosFolio.app 放到“应用程序”，或将本预览文件夹与它放在一起。"
[[ "$("$python" -I -c 'import platform; print(platform.machine())')" == arm64 ]] || fail "此预览需要原生 arm64 版 FosFolio。"

export PYTHONNOUSERSITE=1
export CFOS_UPDATES_DIR="$here/updates-cellpose4"
export CFOS_COMPONENTS_DIR="$HOME/Library/Application Support/cfos脑片分析/组件"
unset PYTHONPATH PYTHONHOME
code="$("$python" "$base/app/main.py" --version)"
[[ "$code" == *"$CFOS_UPDATES_DIR/"* ]] || fail "已有软件未能加载本地预览。检测结果为 $code"
exec "$python" "$base/app/main.py" --no-update-check
