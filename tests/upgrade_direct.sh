#!/bin/bash
# 跨版本直接升级测试：安装目录是旧版本（v0.7 / v0.8 / v0.8 + 已装 0.8.1 补丁），发布当前代码的补丁，
# 检查更新后一步升到最新版，不需要先装中间版本。
# 用法：bash tests/upgrade_direct.sh <临时文件夹> [已发布的 0.8.1 补丁 zip]
set -e
S=${1:?临时文件夹}; P081=${2:-}
PY=${PY:-python}
SRC=$(cd "$(dirname "$0")/.." && pwd)
T=$S/upgtest; rm -rf $T; mkdir -p $T/pub $T/src
VER=$($PY -c "import sys; sys.path.insert(0, '$SRC'); import cfos; print(cfos.__version__)")
cp -r $SRC/cfos $SRC/main.py $SRC/packaging $T/src/; find $T/src -name __pycache__ -prune -exec rm -rf {} +
$PY $T/src/packaging/make_patch.py $T/pub --min-base 0.7 --notes $T/src/packaging/更新公告/$VER.md >/dev/null
(cd $T/pub && $PY -m http.server 8766 >/dev/null 2>&1 & echo $! > $T/http.pid); sleep 1
trap 'kill $(cat $T/http.pid) 2>/dev/null' EXIT
export CFOS_UPDATE_URL=http://127.0.0.1:8766/latest.json

run_case() {   # $1 名称  $2 安装目录代码的提交  $3 先装的补丁（可空）
  local D=$T/$1; mkdir -p $D/inst/app
  (cd $SRC && git archive $2 cfos main.py) | tar -x -C $D/inst/app; ln -s $SRC/resources $D/inst/app/resources
  export CFOS_UPDATES_DIR=$D/userdata
  if [ -n "$3" ]; then
    CFOS_BASE_APP=$D/inst/app $PY -c "import sys,os; sys.path.insert(0, os.environ['CFOS_BASE_APP'])
from cfos.core import updater; print('先装', updater.install_patch('$3')['version'])"
  fi
  local before=$(cd /; $PY $D/inst/app/main.py --version 2>/dev/null | tail -1)
  CFOS_BASE_APP=$D/inst/app $PY - <<'PYEOF'
import sys, os; sys.path.insert(0, os.environ["CFOS_BASE_APP"])
from cfos.core import updater
r = updater.check(); assert r["status"] == "patch", r["status"]
z = updater.download_patch(r, progress=lambda d, t, s: None)
print("一步升级到", updater.install_patch(z)["version"])
PYEOF
  local after=$(cd /; $PY $D/inst/app/main.py --version 2>/dev/null | tail -1)
  echo "[$1] 升级前：$before → 升级后：$after"
  echo "$after" | grep -q "$VER"
}
# v0.7 没有自动更新功能（只能用补丁里的手动安装脚本），这里不测
run_case v0.8 a0ece40 ""
[ -n "$P081" ] && run_case v0.8+0.8.1补丁 a0ece40 "$P081"
echo "ALL-OK $VER"
