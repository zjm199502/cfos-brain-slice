#!/bin/bash
# 补丁更新的端到端测试：模拟安装目录（v0.8）→ 发布 0.8.1 补丁 → 检查 / 下载 / 安装 → 重启后用补丁 → 坏补丁自动停用
# 用法：bash tests/update_e2e.sh <临时文件夹>   （需要 python 能导入 cfos 的依赖；会用 8765 端口起一个本地服务器）
set -e
S=${1:?临时文件夹}
PY=${PY:-python}
T=$S/updtest; rm -rf $T; mkdir -p $T/inst/app $T/pub $T/src
SRC=$(cd "$(dirname "$0")/.." && pwd)
# 安装目录用旧版本的代码（BASE_REF，默认 origin/main 上的 v0.8），检查补丁能把新文件、图标都带过去
(cd $SRC && git archive ${BASE_REF:-origin/main} cfos main.py) | tar -x -C $T/inst/app; ln -s $SRC/resources $T/inst/app/resources
find $T/inst -name __pycache__ -prune -exec rm -rf {} +
export CFOS_UPDATES_DIR=$T/userdata
echo "== 1. 安装版本"; (cd /; $PY $T/inst/app/main.py --version)
# 发布 0.8.1
cp -r $SRC/cfos $SRC/main.py $SRC/packaging $T/src/; find $T/src -name __pycache__ -prune -exec rm -rf {} +
printf -- '- 测试补丁 0.8.1\n' > $T/src/packaging/更新公告/0.8.1.md
$PY $T/src/packaging/make_patch.py $T/pub --min-base 0.7
(cd $T/pub && $PY -m http.server 8765 >/dev/null 2>&1 & echo $! > $T/http.pid); sleep 1
export CFOS_UPDATE_URL=http://127.0.0.1:8765/latest.json
echo "== 2. 检查、下载、安装"
CFOS_BASE_APP=$T/inst/app $PY - <<'PYEOF'
import sys, os; sys.path.insert(0, os.environ["CFOS_BASE_APP"])
from cfos.core import updater
r = updater.check(); print("status", r["status"], r["manifest"]["version"]); assert r["status"] == "patch"
assert "zjm199502@163.com" in r["manifest"]["notes"]
z = updater.download_patch(r, progress=lambda d, t, s: None); print("downloaded", z.name, z.stat().st_size)
info = updater.install_patch(z); print("installed", info["version"])
PYEOF
echo "== 3. 重启后"; (cd /; $PY $T/inst/app/main.py --version) | tee $T/v.txt; grep -q "0.8.1" $T/v.txt; grep -q userdata $T/v.txt
echo "== 3b. 已是最新"
CFOS_BASE_APP=$T/inst/app $PY -c "
import sys,os; sys.path.insert(0, os.environ['CFOS_BASE_APP'])
from cfos.core import updater; print(updater.check()['status'])" | grep -q latest && echo latest-ok
echo "== 3c. 后台进程（spawn）也用补丁代码"
cat > $T/spawn_check.py <<'PYEOF'
import runpy, sys
sys.argv = [sys.argv[1]]
PYEOF
CFOS_BASE_APP=$T/inst/app $PY - $T/inst/app <<'PYEOF'
import sys, os, subprocess
app = sys.argv[1]
code = ("import runpy,sys; sys.argv=['x']; g=runpy.run_path(%r, run_name='__mp_main__'); import cfos; print(cfos.__version__, cfos.__file__)" % (app + "/main.py"))
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/").stdout
print("child:", out.strip()); assert out.startswith("0.8.1")
PYEOF
echo "== 4. 恢复到安装版本"
CFOS_BASE_APP=$T/inst/app $PY -c "
import sys,os; sys.path.insert(0, os.environ['CFOS_BASE_APP'])
from cfos.core import updater; updater.revert_to_base()"
(cd /; $PY $T/inst/app/main.py --version) | grep -q "^v0.8 " && echo revert-ok
echo "== 5. 手动从文件安装（同一个 zip）"
CFOS_BASE_APP=$T/inst/app $PY -c "
import sys,os; sys.path.insert(0, os.environ['CFOS_BASE_APP'])
from cfos.core import updater; print(updater.install_patch('$T/pub/cfos-patch-0.8.1.zip')['version'])"
(cd /; $PY $T/inst/app/main.py --version) | grep -q "0.8.1" && echo manual-ok
echo "== 6. 坏补丁：校验不一致 / 版本太旧 / 需要新的完整包"
CFOS_BASE_APP=$T/inst/app $PY - $T <<'PYEOF'
import sys, os, zipfile, json, shutil
sys.path.insert(0, os.environ["CFOS_BASE_APP"])
from cfos.core import updater
T = sys.argv[1]; src = f"{T}/pub/cfos-patch-0.8.1.zip"
bad = f"{T}/bad.zip"
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(bad, "w") as zo:
    for n in zi.namelist():
        d = zi.read(n)
        if n == "app/cfos/app.py": d += b"\n# tampered\n"
        zo.writestr(n, d)
for f, want in ((bad, "校验"),):
    try: updater.install_patch(f); print("NOT REJECTED"); sys.exit(1)
    except ValueError as e: print("rejected:", str(e)[:40]); assert want in str(e)
# 需要更新的完整包
nb = f"{T}/needfull.zip"
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(nb, "w") as zo:
    for n in zi.namelist():
        d = zi.read(n)
        if n == "patch.json":
            j = json.loads(d); j["min_base"] = "0.9"; d = json.dumps(j).encode()
        zo.writestr(n, d)
try: updater.install_patch(nb); sys.exit(1)
except ValueError as e: print("rejected:", str(e)[:50]); assert "完整安装包" in str(e)
PYEOF
echo "== 7. 补丁启动失败两次 → 自动停用"
P=$CFOS_UPDATES_DIR/0.8.1/app/cfos/app.py
echo 'raise RuntimeError("坏补丁")' >> $P
for i in 1 2; do (cd /; $PY $T/inst/app/main.py --version >/dev/null 2>&1) || echo "start $i failed (expected)"; done
(cd /; $PY $T/inst/app/main.py --version) | tee $T/v3.txt; grep -q "^v0.8 " $T/v3.txt && echo autodisable-ok
cat $CFOS_UPDATES_DIR/current.json
kill $(cat $T/http.pid)
echo ALLOK
