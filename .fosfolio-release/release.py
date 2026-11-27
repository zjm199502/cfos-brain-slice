"""Reproduce the validated ZIP, verify its runtime installation, then stage the four update feeds."""
from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, tempfile, zipfile, zlib

base = "512c2fd0bf9a0136cf3dda587171cecff6f0b6e6"
target = "release/fosfolio-0.9.22"
version = "0.9.22"
name = f"cfos-patch-{version}.zip"
expected = "a9cb372bb7c24e70e15cfa93cbec1e231bdf5d92fed21711a5cf7f873536930d"
channels = ("updates", "updates/v0.9", "updates/v0.9/macos", "updates/v0.9/windows")
notes = """FosFolio 0.9.22 累计更新。安装后重启软件。

• 在「脑片拆分」或「细胞识别」中自由圈选 ROI，立即查看各通道细胞数及共标去重计数，并导出 Excel。圈选线宽固定为 1 pt。
• 九个步骤共享任务进度与日志，显示运行时间和可估算的剩余时间。
• 「拖动脑片」可用鼠标旋转、平移整幅视图。「脑图谱配准」中的「拖动整张图谱」可调整图谱相对脑片的位置。右键框住白色轮廓，拖框内平移、拖上方圆柄旋转、拖角点等比缩放。局部修正保存在所选图谱中，并用于细胞脑区归属及配准导出。
• Esc 退出当前工具并返回拖图；尚未松开鼠标的变换会取消。"""

def git(*args, data=None):
    return subprocess.check_output(["git", *args], input=data)
def sha(data):
    return hashlib.sha256(data).hexdigest()
def read(path):
    return git("show", base+":"+path)
def text(obj):
    return (json.dumps(obj,ensure_ascii=False,indent=1)+"\n").encode()

git("fetch","--filter=blob:none","--depth=1","origin",target)
if git("rev-parse","FETCH_HEAD").decode().strip()!=base:
    raise RuntimeError("The release branch has moved")
old = read("updates/v0.9/cfos-patch-0.9.19.zip")
spec = json.loads(zlib.decompress(base64.b64decode(Path(__file__).with_name("patch.delta").read_bytes())))
assert sha(old)==spec["previous"]
patch = b"".join(base64.b64decode(c) if isinstance(c,str) else old[c[0]:c[0]+c[1]] for c in spec["chunks"])
assert sha(patch)==expected==spec["result"]
with zipfile.ZipFile(io.BytesIO(patch)) as z:
    manifest=json.loads(z.read("patch.json"))
    assert manifest["version"]==version and manifest["min_base"]=="0.9.0"
    assert set(z.namelist())=={"patch.json",*manifest["files"]}
    for p,h in manifest["files"].items():
        data=z.read(p)
        assert sha(data)==h,p
        if p.endswith(".py"):
            compile(data,p,"exec")
    with tempfile.TemporaryDirectory() as folder:
        tmp=Path(folder)
        (tmp/name).write_bytes(patch)
        z.extractall(tmp/"source")
        sys.path.insert(0,str(tmp/"source/app"))
        from cfos.core import updater
        (tmp/"base/cfos").mkdir(parents=True)
        (tmp/"base/cfos/__init__.py").write_text('__version__ = "0.9.0"\n')
        os.environ.update(CFOS_UPDATES_DIR=str(tmp/"installed"),CFOS_BASE_APP=str(tmp/"base"))
        updater.install_patch(tmp/name)
        assert updater.current_version()==version
print("Verified and installed",name,expected,flush=True)
if "--verify-only" in sys.argv:
    sys.exit(0)

files={}
for channel in channels:
    path=channel+"/latest.json"
    meta=json.loads(read(path))
    assert meta["version"]=="0.9.19"
    meta.update(version=version,date="2026-11-27",notes=notes,patch=dict(file=name,size=len(patch),sha256=expected))
    files[path]=text(meta)
    files[channel+"/"+name]=patch
    path=channel+"/更新公告.md"
    files[path]=("## 2026-11-27 · FosFolio 0.9.22\n\n"+notes+"\n\n").encode()+read(path)
for path in ("README.md","updates/README.md","updates/v0.8/README.md"):
    content=read(path).decode()
    lines=content.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if line.startswith("## 当前版本") or "累计补丁](" in line:
            lines[i]=line.replace("0.9.19","0.9.22")
    files[path]="".join(lines).encode()
with tempfile.TemporaryDirectory() as folder:
    os.environ["GIT_INDEX_FILE"]=str(Path(folder)/"index")
    git("-c","core.sparseCheckout=false","read-tree",base)
    same={}
    for p,data in files.items():
        digest=sha(data)
        if digest not in same:
            same[digest]=git("hash-object","-w","--stdin",data=data).decode().strip()
        git("update-index","--add","--cacheinfo","100644",same[digest],p)
    tree=git("write-tree").decode().strip()
commit=git("-c","user.name=github-actions[bot]","-c","user.email=41898282+github-actions[bot]@users.noreply.github.com",
           "commit-tree",tree,"-p",base,"-m","发布 FosFolio 0.9.22 双端累计更新").decode().strip()
git("push","origin",commit+":refs/heads/"+target)
print(commit,flush=True)
