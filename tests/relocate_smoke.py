"""重新定位原图：整体搬家自动找回、换电脑按文件名搜索、同名文件区分。用法：python tests/relocate_smoke.py <项目> <临时目录>"""
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfos.core.project import Project
from cfos.core import relocate as RL

src, tmp = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(tmp, ignore_errors=True)
p0 = Project.load(src); img = Path(p0.images[0].path).resolve(); name = img.name
# 1. 项目和图片放在一起，整体搬家
A = tmp / "A"; (A / "原图").mkdir(parents=True)
os.symlink(img, A / "原图" / name)
shutil.copytree(src, A / "proj", ignore=shutil.ignore_patterns("*.xlsx", "细胞mask*", "质控图*"))
p = Project.load(A / "proj"); p.images[0].path = str(A / "原图" / name); p.save()
print("rel =", p.images[0].rel, "size =", p.images[0].size)
shutil.move(str(A), str(tmp / "B"))
p = Project.load(tmp / "B" / "proj")
print("整体搬家后自动找回:", p.relocated, p.images[0].path); assert p.relocated == 1 and Path(p.images[0].path).exists()
# 2. 换电脑：路径是 Windows 盘符，按文件名搜索；有一个同名但大小不同的干扰文件
d = json.loads((tmp / "B" / "proj" / "cfos_project.json").read_text(encoding="utf-8"))
d["images"][0]["path"] = "D:\\实验数据\\批次1\\" + name; d["images"][0]["rel"] = None
(tmp / "B" / "proj" / "cfos_project.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
p = Project.load(tmp / "B" / "proj"); assert p.relocated == 0 and len(RL.missing(p)) == 1
disk = tmp / "新电脑E盘"; (disk / "备份" / "旧").mkdir(parents=True); (disk / "数据" / "批次1").mkdir(parents=True)
(disk / "备份" / "旧" / name).write_bytes(b"not the same file")
os.symlink(img, disk / "数据" / "批次1" / name)
res = RL.relocate(p, disk); print(RL.summary_text(res))
assert len(res["fixed"]) == 1 and res["fixed"][0][1].endswith(str(Path("数据") / "批次1" / name))
p = Project.load(tmp / "B" / "proj"); assert not RL.missing(p) and p.load_cells(p.images[0]) is not None
# 3. 两个同名同大小的文件：上一级文件夹名相同的优先；都不同则报告无法确定
d = json.loads((tmp / "B" / "proj" / "cfos_project.json").read_text(encoding="utf-8"))
d["images"][0]["path"] = "D:\\x\\批次1\\" + name
(tmp / "B" / "proj" / "cfos_project.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
(disk / "副本").mkdir(); os.symlink(img, disk / "副本" / name)
p = Project.load(tmp / "B" / "proj")
p.images[0].rel = None; p.images[0].path = "D:\\x\\批次1\\" + name
res = RL.relocate(p, disk); print("同名优先上级文件夹:", res["fixed"]); assert "批次1" in res["fixed"][0][1]
p.images[0].path = "D:\\y\\别的\\" + name
res = RL.relocate(p, disk); print(RL.summary_text(res)); assert len(res["ambiguous"]) == 1
print("OK")
