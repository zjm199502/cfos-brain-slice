"""生成补丁包和更新公告，放到 installers 分支的 updates/ 文件夹里发布。

用法：python packaging/make_patch.py <输出文件夹> --min-base 0.9.0 [--notes packaging/更新公告/1.0.0.md]

输出：
- cfos-patch-<版本>.zip：程序代码和界面图片（app/cfos、app/main.py）+ patch.json（版本、所需完整安装包版本、每个文件的 SHA-256）
  + 图谱、原生辅助工具、随附许可证和软件内补丁安装说明
- latest.json：客户端检查更新时读取（版本、更新公告、补丁文件名、大小、SHA-256）
- 更新公告.md：所有版本的更新公告（新的在最前面）

min_base：需要的完整安装包最低版本。只改了 cfos 代码时保持不变；改了 Python 组件、模型、图谱、
安装程序时必须提高到这个版本（用户会被提示下载完整安装包）。
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from cfos.core import updater  # noqa: E402

# Include the application-owned data and native helpers used by both platforms.
# The runtime, model weights and installer resources are outside cfos/.
PATCH_RESOURCES = ("cfos/core/ccf_data/", "cfos/core/stam_data/",
                   "cfos/core/giraff_runtime/", "cfos/mac_plugins/")

CONTACT_MD = f"联系方式：zjm199502@163.com，{updater.SOURCE_URL}"

README = """FosFolio v{VERSION} 软件内补丁
兼容基础安装版本为 v{MIN_BASE} 起的指定系列。请先保存项目。
打开「帮助 → 检查更新」，选择「现在更新」，安装后重启软件。
如果已手动下载此 ZIP，请从帮助菜单选择「从文件安装补丁」，直接选择 ZIP。
{CONTACT}
"""


def _zinfo(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=dt.datetime.now().timetuple()[:6])
    zi.external_attr = (mode | 0o100000) << 16
    zi.compress_type = zipfile.ZIP_DEFLATED
    return zi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--min-base", required=True, help="需要的完整安装包最低版本，例如 0.7")
    ap.add_argument("--notes", help="这个版本的更新公告（Markdown）")
    a = ap.parse_args()
    version = updater._read_version_file(APP / "cfos" / "__init__.py")
    notes_file = Path(a.notes) if a.notes else APP / "packaging" / "更新公告" / f"{version}.md"
    notes = notes_file.read_text(encoding="utf-8").strip()
    if CONTACT_MD not in notes:
        notes += "\n\n" + CONTACT_MD
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    files = {}
    for f in sorted((APP / "cfos").rglob("*")):
        if "__pycache__" in f.parts or not f.is_file():
            continue
        rel = f.relative_to(APP).as_posix()
        if f.suffix in {".py", ".png", ".svg"} or f.name == "LICENSE.txt" or rel.startswith(PATCH_RESOURCES):
            files["app/" + rel] = f.read_bytes()
    files["app/main.py"] = (APP / "main.py").read_bytes()
    info = {"version": version, "min_base": a.min_base, "date": dt.date.today().isoformat(), "notes": notes,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    zname = f"cfos-patch-{version}.zip"
    zp = out / zname
    fill = lambda t: t.replace("{VERSION}", version).replace("{MIN_BASE}", a.min_base).replace("{CONTACT}", CONTACT_MD)
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(_zinfo("patch.json"), json.dumps(info, ensure_ascii=False, indent=1))
        for k, v in files.items():
            z.writestr(_zinfo(k), v)
        z.writestr(_zinfo("安装说明.txt"), fill(README).replace("\n", "\r\n"))
        z.writestr(_zinfo("更新公告.md"), notes)
    data = zp.read_bytes()
    # 检查更新时显示的公告：这一版 + 之前的小更新（补丁是整套代码，用户可能跳过了中间几版，
    # 例如 0.8 直接升到 0.8.2，也要能看到 0.8.1 改了什么）。只列比 min_base 新的版本。
    hist = out / "更新公告.md"
    earlier = []
    if hist.exists():
        import re
        txt = hist.read_text(encoding="utf-8")
        for m in re.finditer(r"^## v([0-9.]+)（[^）]*）\n(.*?)(?=^## v|\Z)", txt, re.S | re.M):
            v, body = m.group(1), m.group(2).strip()
            if updater.vtuple(a.min_base) < updater.vtuple(v) < updater.vtuple(version):
                body = body.replace(CONTACT_MD, "").strip()
                body = re.sub(r"^# .*\n*", "", body, flags=re.M)        # 每段自己的大标题（“# v0.8.1 更新内容”）不要
                body = re.sub(r"^## ", "#### ", body, flags=re.M)
                earlier.append(f"### v{v}\n\n{body}")
    shown = notes if not earlier else (
        notes.replace(CONTACT_MD, "").strip() + "\n\n## 之前的小更新（已经装过这些版本的可以跳过）\n\n"
        + "\n\n".join(earlier) + "\n\n" + CONTACT_MD)
    latest = {"version": version, "min_base": a.min_base, "date": info["date"], "notes": shown,
              "patch": {"file": zname, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
              "full_installer": updater.FULL_PAGE}
    (out / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")
    # 累积的更新公告（新的在最前面）
    hist = out / "更新公告.md"
    old = hist.read_text(encoding="utf-8") if hist.exists() else "# FosFolio 脑片图册 更新公告\n"
    head, _, rest = old.partition("\n")
    marker = f"## v{version}（"
    if marker in rest:          # 重新发布同一版本：替换旧的那一段
        i = rest.index(marker); j = rest.find("\n## v", i + 1)
        rest = rest[:i] + rest[j + 1:] if j >= 0 else rest[:i]
    section = f"## v{version}（{info['date']}）\n\n{notes}\n"
    hist.write_text(f"{head}\n\n{section}\n{rest.strip()}\n", encoding="utf-8")
    print(f"补丁 {zname}：{len(data) / 1e6:.2f} MB，{len(files)} 个文件，需要完整安装包 ≥ v{a.min_base}")


if __name__ == "__main__":
    main()
