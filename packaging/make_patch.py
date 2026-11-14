"""生成补丁包和更新公告，放到 installers 分支的 updates/ 文件夹里发布。

用法：python packaging/make_patch.py <输出文件夹> --min-base 0.8 [--notes packaging/更新公告/0.8.md]

输出：
- cfos-patch-<版本>.zip：程序代码和界面图片（app/cfos、app/main.py）+ patch.json（版本、所需完整安装包版本、每个文件的 SHA-256）
  + 旧版本手动安装脚本（Windows .bat / Mac .command）+ 安装说明.txt
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

PATCH_SUFFIXES = {".py", ".png", ".svg"}   # 程序代码和界面图片（图标、logo）

CONTACT_MD = f"联系方式：zjm199502@163.com，{updater.SOURCE_URL}"

WIN_BAT = r"""@echo off
chcp 65001 >nul
setlocal
set "SRC=%~dp0app"
set "DST=%LOCALAPPDATA%\cFos脑片分析\app"
if not exist "%SRC%\cfos\__init__.py" (
  echo 请先把补丁 zip 解压（右键 → 全部解压），再双击解压出来的这个文件。
  pause
  exit /b 1
)
if not exist "%DST%\main.py" (
  echo 没有在默认位置找到已安装的 FosMap 脑片活动图谱（原名 cFos 脑片分析）：
  echo   %DST%
  echo 请把补丁里 app 文件夹中的全部内容，复制到你的安装目录下的 app 文件夹里（覆盖同名文件）。
  pause
  exit /b 1
)
echo 将安装补丁 v{VERSION} 到：%DST%
echo 请先关闭 FosMap 脑片活动图谱（原名 cFos 脑片分析），然后按任意键开始…
pause >nul
xcopy "%SRC%" "%DST%" /E /Y /I /Q >nul
if errorlevel 1 (
  echo 复制失败。请确认软件已经关闭后重试。
  pause
  exit /b 1
)
echo.
echo 补丁已安装，可以打开软件了。以后的更新可以在软件里“帮助 → 检查更新”自动完成。
pause
"""

MAC_CMD = """#!/bin/bash
# FosMap 脑片活动图谱（原名 cFos 脑片分析）补丁 v{VERSION}：手动安装到已安装的程序里（v0.7 等没有自动更新功能的旧版本用）
D="$(cd "$(dirname "$0")" && pwd)"
for A in "/Applications/FosMap脑片活动图谱.app" "$HOME/Applications/FosMap脑片活动图谱.app" \
         "/Applications/cFos脑片分析.app" "$HOME/Applications/cFos脑片分析.app"; do
  [ -d "$A" ] && break
done
T="$A/Contents/Resources/app"
if [ ! -f "$T/main.py" ]; then
  echo "没有找到已安装的 FosMap脑片活动图谱.app 或 cFos脑片分析.app（应在“应用程序”文件夹里）。"
  echo "请把补丁里 app 文件夹中的全部内容复制到：（程序名）.app/Contents/Resources/app/"
  read -r -p "按回车键关闭…"; exit 1
fi
echo "将安装补丁 v{VERSION} 到：$T"
read -r -p "请先退出 FosMap 脑片活动图谱，然后按回车键开始…"
if cp -R "$D/app/." "$T/"; then
  echo; echo "补丁已安装，可以打开软件了。以后的更新可以在软件里“帮助 → 检查更新”自动完成。"
else
  echo; echo "复制失败。可以在终端里运行： sudo cp -R \\"$D/app/.\\" \\"$T/\\""
fi
read -r -p "按回车键关闭…"
"""

README = """FosMap 脑片活动图谱（原名 cFos 脑片分析）补丁 v{VERSION}

【已经装了 v0.8 或更新版本】
打开软件 →“帮助 → 从文件安装补丁…”→ 选这个 zip 文件（不用解压）→ 重启软件。
（一般不需要手动下载：软件启动时会自动检查更新并提示安装。）

【还在用 v0.7 等旧版本（软件里没有“帮助 → 检查更新”）】
需要的完整安装包版本：v{MIN_BASE} 或更新。你的版本低于它时，请到 GitHub 下载完整安装包。
1. 关闭软件，把这个 zip 解压（Windows：右键 → 全部解压；Mac：双击）。
2. Windows：双击“手动安装（Windows 旧版本用）.bat”。
   Mac：右键点“手动安装（Mac 旧版本用）.command”→“打开”（直接双击可能提示“无法验证开发者”）。
3. 重新打开软件。之后的更新就能在软件里自动完成了。

更新公告见“更新公告.md”。
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
        if "__pycache__" in f.parts or not f.is_file() or f.suffix not in PATCH_SUFFIXES:
            continue
        files["app/" + f.relative_to(APP).as_posix()] = f.read_bytes()
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
        z.writestr(_zinfo("手动安装（Windows 旧版本用）.bat"), fill(WIN_BAT).replace("\n", "\r\n"))
        z.writestr(_zinfo("手动安装（Mac 旧版本用）.command", 0o755), fill(MAC_CMD))
    data = zp.read_bytes()
    latest = {"version": version, "min_base": a.min_base, "date": info["date"], "notes": notes,
              "patch": {"file": zname, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
              "full_installer": updater.FULL_PAGE}
    (out / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")
    # 累积的更新公告（新的在最前面）
    hist = out / "更新公告.md"
    old = hist.read_text(encoding="utf-8") if hist.exists() else "# FosMap 脑片活动图谱 更新公告\n"
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
