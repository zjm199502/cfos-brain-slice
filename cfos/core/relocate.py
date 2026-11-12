"""项目换电脑 / 换硬盘后重新找到原图。

项目文件里记录了每张图片的完整路径；另外还记录了：
- 相对项目文件夹的路径：图片和项目文件夹一起整体拷走时，打开项目就能自动找到；
- 文件大小：在新位置按文件名搜索时，用来确认是同一个文件（同名文件很常见，如 1.qptiff）。
都找不到时，让用户选一个文件夹（可以是整个硬盘），在里面按文件名搜索。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

IMAGE_EXT = {".qptiff", ".tif", ".tiff"}


def file_name(path: str) -> str:
    """不论是 Windows 还是其他系统写下的路径都能取出文件名。"""
    return re.split(r"[\\/]", str(path))[-1]


def parent_name(path: str) -> str:
    parts = re.split(r"[\\/]", str(path))
    return parts[-2] if len(parts) > 1 else ""


def rel_path(path: str, folder: Path) -> str | None:
    try:
        return os.path.relpath(path, folder)
    except ValueError:  # Windows 上不在同一个盘
        return None


def file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def auto_fix(project) -> int:
    """打开项目时调用：原路径不存在、但按相对路径能找到的，自动改过来。返回改了几张。"""
    n = 0
    for im in project.images:
        if Path(im.path).exists():
            continue
        rel = getattr(im, "rel", None)
        if rel:
            cand = Path(os.path.normpath(project.folder / rel.replace("\\", os.sep)))
            if cand.exists() and _size_ok(im, cand):
                im.path = str(cand); n += 1
    return n


def missing(project) -> list:
    return [im for im in project.images if not Path(im.path).exists()]


def _size_ok(im, cand) -> bool:
    want = getattr(im, "size", None)
    return want is None or file_size(cand) == want


def relocate(project, search_dir, progress=None, cancel=None) -> dict:
    """在 search_dir（含子文件夹）中按文件名寻找缺失的图片。
    返回 {"fixed": [(旧路径, 新路径)], "missing": [旧路径], "ambiguous": [(旧路径, [候选...])]}。"""
    todo = missing(project)
    want = {}
    for im in todo:
        want.setdefault(file_name(im.path).lower(), []).append(im)
    found: dict[str, list[Path]] = {}
    k = 0
    for root, dirs, files in os.walk(search_dir):
        # 跳过系统和隐藏文件夹，搜索整个硬盘时快很多
        dirs[:] = [d for d in dirs if not d.startswith((".", "$")) and d not in ("Windows", "Program Files",
                                                                                "Program Files (x86)", "AppData")]
        k += 1
        if progress and k % 200 == 0:
            progress(0, 0, f"正在搜索：{root}")
        if cancel is not None and cancel():
            raise InterruptedError
        for f in files:
            if f.lower() in want:
                found.setdefault(f.lower(), []).append(Path(root) / f)
    fixed, still, amb = [], [], []
    taken = {Path(im.path).resolve() for im in project.images if Path(im.path).exists()}
    for name, ims in want.items():
        cands = [c for c in found.get(name, []) if c.resolve() not in taken]
        for im in ims:
            ok = [c for c in cands if _size_ok(im, c)]
            if len(ok) > 1:  # 同名同大小：优先上一级文件夹名也相同的
                same = [c for c in ok if c.parent.name == parent_name(im.path)]
                ok = same if len(same) == 1 else ok
            if len(ok) == 1:
                old = im.path
                im.path = str(ok[0]); cands.remove(ok[0]); taken.add(ok[0].resolve())
                im.error = ""
                fixed.append((old, im.path))
            elif ok:
                amb.append((im.path, [str(c) for c in ok]))
            else:
                still.append(im.path)
    if fixed:
        project.save()
    return {"fixed": fixed, "missing": still, "ambiguous": amb}


def summary_text(res) -> str:
    lines = [f"已重新找到 {len(res['fixed'])} 张图片。"]
    if res["ambiguous"]:
        lines.append(f"\n{len(res['ambiguous'])} 张图片找到多个同名、同大小的文件，无法确定是哪一个，请用“替换这张图片的文件”手动指定：")
        lines += [f"• {file_name(o)}：" + "；".join(c[:3]) for o, c in res["ambiguous"][:10]]
    if res["missing"]:
        lines.append(f"\n{len(res['missing'])} 张图片在这个文件夹里没有找到（文件名要和原来一致）：")
        lines += [f"• {file_name(o)}" for o in res["missing"][:15]]
    return "\n".join(lines)
