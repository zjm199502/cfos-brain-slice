"""项目页逐脑片进度与导出历史。"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import PurePosixPath, PureWindowsPath

import pandas as pd


def source_folders(images):
    """返回每张图片的最短可区分文件夹后缀和完整文件夹路径。"""
    folders = []
    for image in images:
        path = image.path
        cls = PureWindowsPath if "\\" in path or PureWindowsPath(path).drive else PurePosixPath
        folders.append(cls(path).parent)
    from collections import Counter
    distinct = set(folders)
    occurrences = Counter(type(folder)(*folder.parts[-n:]) for folder in distinct
                          for n in range(1, len(folder.parts) + 1))
    suffixes = {}
    for folder in distinct:
        parts = folder.parts
        for length in range(1, len(parts) + 1):
            suffix = type(folder)(*parts[-length:])
            if occurrences[suffix] == 1:
                break
        suffixes[folder] = str(suffix) if parts else "."
    return [(suffixes[folder], str(folder)) for folder in folders]


def _box(section):
    return ",".join(str(int(value)) for value in section.bbox0)


def record_output(project, step, sections):
    """记录成功写出的实际脑片，与源图存储标识及当时的脑片框绑定。"""
    rows = [(image.key, section.name, _box(section), step) for image, section in sections]
    if not rows:
        return
    try:
        with closing(sqlite3.connect(project.results_dir / "workflow.sqlite")) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS outputs (
                       image TEXT, section TEXT, box TEXT, step TEXT,
                       PRIMARY KEY (image, section, box, step))""")
            db.executemany("INSERT OR IGNORE INTO outputs VALUES (?, ?, ?, ?)", rows)
    except (OSError, sqlite3.Error):
        logging.getLogger("cfos").exception("无法记录脑片输出进度")


def record_table_output(project, step, table):
    if not {"图片", "脑片"}.issubset(table.columns):
        return
    pairs = set(table[["图片", "脑片"]].drop_duplicates().itertuples(index=False, name=None))
    sections = ((im, s) for im in project.images for s in im.sections if (im.key, s.name) in pairs)
    if step == "group":
        sections = ((im, s) for im, s in sections
                    if all((project.samples.get(f"{im.key}/{s.name}") or {}).get(k) for k in ("animal", "group")))
    record_output(project, step, sections)


def _output_history(project, image):
    path = project.folder / "results" / "workflow.sqlite"
    if not path.is_file():
        return set()
    try:
        with closing(sqlite3.connect(path)) as db:
            return set(db.execute("SELECT section, box, step FROM outputs WHERE image = ?", (image.key,)))
    except sqlite3.Error:
        logging.getLogger("cfos").exception("无法读取脑片输出进度")
        return set()


def _summarize(df):
    if not isinstance(df, pd.DataFrame) or "section" not in df:
        return None
    processed = df.attrs.get("processed_sections")
    detected = set(df["section"]) if len(df) else set()
    with_combo = set(df.loc[df["combo"].notna(), "section"]) if len(df) and "combo" in df else set()
    return [None if processed is None else sorted(processed), sorted(detected), sorted(with_combo)]


def save_cell_summary(path, df):
    """大型细胞文件附带轻量索引，项目页可以避免为显示状态而加载轮廓。"""
    index = path.with_suffix(".progress.json")
    tmp = index.with_suffix(".tmp")
    try:
        stat = path.stat()
        record = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "summary": _summarize(df)}
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(index)
    except (OSError, ValueError, TypeError, KeyError):
        logging.getLogger("cfos").exception("无法写入细胞结果进度索引")


@lru_cache(maxsize=32)
def _cell_summary(path, size, mtime_ns):
    """旧结果无索引时只读一次，缓存小型摘要，不保留轮廓。"""
    index = path.with_suffix(".progress.json")
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
        if record["size"] == size and record["mtime_ns"] == mtime_ns:
            return record["summary"]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        saved = pd.read_pickle(path)
        df = saved.get("cells") if isinstance(saved, dict) else saved
        return _summarize(df)
    except Exception:
        logging.getLogger("cfos").exception("无法读取 %s 的细胞进度", path.name)
        return None


@lru_cache(maxsize=32)
def _part_box(path, size, mtime_ns):
    return tuple(pd.read_pickle(path)["bbox0"])


def _has_raw_result(project, image, section):
    if not section.name or any(c in section.name for c in "/\\:\x00"):
        return False
    directory = project.folder / "results" / "parts" / image.key
    for suffix in (".v2.pkl", ".pkl"):
        path = directory / (section.name + suffix)
        if path.is_file():
            try:
                stat = path.stat()
                return _part_box(path, stat.st_size, stat.st_mtime_ns) == tuple(section.bbox0)
            except Exception:
                logging.getLogger("cfos").exception("无法读取 %s 的原始识别缓存", section.name)
                return False
    return False


def section_progress(project, image):
    """按 GUI 步骤返回每张脑片的当前结果；未记录过的旧版输出保留为未知。"""
    if not image.sections:
        return []
    history = _output_history(project, image)
    processed, detected, coloc = set(), set(), set()
    unreadable = False
    try:
        for path in (project.cells_path(image), project._old_cells_path(image)):
            if path.is_file():
                stat = path.stat()
                summary = _cell_summary(path, stat.st_size, stat.st_mtime_ns)
                if summary is None:
                    unreadable = True
                    break
                processed, detected, coloc = summary
                detected, coloc = set(detected), set(coloc)
                if processed is None:  # 历史整图结果覆盖当时所有启用脑片。
                    processed = {s.name for s in image.sections if s.enabled}
                else:
                    processed = set(processed)
                break
    except Exception:
        logging.getLogger("cfos").exception("无法读取 %s 的细胞进度", image.key)
        unreadable = True

    rows = []
    for s in image.sections:
        detect, common = "待识别", "待识别"
        if unreadable:
            detect, common = "结果读取失败", "结果读取失败"
        elif s.name in processed:
            detect = "已识别（有细胞）" if s.name in detected else "已识别（零细胞）"
            common = "已有共标结果" if s.name in coloc else ("待计算" if s.name in detected else "零细胞")
        elif _has_raw_result(project, image, s):
            detect, common = "已检测，待汇总", "待汇总"
        display = ("脑片设置" if image.section_display.get(s.name) else
                   "图片设置" if image.display_overrides else "项目设置")
        reg = s.registration
        registered = "待配准"
        if reg.get("plane"):
            registered = ("已配准（待复核）" if reg.get("warnings") else
                          "已手动调整" if reg.get("status") == "已手动调整" else "已配准")
        sample = project.samples.get(f"{image.key}/{s.name}") or {}
        key = (s.name, _box(s))
        groups = "已分组" if sample.get("group") and sample.get("animal") else "待分组"
        if (*key, "group") in history:
            groups += " · 曾参与统计"
        rows.append((s.name, ("已导入", "已拆分" if s.enabled else "已拆分（停用）",
                              display, detect, common, registered, groups,
                              "曾导出" if (*key, "excel") in history else "无导出记录",
                              "曾制图" if (*key, "figure") in history else "无制图记录")))
    return rows
