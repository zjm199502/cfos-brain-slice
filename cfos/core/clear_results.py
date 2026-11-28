"""清除某一步的分析结果，便于重新分析（v1.0.2）。

每一步都分两步走：先 ``clear(..., dry_run=True)`` 列出将删除什么、哪些后续步骤要重做，
用户确认后再 ``clear(..., dry_run=False)`` 真正删除。范围与各页“所选图片和脑片”一致：
selection[图片] 为 None 表示整张图片（包括未启用分析的脑片），否则为脑片名称列表；
selection 本身为 None 表示整个项目。

磁盘上的缓存同时删除（原始识别缓存、配准位移场、荧光强度缓存、质量检查缓存），
后续步骤的“曾参与统计 / 曾导出 / 曾制图”记录一并去掉，项目页会显示需要重新做。
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .atlas_choice import has_registration
from .project import ImageEntry, Project, processed_section_names

log = logging.getLogger("cfos")

SECTIONS, DISPLAY, DETECT, COLOC, REGISTRATION, GROUPS, EXPORT, FIGURE, ALL = (
    "sections", "display", "detect", "coloc", "registration", "groups", "export", "figure", "all")

STEP_TITLES = {
    ALL: "全部分析结果",
    SECTIONS: "② 脑片拆分结果",
    DISPLAY: "③ 单独设置的亮度 / 对比度",
    DETECT: "④ 细胞识别结果",
    COLOC: "⑤ 共标结果",
    REGISTRATION: "⑥ 图谱配准结果",
    GROUPS: "⑦ 分组",
    EXPORT: "⑧ 导出记录",
    FIGURE: "⑨ 截图框和制图记录",
}

# 清除某一步后，哪些后续步骤的结果不再对应当前数据（需要重新做）
DOWNSTREAM = {
    ALL: [],
    SECTIONS: ["③ 脑片单独的亮度设置", "④ 细胞识别", "⑤ 共标", "⑥ 配准", "⑦ 分组与统计", "⑧ 导出", "⑨ 制图"],
    DISPLAY: ["④ 细胞识别（识别用到亮度窗口，下次识别这些脑片会重新识别，不会沿用旧结果）"],
    DETECT: ["⑤ 共标", "⑦ 组间统计", "⑧ 导出 Excel", "⑨ 制图"],
    COLOC: ["⑦ 组间统计", "⑧ 导出 Excel", "⑨ 制图"],
    REGISTRATION: ["⑦ 组间统计", "⑧ 导出 Excel（脑区归属要按新配准重新算）", "⑨ 制图"],
    GROUPS: ["⑦ 组间统计 / 功能网络 / 行为相关", "⑧ 导出 Excel（分组列）"],
    EXPORT: [],
    FIGURE: [],
}

# 清除某一步时去掉的输出记录（项目页“曾参与统计 / 曾导出 / 曾制图”）
_HISTORY = {
    SECTIONS: ("group", "excel", "figure"), ALL: ("group", "excel", "figure"),
    DETECT: ("group", "excel", "figure"), COLOC: ("group", "excel", "figure"),
    REGISTRATION: ("group", "excel", "figure"), GROUPS: ("group", "excel"),
    EXPORT: ("excel",), FIGURE: ("figure",), DISPLAY: (),
}

OUTPUT_FOLDER_PREFIXES = ("组间统计_", "功能网络_", "行为相关_")


@dataclass
class ClearReport:
    step: str
    lines: list[str] = field(default_factory=list)      # 将删除 / 已删除的内容
    downstream: list[str] = field(default_factory=list)  # 需要重新做的后续步骤
    notes: list[str] = field(default_factory=list)      # 不会删除、但需要知道的事
    manual_records: int = 0                              # 范围内人工修正的细胞数（④ 可选一并清除）
    output_folders: list[Path] = field(default_factory=list)  # 以往的统计结果文件夹（⑦ 可选一并删除）
    images: int = 0
    sections: int = 0

    @property
    def empty(self) -> bool:
        return not self.lines


def _safe_name(name: str) -> bool:
    return bool(name) and name not in (".", "..") and not any(c in name for c in "/\\:\x00")


def _targets(project: Project, selection):
    """[(图片, 范围内脑片列表, 是否整张图片)]"""
    out = []
    for im in project.images:
        if selection is None:
            names = None
        elif im.key in selection:
            names = selection[im.key]
        else:
            continue
        whole = names is None
        chosen = list(im.sections) if whole else [s for s in im.sections if s.name in set(names)]
        if whole or chosen:
            out.append((im, chosen, whole))
    return out


def _unlink(path: Path, dry_run: bool) -> bool:
    if not path.is_file():
        return False
    if not dry_run:
        try:
            path.unlink()
        except OSError:
            log.exception("无法删除 %s", path)
            return False
    return True


def _results(project: Project) -> Path:
    # 不用 project.results_dir：它会顺手建文件夹，预览删除内容时不应改动磁盘
    return project.folder / "results"


# ---------------------------------------------------------------- 输出记录
def _delete_history(project: Project, pairs, steps, dry_run: bool) -> int:
    path = _results(project) / "workflow.sqlite"
    if not steps or not pairs or not path.is_file():
        return 0
    marks = ",".join("?" * len(steps))
    n = 0
    try:
        with closing(sqlite3.connect(path)) as db, db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "outputs" not in tables:
                return 0
            for image, section in pairs:
                args = (image, section, *steps)
                where = f"image = ? AND section = ? AND step IN ({marks})"
                n += db.execute(f"SELECT COUNT(*) FROM outputs WHERE {where}", args).fetchone()[0]
                if not dry_run:
                    db.execute(f"DELETE FROM outputs WHERE {where}", args)
    except sqlite3.Error:
        log.exception("无法更新脑片输出记录")
    return n


# ---------------------------------------------------------------- ④ 细胞识别
def _part_files(project: Project, im: ImageEntry, name: str):
    if not _safe_name(name):
        return []
    d = _results(project) / "parts" / im.key
    return [d / f"{name}.v2.pkl", d / f"{name}.pkl", d / f"{name}.v2.box.json", d / f"{name}.box.json",
            d / f"{name}.v2.tmp", d / f"{name}.tmp"]


def _cell_files(project: Project, im: ImageEntry):
    new, old = project.cells_path(im), project._old_cells_path(im)
    return [new, old, new.with_suffix(".progress.json"), old.with_suffix(".progress.json")]


def _drop_cells(project: Project, im: ImageEntry, names: set[str], whole: bool, dry_run: bool):
    """去掉这些脑片的细胞和识别记录。返回 (脑片数, 细胞数)。"""
    try:
        df = project.load_cells(im)
    except Exception:  # noqa: BLE001  结果损坏时整张清除仍然可以删掉文件
        log.exception("读取 %s 的细胞结果失败", im.key)
        df = None
    if df is None:
        if whole and not dry_run:
            for p in _cell_files(project, im):
                _unlink(p, False)
        return 0, 0
    done = processed_section_names(im, df) | set(df.attrs.get("processed_sections", ()))
    in_scope = df["section"].isin(names) if "section" in df else pd.Series(True, index=df.index)
    cleared = done & names
    n_cells = int(in_scope.sum())
    if not cleared and not n_cells and not whole:
        return 0, 0
    if dry_run:
        return len(cleared), n_cells
    keep = df[~in_scope].copy()
    remaining = done - names
    if whole or (not remaining and keep.empty):
        for p in _cell_files(project, im):
            _unlink(p, False)
        return len(cleared), n_cells
    keep.attrs = dict(df.attrs)
    keep.attrs["processed_sections"] = [s.name for s in im.sections if s.name in remaining]
    if "carried_forward_sections" in keep.attrs:
        keep.attrs["carried_forward_sections"] = [n for n in keep.attrs["carried_forward_sections"] if n not in names]
    for key in ("section_models", "section_devices", "section_settings"):
        if key in keep.attrs:
            keep.attrs[key] = {n: v for n, v in keep.attrs[key].items() if n not in names}
    keep.attrs.pop("result_outlines", None)
    project.save_cells(im, keep.reset_index(drop=True))
    return len(cleared), n_cells


def _manual_in_scope(project: Project, im: ImageEntry, names: set[str], whole: bool):
    from .manual import load_manual
    d = load_manual(project, im)

    def hit(r):
        return whole or r.get("section") in names
    return d, sum(hit(r) for kind in ("added", "removed") for r in d[kind]), hit


def _drop_manual(project: Project, im: ImageEntry, names: set[str], whole: bool, dry_run: bool) -> int:
    from .manual import manual_path, save_manual
    d, n, hit = _manual_in_scope(project, im, names, whole)
    if n and not dry_run:
        rest = {kind: [r for r in d[kind] if not hit(r)] for kind in ("added", "removed")}
        if rest["added"] or rest["removed"]:
            save_manual(project, im, rest)
        else:
            _unlink(manual_path(project, im), False)
    return n


def _status_after_detect(project: Project, im: ImageEntry):
    enabled = sum(s.enabled for s in im.sections)
    try:
        df = project.load_cells(im)
    except Exception:  # noqa: BLE001
        df = None
    done = len(processed_section_names(im, df)) if df is not None else 0
    if not im.sections:
        im.status = "未处理"
    elif not done:
        im.status = f"已拆分 {len(im.sections)} 张脑片"
    elif done < enabled:
        im.status = f"已识别细胞（{done}/{enabled} 张脑片）"


def _clear_detect(project, targets, report, dry_run, include_manual):
    n_sec = n_cells = n_parts = n_manual_scope = n_manual = 0
    for im, chosen, whole in targets:
        names = {s.name for s in chosen}
        s_count, c_count = _drop_cells(project, im, names, whole, dry_run)
        n_sec += s_count; n_cells += c_count
        if whole:
            d = _results(project) / "parts" / im.key
            if d.is_dir():
                n_parts += sum(1 for p in d.iterdir() if p.suffix == ".pkl")
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
        else:
            for s in chosen:
                n_parts += sum(_unlink(p, dry_run) and p.suffix == ".pkl" for p in _part_files(project, im, s.name))
        n_manual_scope += _manual_in_scope(project, im, names, whole)[1]
        if include_manual:
            n_manual += _drop_manual(project, im, names, whole, dry_run)
        if not dry_run:
            _status_after_detect(project, im)
    if n_sec or n_cells:
        report.lines.append(f"{n_sec} 张脑片的细胞识别结果（共 {n_cells} 个细胞，含共标结果）")
    if n_parts:
        report.lines.append(f"{n_parts} 个原始识别缓存（删除后重新识别不会再沿用旧结果，会真正重新识别）")
    if n_manual:
        report.lines.append(f"{n_manual} 条人工修正（补加 / 删除的细胞）")
    report.manual_records = n_manual_scope


# ---------------------------------------------------------------- ⑤ 共标
def _clear_coloc(project, targets, report, dry_run):
    n_sec = n_cells = 0
    for im, chosen, _whole in targets:
        names = {s.name for s in chosen}
        df = project.load_cells(im)
        if df is None or "combo" not in df or "section" not in df:
            continue
        hit = df["section"].isin(names) & df["combo"].notna()
        if not hit.any():
            continue
        n_sec += df.loc[hit, "section"].nunique(); n_cells += int(hit.sum())
        if not dry_run:
            df["combo"] = df["combo"].astype(object)
            df.loc[hit, "combo"] = None
            df.attrs.pop("result_outlines", None)
            project.save_cells(im, df)
    if n_sec:
        report.lines.append(f"{n_sec} 张脑片的共标结果（{n_cells} 个细胞的阳性组合；细胞识别结果保留）")


# ---------------------------------------------------------------- ⑥ 配准
def _warp_names(reg: dict):
    names = set()
    for key in ("warp_file", "base_warp_file"):
        v = reg.get(key)
        if isinstance(v, str) and v:
            names.add(v)
    return names


def _referenced_warps(project: Project):
    used = set()
    for im in project.images:
        for s in im.sections:
            used |= _warp_names(s.registration or {})
            for reg in (s.other_registrations or {}).values():
                if isinstance(reg, dict):
                    used |= _warp_names(reg)
    return used


def _registration_files(project: Project, im: ImageEntry, section, reg: dict):
    from . import registration as R
    results = _results(project)
    files = []
    try:
        record = R.Registration.from_dict(reg)
    except Exception:  # noqa: BLE001
        return files
    folder = results / ("stam_giraff" if record.fit_atlas == R.STAM else "registration")
    for name in _warp_names(reg):
        if len(name) == 68 and name.endswith(".npy") and all(c in "0123456789abcdef" for c in name[:-4]):
            files.append(folder / name)
    if record.warp_file is None and _safe_name(section.name):
        files.append(folder / f"{im.key}_{section.name}_warp.npy")
    return files


def _intensity_file(project: Project, im: ImageEntry, name: str):
    return _results(project) / "intensity" / im.key / f"{name}.pkl" if _safe_name(name) else None


def _clear_registration(project, targets, report, dry_run):
    n_sec = n_files = 0
    candidates = []
    for im, chosen, _whole in targets:
        touched = False
        for s in chosen:
            reg = s.registration or {}
            if not reg:
                continue
            if has_registration(reg):
                n_sec += 1
            candidates += _registration_files(project, im, s, reg)
            inten = _intensity_file(project, im, s.name)
            if inten is not None:
                candidates.append(inten)
            if not dry_run:
                s.registration = {}
                touched = True
        if touched and "配准" in im.status:
            enabled = [s for s in im.sections if s.enabled]
            done = sum(bool(has_registration(s.registration)) for s in enabled)
            im.status = f"已配准 {done}/{len(enabled)} 张脑片" if done else f"已拆分 {len(im.sections)} 张脑片"
    if n_sec or candidates:
        used = _referenced_warps(project) if not dry_run else set()
        if dry_run:
            # 预览时把即将清除的配准排除在外，再看哪些位移场仍被其他脑片使用
            cleared = {id(s) for _im, chosen, _w in targets for s in chosen}
            for im in project.images:
                for s in im.sections:
                    if id(s) not in cleared:
                        used |= _warp_names(s.registration or {})
                    for reg in (s.other_registrations or {}).values():
                        if isinstance(reg, dict):
                            used |= _warp_names(reg)
        for p in dict.fromkeys(candidates):
            if p.name in used:
                continue
            if _unlink(p, dry_run) and p.suffix == ".npy":
                n_files += 1
    if n_sec:
        line = f"{n_sec} 张脑片在当前图谱（{project.registration_atlas}）中的配准结果，包括手动调整和标志点"
        if n_files:
            line += f"；{n_files} 个配准位移场文件"
        report.lines.append(line)
        report.notes.append("另一图谱的配准结果不受影响；切换图谱后仍可使用。")


# ---------------------------------------------------------------- ⑦ 分组
def _clear_groups(project, targets, report, dry_run, include_outputs):
    n = 0
    for im, chosen, _whole in targets:
        for s in chosen:
            key = f"{im.key}/{s.name}"
            sample = project.samples.get(key)
            if sample and (sample.get("group") or sample.get("animal")):
                n += 1
                if not dry_run:
                    project.samples.pop(key, None)
    if n:
        report.lines.append(f"{n} 张脑片的动物编号和组别")
    folders = sorted(p for p in project.folder.iterdir()
                     if p.is_dir() and p.name.startswith(OUTPUT_FOLDER_PREFIXES)) if project.folder.is_dir() else []
    report.output_folders = folders
    if include_outputs and folders:
        report.lines.append(f"项目文件夹中以往的 {len(folders)} 个统计结果文件夹（组间统计 / 功能网络 / 行为相关）")
        if not dry_run:
            for p in folders:
                shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------- ③ 显示
def _clear_display(project, targets, report, dry_run):
    n_img = n_sec = 0
    for im, chosen, whole in targets:
        names = {s.name for s in chosen}
        if whole and (im.display_overrides or im.model_display.get("")):
            n_img += 1
            if not dry_run:
                im.display_overrides = {}
                im.model_display.pop("", None)
        for name in names:
            if im.section_display.get(name) or name in im.model_display:
                n_sec += 1
                if not dry_run:
                    im.section_display.pop(name, None)
                    im.model_display.pop(name, None)
    if n_img:
        report.lines.append(f"{n_img} 张图片单独设置的亮度 / 对比度（恢复为项目设置）")
    if n_sec:
        report.lines.append(f"{n_sec} 张脑片单独设置的亮度 / 对比度（恢复为图片或项目设置）")


# ---------------------------------------------------------------- ⑨ 制图
def _clear_figure(project, targets, report, dry_run):
    from .figure import section_for
    n = 0
    for im, chosen, whole in targets:
        names = {s.name for s in chosen}
        keep = []
        for r in im.figure_rois:
            s = section_for(im, r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
            if whole or (s is not None and s.name in names):
                n += 1
            else:
                keep.append(r)
        if not dry_run:
            im.figure_rois = keep
    if n:
        report.lines.append(f"{n} 个截图框")


# ---------------------------------------------------------------- ② 拆分
def _clear_sections(project, targets, report, dry_run):
    n_img = n_sec = 0
    for im, chosen, whole in targets:
        if not chosen:
            continue
        n_img += 1; n_sec += len(chosen)
        if dry_run:
            continue
        names = {s.name for s in chosen}
        keys = {f"{s.name}@{','.join(map(str, s.bbox0))}" for s in chosen}
        for k in list(im.review_settings):
            if k in keys:
                im.review_settings.pop(k, None)
        im.field_reviews = [r for r in im.field_reviews if r.get("脑片") not in names]
        if whole:
            im.replace_sections([])
            im.changed_outlines = {}       # 这张图片已没有任何结果，下次拆分从头开始
            im.review_settings = {}
            im.field_reviews = []
            im.status = "未处理"
            for sub in ("intensity",):
                shutil.rmtree(_results(project) / sub / im.key, ignore_errors=True)
            _unlink(_results(project) / "quality" / f"{im.key}.json", False)
        else:
            im.replace_sections([s for s in im.sections if s.name not in names])
            im.status = f"已拆分 {len(im.sections)} 张脑片" if im.sections else "未处理"
    if n_sec:
        report.lines.insert(0, f"{n_img} 张图片中的 {n_sec} 个脑片轮廓（需要重新拆分）")


# ---------------------------------------------------------------- 入口
def clear(project: Project, step: str, selection=None, *, dry_run: bool = True,
          include_manual: bool = False, include_outputs: bool = False) -> ClearReport:
    """清除 selection 范围内 step 这一步的结果。dry_run=True 时只统计，不改动任何东西。

    ② 拆分 / 全部：同时清除这些脑片的所有后续结果（识别、人工修正、配准、分组、单独的显示设置），
    因为脑片轮廓没了，这些结果不再对应任何脑片。
    """
    if step not in STEP_TITLES:
        raise ValueError(f"未知步骤：{step}")
    report = ClearReport(step=step, downstream=list(DOWNSTREAM[step]))
    targets = _targets(project, selection)
    report.images = len(targets)
    report.sections = sum(len(chosen) for _im, chosen, _w in targets)
    cascade = step in (SECTIONS, ALL)
    if step == DISPLAY or cascade:
        _clear_display(project, targets, report, dry_run)
    if step == DETECT or cascade:
        _clear_detect(project, targets, report, dry_run, include_manual or cascade)
    if step == COLOC:
        _clear_coloc(project, targets, report, dry_run)
    if step == REGISTRATION or cascade:
        _clear_registration(project, targets, report, dry_run)
    if step == GROUPS or cascade:
        _clear_groups(project, targets, report, dry_run, include_outputs and step == GROUPS)
    if step == FIGURE:
        _clear_figure(project, targets, report, dry_run)
    pairs = [(im.key, s.name) for im, chosen, _w in targets for s in chosen]
    n_hist = _delete_history(project, pairs, _HISTORY[step], dry_run)
    if n_hist:
        report.lines.append(f"{n_hist} 条“曾参与统计 / 曾导出 / 曾制图”记录（之前导出的文件是旧结果，需要重新导出）")
    if cascade:
        _clear_sections(project, targets, report, dry_run)
    if step == DETECT and report.manual_records and not include_manual:
        report.notes.append(f"范围内有 {report.manual_records} 条人工修正（补加 / 删除的细胞）会保留，"
                            "重新识别后自动重新应用。")
    if step == EXPORT:
        report.notes.append("已导出的 Excel 文件本身不会删除；重新导出时选同一个文件名会覆盖它（系统会先询问）。")
    if step == FIGURE:
        report.notes.append("已保存的图片文件本身不会删除。")
    if not dry_run and not report.empty:
        project.save()
    return report


def describe(report: ClearReport) -> str:
    """确认框 / 完成提示里的文字。"""
    parts = ["将删除：" if report.lines else "所选范围内没有可清除的结果。"]
    parts += [f"• {line}" for line in report.lines]
    if report.lines and report.downstream:
        parts += ["", "之后需要重新做：", *[f"• {d}" for d in report.downstream]]
    if report.notes:
        parts += ["", *[f"注：{n}" for n in report.notes]]
    return "\n".join(parts)


def rerun_behavior() -> list[tuple[str, str]]:
    """各步骤“重新分析”时对旧结果的处理（显示在帮助里，也用于测试文档与代码一致）。"""
    return [
        ("② 脑片拆分", "重新自动拆分会替换所选图片的脑片轮廓；轮廓改变的脑片，其细胞结果自动作废，配准结果丢失，分组按脑片名保留。"),
        ("③ 亮度 / 对比度", "设置立即生效，覆盖原值；识别用到的亮度窗口改变后，下次识别这些脑片会重新识别。"),
        ("④ 细胞识别", "参数没变的脑片直接沿用上次结果（不会重新识别）；参数变了的脑片覆盖旧结果；范围外的脑片保留。"
                     "人工修正的细胞每次都会重新应用。"),
        ("⑤ 共标", "按当前参数从识别结果重新计算，覆盖所选脑片的旧共标结果。"),
        ("⑥ 配准", "重新自动配准覆盖所选脑片的旧结果，包括手动调整和标志点。"),
        ("⑦ 分组与统计", "每次计算都新建一个带时间的结果文件夹，旧结果不会被覆盖，也不会被删除。"),
        ("⑧ 导出 Excel", "导出到你选的文件；同名文件会被覆盖（系统会先询问）。"),
        ("⑨ 截图制图", "保存到你选的位置；同名文件会被覆盖。"),
    ]
