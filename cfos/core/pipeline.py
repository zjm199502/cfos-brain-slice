"""处理流程：识别一张图片（可中断后继续）、刷新共标、一键批量处理。"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from . import detection as det
from . import registration as R
from . import quality as Q
from .field_review import effective_settings
from .coloc import colocalize, nucleus_channel
from .log import log, friendly
from .project import Project, ImageEntry, processed_section_names
from .relocate import IMAGE_EXT
from .sections import find_sections
from .scope import selected_images, selected_sections

IMAGE_EXTS = tuple(sorted(IMAGE_EXT))


def list_images(folder) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTS)
                  and not p.name.startswith("~$"))


# ---------------------------------------------------------------- 排除区域 / 自定义区域
def inside_any(x: np.ndarray, y: np.ndarray, polygons) -> np.ndarray:
    m = np.zeros(len(x), bool)
    pts = np.stack([x, y], 1) if len(x) else np.zeros((0, 2))
    for poly in polygons:
        if len(poly) >= 3:
            m |= MplPath(np.asarray(poly, float)).contains_points(pts)
    return m


def drop_excluded(df: pd.DataFrame, im: ImageEntry) -> pd.DataFrame:
    """去掉落在排除区域（折叠、气泡、破损、边缘伪影）里的细胞。"""
    if df is None or df.empty or not im.exclusions:
        return df
    bad = inside_any(df["x"].to_numpy(float), df["y"].to_numpy(float), im.exclusions)
    return df[~bad]


# ---------------------------------------------------------------- 识别
def recompute_image_coloc(project: Project, im: ImageEntry, pixel_um0: float | None = None, *, section_names=None) -> pd.DataFrame | None:
    """Recalculate colocalization and replay manual edits on the accepted detections."""
    previous = project.load_cells(im)
    selected = processed_section_names(im, previous)
    if section_names is not None:
        selected.intersection_update(section_names)
    if not selected:
        return None
    if pixel_um0 is None:
        slide = im.open_slide()
        try:
            pixel_um0 = slide.pixel_size_um
        finally:
            slide.close()
    return _merge_frames(project, im, pixel_um0, [], [], partial=True, previous=previous, saved_sections=selected)


def _merge_frames(project, im, pixel_um0, frames, names, partial=False, previous=None, saved_sections=()):
    sections = {sec.name: sec for sec in im.sections}
    filtered = []
    settings_by_section = {}
    for name, raw in zip(names, frames):
        settings = effective_settings(project.detection, im, sections[name])
        filtered.append(det.apply_filters(raw, settings))
        if "inference_settings" in raw.attrs:
            settings_by_section[name] = {
                "inference_settings": raw.attrs["inference_settings"], "z_mode": raw.attrs.get("z_mode"),
                "model_identity": raw.attrs.get("model_identity"),
                "filters": {c: {k: getattr(p, k) for k in ("min_area_um2", "max_area_um2", "min_intensity",
                                                          "bg_fold", "min_contrast")}
                            for c, p in settings.channels.items() if c in raw.attrs.get("channel_models", {})}}
    from .manual import apply_manual, load_manual
    manual = load_manual(project, im)

    def correct(frame, section_names, *, exact_removals=False):
        relevant = {kind: [r for r in records if r.get("section") in section_names
                           or (kind == "removed" and not r.get("section"))]
                    for kind, records in manual.items()}
        return apply_manual(frame, relevant, pixel_um0, len(project.channel_names), exact_removals=exact_removals)

    corrected = []
    if filtered:
        corrected.append(correct(pd.concat(filtered, ignore_index=True), set(names)))
    if saved_sections:
        old = previous[previous["section"].isin(saved_sections)]
        if "manual" in old:
            old = old[old["manual"] != 1]
        corrected.append(correct(old, saved_sections, exact_removals=True))
    df = corrected[0] if len(corrected) == 1 else pd.concat(corrected, ignore_index=True)
    df["cell_id"] = np.arange(len(df))
    df = colocalize(df, pixel_um0, project.coloc, nucleus_channel(project.channel_names)) if len(df) else df
    processed = set(names).union(saved_sections)
    retained = (set(saved_sections).intersection(previous.attrs.get("carried_forward_sections", ()))
                if previous is not None else set())
    if partial:
        if previous is None:
            previous = project.load_cells(im)
        if previous is not None:
            others = sections.keys() - processed
            keep = previous[previous["section"].isin(others)]
            # Older whole-image results predate explicit tracking of zero-cell sections.
            previous_done = previous.attrs.get("processed_sections", processed_section_names(im, previous))
            processed.update(others.intersection(previous_done))
            retained.update(others.intersection(previous.attrs.get("carried_forward_sections", ())))
            if len(keep):
                for column in ("cell_id", "merged_id"):
                    if len(df) and column in keep:
                        df[column] += int(keep[column].max()) + 1
                df = pd.concat([keep, df], ignore_index=True)
    df.attrs["processed_sections"] = [s.name for s in im.sections if s.name in processed]
    df.attrs["carried_forward_sections"] = [s.name for s in im.sections if s.name in retained]
    sources = {}
    devices = {}
    if previous is not None:
        for attribute, dest in [("section_models", sources), ("section_devices", devices),
                                ("section_settings", settings_by_section)]:
            for key, value in previous.attrs.get(attribute, {}).items():
                if key in processed and key not in names:
                    dest[key] = value
    for key, raw in zip(names, frames):
        if "channel_models" in raw.attrs:
            sources[key] = raw.attrs["channel_models"]
        if "channel_devices" in raw.attrs:
            devices[key] = raw.attrs["channel_devices"]
    df.attrs["section_models"] = sources
    df.attrs["section_devices"] = devices
    df.attrs["section_settings"] = settings_by_section
    project.save_cells(im, df)
    return df


def detect_image(project: Project, im: ImageEntry, sl=None, redo=False, report=None, cancel=None,
                 progress_range=(0, 100), total=100, section_names=None) -> dict:
    """识别一张图片的所有脑片。每张脑片识别完马上保存，参数没变的脑片直接用上次的结果，
    所以中途取消、死机或断电后，再点一次就从没做完的脑片继续。返回 {"reused": n, "done": n}。"""
    own = sl is None
    if own:
        sl = im.open_slide(z_mode=project.z_mode)
    try:
        if not im.sections and section_names is None:
            im.sections = find_sections(sl, channels=project.tissue_channels())
        secs = [s for s in im.sections if s.enabled]
        partial = section_names is not None
        if partial:
            selected = set(section_names)
            secs = [s for s in secs if s.name in selected]
            if not secs or len(secs) != len(selected):
                raise ValueError("请选择当前图片中已启用的脑片。")
        from .unmix import prepare as _prepare_unmix
        _prepare_unmix(project, sl, im)        # 识别前去除串色：开启了但还没估算时，先用这张图片估算
        import hashlib
        import json
        src = Path(im.path).stat()
        identity = [det.cellpose4.model_identity(), project.z_mode, src.st_size, src.st_mtime_ns, sl.pixel_size_um]
        if getattr(sl, "scene_id", None) is not None:
            identity.append(sl.scene_source_signature())
        frames = []
        a, b = progress_range
        reused = done = 0
        name = im.display_name
        for k, s in enumerate(secs):
            if cancel is not None and cancel():
                raise InterruptedError
            settings = effective_settings(project.detection, im, s)
            signature_data = [identity, det.raw_signature(settings), s.bbox0, s.contour0]
            sig = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
            cached = None if redo else project.load_part(im, s, sig)
            if cached is not None:
                frames.append(cached)
                reused += 1
                continue

            def prog(d, t, k=k, s=s):
                if report:
                    report(a + int((b - a) * (k + d / max(t, 1)) / max(len(secs), 1)), total,
                           f"{name}：识别 {s.name} 的细胞（{d}/{t} 块）")
            log.info("识别 %s %s", name, s.name)
            raw = det.detect_section(sl, s, settings, progress=prog, cancel=cancel, prefilter=False)
            raw.attrs["model_identity"] = identity[0]
            project.save_part(im, s, sig, raw)
            frames.append(raw)
            done += 1
        if frames:
            result = _merge_frames(project, im, sl.pixel_size_um, frames, [s.name for s in secs], partial)
        im.status = "已识别细胞"
        if partial:
            completed = len(processed_section_names(im, result))
            enabled_count = sum(s.enabled for s in im.sections)
            if completed < enabled_count:
                im.status = f"已识别细胞（{completed}/{enabled_count} 张脑片）"
        im.error = ""
        project.save()
        return {"reused": reused, "done": done}
    finally:
        if own:
            sl.close()


def refresh_coloc(project: Project, report=None, *, selection=None, cancel=None) -> int:
    """按选定的图片和脑片更新共标，保留识别时的阈值与阳性判定。"""
    n = 0
    targets = selected_images(project, selection)
    for i, im in enumerate(targets):
        if cancel and cancel():
            raise InterruptedError
        if report:
            report(i, len(targets), f"正在重新计算共标：{im.key}")
        names = selection.get(im.key) if selection is not None else None
        if recompute_image_coloc(project, im, section_names=names) is not None:
            n += 1
    return n


# ---------------------------------------------------------------- 批量
def run_batch(project: Project, files, do_register=True, do_nonlinear=True, redo=False,
              report=None, cancel=None, *, selection=None) -> list[dict]:
    """返回每张脑片的检查清单（需要人工检查的会标出原因）。某张图片出错时记录原因并继续处理下一张。"""
    def say(d, t, txt):
        if report:
            report(d, t, txt)

    def check_cancel():
        if cancel is not None and cancel():
            raise InterruptedError

    targets = []
    if selection is not None:
        targets = [im for im in selected_images(project, selection)
                   if selection[im.key] is None or selected_sections(im, selection)]
        if not targets:
            return []
    checklist = []
    sources = dict.fromkeys(Path(f).resolve() for f in files) if selection is None else ()
    for f in sources:
        check_cancel()
        try:
            entries, notes = project.import_images(f)
        except Exception as e:  # noqa: BLE001
            log.error("无法打开 %s\n%s", f, traceback.format_exc())
            checklist.append({"类型": "导入", "图片": f.name, "脑片": "", "配准": "", "需要检查": f"出错：{friendly(e)}"})
            continue
        targets.extend(entries)
        for note in notes:
            checklist.append({"类型": "导入", "图片": f.name, "脑片": "", "配准": "", "需要检查": note})
    project.save()
    nuc = nucleus_channel(project.channel_names)
    total = len(targets) * 100
    for i, im in enumerate(targets):
        check_cancel()
        base = i * 100
        name = im.display_name
        try:
            sl = im.open_slide()
        except Exception as e:  # noqa: BLE001
            log.error("无法打开 %s\n%s", im.path, traceback.format_exc())
            im.error = friendly(e); im.status = "出错"
            checklist.append({"图片": name, "脑片": "", "配准": "", "需要检查": f"出错：{im.error}"})
            continue
        try:
            secs = selected_sections(im, selection)
            names = None
            if selection is not None and selection[im.key] is not None:
                names = [s.name for s in secs]
            if names is None and (not im.sections or redo):
                say(base, total, f"{name}：拆分脑片")
                im.sections = find_sections(sl, channels=project.tissue_channels())
                if not im.sections:
                    raise ValueError("没有找到脑片（图片可能是空白的，或组织太暗）")
                secs = selected_sections(im, selection)
            detect_image(project, im, sl, redo=redo, report=report, cancel=cancel,
                         progress_range=(base + 5, base + 75), total=total, section_names=names)
            if do_register and secs:
                todo = [s for s in secs if redo or not s.registration.get("plane")]
                if todo:
                    say(base + 78, total, f"{name}：配准")
                    from . import autoreg
                    regs = autoreg.run(project, [im], nuc, nonlinear=do_nonlinear, cancelled=cancel,
                                       only={im.key: {s.name for s in todo}}, slides={im.source_id: sl},
                                       progress=lambda d, t, txt: say(base + 78 + int(17 * d / max(t, 1)), total,
                                                                      f"{name}：{txt}")).get(im.source_id, {})
                    for s in todo:
                        if s.name in regs:
                            s.registration = regs[s.name].to_dict()
                    project.save()
                registered = sum(s.enabled and bool(s.registration.get("plane")) for s in im.sections)
                total_enabled = sum(s.enabled for s in im.sections)
                im.status = "已配准" if registered == total_enabled else f"已配准（{registered}/{total_enabled} 张脑片）"
        except InterruptedError:
            project.save(); sl.close()
            raise
        except Exception as e:  # noqa: BLE001
            log.error("处理 %s 出错\n%s", name, traceback.format_exc())
            im.error = friendly(e); im.status = "出错"
            checklist.append({"图片": name, "脑片": "", "配准": "", "需要检查": f"出错：{im.error}"})
            sl.close(); project.save()
            continue
        # 检查清单（包括自动质量检查：配准、模糊、过曝、组织缺损）
        try:
            say(base + 97, total, f"{name}：质量检查")
            qm = Q.check_image(project, im, sl, section_names=names)
        except Exception:  # noqa: BLE001
            log.warning("质量检查失败 %s\n%s", name, traceback.format_exc()); qm = {}
        for s in secs:
            reg = R.Registration.from_dict(s.registration)
            note = []
            st = reg.status
            if s.name in qm:
                _, lvl, why = Q.grade(qm[s.name])
                note += why
            if do_register and not reg.done:
                note.append("未配准")
            note += [f"配准：{w}" for w in reg.warnings]
            checklist.append({"图片": name, "脑片": s.name, "配准": st,
                              "质量": Q.grade(qm[s.name])[1] if s.name in qm else "",
                              "需要检查": "；".join(note) if note else "", "约 Bregma (mm)":
                              round(R.adjusted_plane(reg).ap_mm(), 2) if reg.done else None})
        sl.close()
        project.save()
    say(total, total, "完成")
    return checklist
