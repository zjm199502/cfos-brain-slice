"""处理流程：识别一张图片（可中断后继续）、重新筛选/共标、一键批量处理。"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from . import detection as det
from . import registration as R
from . import quality as Q
from .coloc import colocalize, nucleus_channel
from .log import log, friendly
from .project import Project, ImageEntry
from .sections import find_sections
from .slide_io import open_slide

IMAGE_EXTS = (".qptiff", ".tif", ".tiff", ".kfbf", ".vsi")


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
def rebuild_cells(project: Project, im: ImageEntry, pixel_um0: float) -> pd.DataFrame | None:
    """从每张脑片的原始识别结果，按当前阈值重新筛选并计算共标（不重新识别）。"""
    frames = []
    names = []
    for s in im.sections:
        if not s.enabled:
            continue
        raw = project.load_part(im, s)
        if raw is not None:
            frames.append(raw)
            names.append(s.name)
    if not frames:
        return None
    return _merge_frames(project, im, pixel_um0, frames, names)


def _merge_frames(project, im, pixel_um0, frames, names):
    df = pd.concat(frames, ignore_index=True)
    df = det.apply_filters(df, project.detection)
    from .manual import apply_manual, load_manual
    df = apply_manual(df, load_manual(project, im), pixel_um0, len(project.channel_names))
    df["cell_id"] = np.arange(len(df))
    df = colocalize(df, pixel_um0, project.coloc, nucleus_channel(project.channel_names)) if len(df) else df
    df.attrs["processed_sections"] = names
    project.save_cells(im, df)
    return df


def detect_image(project: Project, im: ImageEntry, sl=None, redo=False, report=None, cancel=None,
                 progress_range=(0, 100), total=100) -> dict:
    """识别一张图片的所有脑片。每张脑片识别完马上保存，参数没变的脑片直接用上次的结果，
    所以中途取消、死机或断电后，再点一次就从没做完的脑片继续。返回 {"reused": n, "done": n}。"""
    own = sl is None
    if own:
        sl = open_slide(im.path)
    try:
        if not im.sections:
            im.sections = find_sections(sl, channels=project.tissue_channels())
        secs = [s for s in im.sections if s.enabled]
        from .unmix import prepare as _prepare_unmix
        _prepare_unmix(project, sl, im)        # 识别前去除串色：开启了但还没估算时，先用这张图片估算
        import hashlib
        import json
        from . import accel
        from .sdnet import onnx_for
        models = {}
        for p in project.detection.channels.values():
            if p.detect and p.model not in models:
                directory = det._model_dir(p.model)
                stamp = []
                if directory:
                    files = set(directory.iterdir())
                    if accel.net_backend() == "onnx" and not (directory / "model.onnx").is_file():
                        selected = onnx_for(directory)
                        if selected is not None:
                            files.add(selected)
                    for file in sorted(files):
                        if file.suffix in (".json", ".h5", ".onnx") and file.is_file():
                            stat = file.stat()
                            stamp.append((file.name, stat.st_size, stat.st_mtime_ns))
                identity = (str(directory), stamp, accel.net_backend())
                if det._MODEL_IDENTITIES.get(p.model) != identity:
                    det.forget_model(p.model)
                    det._MODEL_IDENTITIES[p.model] = identity
                models[p.model] = identity
        src = Path(im.path).stat()
        identity = [det.raw_signature(project.detection), project.z_mode, models,
                    src.st_size, src.st_mtime_ns, sl.pixel_size_um]
        frames = []
        a, b = progress_range
        reused = done = 0
        name = Path(im.path).name
        for k, s in enumerate(secs):
            if cancel is not None and cancel():
                raise InterruptedError
            sig = hashlib.sha256(json.dumps([identity, s.bbox0, s.contour0], sort_keys=True).encode()).hexdigest()
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
            raw = det.detect_section(sl, s, project.detection, progress=prog, cancel=cancel, prefilter=False)
            project.save_part(im, s, sig, raw)
            frames.append(raw)
            done += 1
        if frames:
            _merge_frames(project, im, sl.pixel_size_um, frames, [s.name for s in secs])
        im.status = "已识别细胞"
        im.error = ""
        project.save()
        return {"reused": reused, "done": done}
    finally:
        if own:
            sl.close()


def refilter_all(project: Project, report=None) -> int:
    """所有图片按当前阈值和共标参数重新计算（几秒钟，不需要重新识别）。返回处理的图片数。"""
    n = 0
    for i, im in enumerate(project.images):
        if report:
            report(i, len(project.images), f"正在重新筛选和计算共标：{im.key}")
        parts = {sec.name: raw for sec in im.sections if sec.enabled
                 if (raw := project.load_part(im, sec)) is not None}
        if not parts:
            # 旧版本项目没有原始识别结果，只能在原结果上重新算共标
            df = project.load_cells(im)
            if df is None or df.empty:
                continue
            sl = open_slide(im.path); px = sl.pixel_size_um; sl.close()
            from .manual import apply_manual, load_manual
            df = df.drop(columns=[c for c in ("merged_id", "combo", "nuc_by_intensity") if c in df.columns])
            if "manual" in df.columns:
                df = df[df["manual"] != 1]  # 上次补加的细胞先去掉，下面按记录重新加
            df = apply_manual(df, load_manual(project, im), px, len(project.channel_names))
            df = colocalize(df, px, project.coloc, nucleus_channel(project.channel_names))
            project.save_cells(im, df)
        else:
            sl = open_slide(im.path); px = sl.pixel_size_um; sl.close()
            _merge_frames(project, im, px, list(parts.values()), list(parts))
        n += 1
    return n


# ---------------------------------------------------------------- 批量
def run_batch(project: Project, files, do_register=True, do_nonlinear=True, redo=False,
              report=None, cancel=None) -> list[dict]:
    """返回每张脑片的检查清单（需要人工检查的会标出原因）。某张图片出错时记录原因并继续处理下一张。"""
    def say(d, t, txt):
        if report:
            report(d, t, txt)

    def check_cancel():
        if cancel is not None and cancel():
            raise InterruptedError

    existing = {Path(im.path).resolve(): im for im in project.images}
    checklist = []
    for f in files:
        if Path(f).resolve() not in existing:
            try:
                im, _ = project.import_image(f)
            except Exception as e:  # noqa: BLE001
                log.error("无法打开 %s\n%s", f, traceback.format_exc())
                checklist.append({"图片": Path(f).name, "脑片": "", "配准": "", "需要检查": f"出错：{friendly(e)}"})
                continue
            existing[Path(f).resolve()] = im
    project.save()
    targets = [existing[Path(f).resolve()] for f in files if Path(f).resolve() in existing]
    nuc = nucleus_channel(project.channel_names)
    total = len(targets) * 100
    for i, im in enumerate(targets):
        check_cancel()
        base = i * 100
        name = Path(im.path).name
        try:
            sl = open_slide(im.path)
        except Exception as e:  # noqa: BLE001
            log.error("无法打开 %s\n%s", im.path, traceback.format_exc())
            im.error = friendly(e); im.status = "出错"
            checklist.append({"图片": name, "脑片": "", "配准": "", "需要检查": f"出错：{im.error}"})
            continue
        try:
            if not im.sections or redo:
                say(base, total, f"{name}：拆分脑片")
                im.sections = find_sections(sl, channels=project.tissue_channels())
                if not im.sections:
                    raise ValueError("没有找到脑片（图片可能是空白的，或组织太暗）")
            secs = [s for s in im.sections if s.enabled]
            detect_image(project, im, sl, redo=redo, report=report, cancel=cancel,
                         progress_range=(base + 5, base + 75), total=total)
            if do_register and secs:
                todo = [s for s in secs if redo or not s.registration.get("plane")]
                if todo:
                    say(base + 78, total, f"{name}：配准")
                    from . import autoreg
                    regs = autoreg.run(project, [im], nuc, nonlinear=do_nonlinear, cancelled=cancel,
                                       only={im.key: {s.name for s in todo}}, slides={im.path: sl},
                                       progress=lambda d, t, txt: say(base + 78 + int(17 * d / max(t, 1)), total,
                                                                      f"{name}：{txt}")).get(im.path, {})
                    for s in todo:
                        if s.name in regs:
                            s.registration = regs[s.name].to_dict()
                    project.save()
                im.status = "已配准"
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
            qm = Q.check_image(project, im, sl)
        except Exception:  # noqa: BLE001
            log.warning("质量检查失败 %s\n%s", name, traceback.format_exc()); qm = {}
        for s in [s for s in im.sections if s.enabled]:
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
