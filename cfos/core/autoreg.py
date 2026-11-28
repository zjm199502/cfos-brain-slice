"""Locate and fit sections directly against the selected native reference volume."""
import numpy as np

from . import giraff, registration as R
from .atlas import SHAPE_ASR, edit_coords
from .atlas_choice import CCF, STAM

AP_BY_LAYERS = False  # 非线性配准时按海马细胞层（重贴轮廓后）比较前后位置，并按切片顺序联合选择（series.choose_ap）。
                      # 默认关闭：示例数据上重贴轮廓后的评分与深部贴合后的结果不一致（S03 挪 300 µm 反而更差，合计 P90 129 → 179 µm）


def fit_section(slide, section, nuc, results_dir, nonlinear, atlas, say=None, cancelled=None, *, thumb=None):
    image, mask, ds = thumb if thumb is not None else R.section_thumbnail(slide, section, nuc)
    pixel_um = ds * slide.pixel_size_um
    match, _, _ = giraff.fit(image, mask, pixel_um, progress=say, cancelled=cancelled, atlas=atlas)
    image_plane = np.reshape(match["plane_image_mm"], (3, 3))
    if atlas == STAM:
        plane = edit_coords(image_plane)
    else:
        # Moving volume axes are AP,DV,ML. QuickNII reverses all three axes.
        plane = -image_plane[:, [0, 2, 1]] / .025
        plane[0] += np.array(SHAPE_ASR)[[2, 0, 1]] - 1
    reg = R.Registration(plane=plane.ravel().tolist(), fit_atlas=atlas, method="giRAff",
                         orientation=match["orientation"], warp_file="", bbox0=list(section.bbox0),
                         shape_hw=list(image.shape), downsample=float(ds), status=f"{atlas} · giRAff",
                         initialization={k:v for k,v in match.items() if k not in ("orientation","plane_image_mm")})
    if nonlinear:
        if cancelled and cancelled():
            raise InterruptedError
        if say:
            say(0, 1, f"正在贴合 {atlas} 组织外轮廓…")
        R.refit_outline(reg, image, mask, results_dir, pixel_um, search_plane=True)
    return reg


def run(project, targets, nuc, nonlinear=True, refine=True, unify=False, progress=None, cancelled=None, only=None, slides=None):
    """giRAff 逐张定位 → 同一动物按切片顺序联合定位前后位置（series.position_series）
    → 按海马细胞层比较前后位置、保持切片顺序（deep_refine.ap_profile + series.choose_ap）→ 非线性贴合。
    联合定位挪动了的或对不上的脑片带 ⚠；纹理微调前后位置时也不会再跳出切片顺序允许的范围。"""
    from . import deep_refine, series
    say = progress or (lambda done, total, msg: None)
    atlas = project.registration_atlas
    out, thumbs = {}, {}
    for image in targets:
        sections = [s for s in image.sections if s.enabled and (only is None or s.name in only.get(image.key, ()))]
        if not sections:
            continue
        slide = (slides or {}).get(image.source_id)
        own = slide is None
        if own:
            slide = image.open_slide(z_mode=project.z_mode)
        try:
            results = {}
            for section in sections:
                if cancelled and cancelled():
                    raise InterruptedError
                say(0, 1, f"正在用 giRAff 定位 {image.key} {section.name}…")
                thumb = R.section_thumbnail(slide, section, nuc)
                results[section.name] = fit_section(
                    slide, section, nuc, project.results_dir, False, atlas, say, cancelled, thumb=thumb)
                thumbs[(image.source_id, section.name)] = (thumb, thumb[2] * slide.pixel_size_um)
            out[image.source_id] = results
        finally:
            if own:
                slide.close()
    new = {(sid, name): reg for sid, regs in out.items() for name, reg in regs.items()}
    say(0, 1, "正在按切片顺序联合定位前后位置…")
    joint = series.position_series(project, new)
    chosen = {"offsets": {}, "notes": {}, "groups": []}
    if nonlinear and AP_BY_LAYERS:
        profiles = {}
        for k, (key, reg) in enumerate(new.items()):
            say(k, len(new), f"正在按海马细胞层比较前后位置 {key[1]}…")
            (image, mask, _), pixel_um = thumbs[key]
            try:
                prof = deep_refine.ap_profile(reg, image, mask, pixel_um, cancelled=cancelled)
            except InterruptedError:
                raise
            except Exception:
                prof = None   # 评分失败时这张照旧用纹理微调
            if prof is not None:
                profiles[key] = prof
        chosen = series.choose_ap(project, new, profiles)
        for key, off in chosen["offsets"].items():
            reg = new[key]
            reg.ap_auto = float(reg.ap_auto) + off
            reg.metrics = {**reg.metrics, "ap_profile": {**profiles[key], "chosen": off}}
    if nonlinear:
        for k, (key, reg) in enumerate(new.items()):
            if cancelled and cancelled():
                raise InterruptedError
            say(k, len(new), f"正在贴合 {atlas} {key[1]}…")
            (image, mask, _), pixel_um = thumbs[key]
            # 已按细胞层选好前后位置的只再微调倾斜
            bounds = (0.0, 0.0) if key in chosen["offsets"] else joint["bounds"].get(key)
            R.refit_outline(reg, image, mask, project.results_dir, pixel_um, search_plane=True, ap_bounds=bounds)
    for key, reg in new.items():
        reg.warnings = list(reg.warnings) + joint["notes"].get(key, []) + chosen["notes"].get(key, [])
        if key in joint["notes"] or key in joint["bounds"]:
            reg.metrics = {**reg.metrics, "series": {"ap_bounds_um": [round(v * series.VOXEL_UM) for v in
                                                                      joint["bounds"].get(key, ())],
                                                     "notes": joint["notes"].get(key, [])}}
    return out
