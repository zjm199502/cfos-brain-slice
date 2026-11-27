"""Locate and fit sections directly against the selected native reference volume."""
import numpy as np

from . import giraff, registration as R
from .atlas import SHAPE_ASR, edit_coords
from .atlas_choice import CCF, STAM


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
        R.refit_outline(reg, image, mask, results_dir, pixel_um)
    return reg


def run(project, targets, nuc, nonlinear=True, refine=True, unify=False, progress=None, cancelled=None, only=None, slides=None):
    say = progress or (lambda done, total, msg: None)
    atlas = project.registration_atlas
    out = {}
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
                result = fit_section(
                    slide, section, nuc, project.results_dir, nonlinear, atlas, say, cancelled)
                results[section.name] = result
            out[image.source_id] = results
        finally:
            if own:
                slide.close()
    return out

