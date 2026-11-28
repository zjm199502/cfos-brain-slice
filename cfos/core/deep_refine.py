"""Texture-driven nonlinear refinement of deep structures for an already fitted atlas section.

The outline refinement (stam_refinement.refine) deliberately fades to zero about 1 mm inside the
tissue, so hippocampus, ventricles, thalamus and striatum keep only the global affine fit.  The
atlas reference template (CCFv3 two-photon autofluorescence) carries almost no mutual information
with a DAPI section (interior MI ≈ 0.015 on the example data), so it cannot drive deep corrections.

Instead the atlas labels themselves are turned into a DAPI-like image: every region is painted with
the median nuclear density that the section shows inside it.  Principal cell layers (DG-sg, CA-sp),
cell-sparse molecular layers, fibre tracts and ventricles then appear in the synthetic image with the
contrast they have in DAPI, and a smooth B-spline maps that synthetic image onto the section.  The
painting is re-estimated after each round, so the region intensities sharpen as alignment improves.
The tissue outline is kept as a second metric so the outline fit is not traded away.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import registration as R

ROUNDS = 2
GRID_UM = (320.0, 480.0)  # B-spline control-point spacing; the wider grid is tried when the first folds
DENSITY_SIGMA_UM = 12.0  # nuclear density smoothing
BACKGROUND_UM = 400.0    # illumination flattening
MIN_JACOBIAN = 0.25


def nuclear_density(image, mask, pixel_um):
    """Flat-field normalised DAPI density in [0, 1] inside the tissue."""
    x = np.asarray(image, np.float32)
    m = np.asarray(mask, bool)
    d = ndi.gaussian_filter(x, max(.7, DENSITY_SIGMA_UM / pixel_um))
    # Normalised convolution keeps the tissue edge from darkening the background estimate.
    s = BACKGROUND_UM / pixel_um
    bg = ndi.gaussian_filter(np.where(m, x, 0), s) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), s), 1e-3)
    r = np.where(m, d / (bg + 1.0), 0).astype(np.float32)
    hi = float(np.percentile(r[m], 99.5)) if m.any() else 1.
    return np.clip(r / max(hi, 1e-6), 0, 1)


def paint_labels(labels, density, mask, min_pixels=8):
    """Synthetic DAPI: each atlas label painted with the median section density inside it."""
    lab = np.asarray(labels)
    inside = (lab > 0) & mask
    keys, inverse = np.unique(lab[inside], return_inverse=True)
    values = density[inside]
    fill = float(np.median(values)) if values.size else 0.
    out = np.zeros(keys.shape, np.float32)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(keys) + 1))
    for i in range(len(keys)):
        v = values[order[bounds[i]:bounds[i + 1]]]
        out[i] = np.median(v) if v.size >= min_pixels else fill
    synth = np.zeros(lab.shape, np.float32)
    synth[inside] = out[inverse]
    synth[(lab > 0) & ~mask] = fill
    return synth


def _bspline(fixed, moving, fixed2, moving2, grid_px, w_outline=.3, iters=400):
    """Two-metric B-spline (texture NCC + outline MI); returns fixed->moving (dx, dy) in pixels."""
    itk = R._itk()
    as_img = lambda a: itk.image_view_from_array(np.ascontiguousarray(a, dtype=np.float32))
    m = itk.ElastixRegistrationMethod[itk.Image[itk.F, 2], itk.Image[itk.F, 2]].New()
    m.SetFixedImage(as_img(fixed)); m.SetMovingImage(as_img(moving))
    m.AddFixedImage(as_img(fixed2)); m.AddMovingImage(as_img(moving2))
    # The bending-energy penalty ignores images, but multi-metric elastix needs one pair per metric.
    m.AddFixedImage(as_img(fixed2)); m.AddMovingImage(as_img(moving2))
    po = itk.ParameterObject.New()
    bs = po.GetDefaultParameterMap("bspline", 3, float(grid_px))
    bs["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    bs["Metric"] = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation",
                    "TransformBendingEnergyPenalty"]
    bs["Metric0Weight"] = [str(1. - w_outline)]
    bs["Metric1Weight"] = [str(w_outline)]
    bs["Metric2Weight"] = ["0.05"]
    bs["FinalGridSpacingInPhysicalUnits"] = [str(float(grid_px))]
    bs["GridSpacingSchedule"] = ["4", "2", "1"]
    bs["MaximumNumberOfIterations"] = [str(iters)]
    bs["NumberOfSpatialSamples"] = ["6000"]
    for k, v in (("FixedImagePyramid", "FixedSmoothingImagePyramid"),
                 ("MovingImagePyramid", "MovingSmoothingImagePyramid")):
        bs[k] = [v] * 3
    # Elastix wants one interpolator and sampler per metric, the bending penalty included.
    bs["Interpolator"] = ["BSplineInterpolator"] * 3
    bs["ImageSampler"] = ["RandomCoordinate"] * 3
    po.AddParameterMap(bs)
    m.SetParameterObject(po); m.SetLogToConsole(False)
    m.UpdateLargestPossibleRegion()
    return R._deformation_field(itk, moving, m.GetTransformParameterObject())


def compose(residual, prior):
    """Total fixed->atlas displacement: x -> x + r(x) -> (x + r(x)) + prior(x + r(x))."""
    if prior is None:
        return tuple(np.asarray(v, np.float32) for v in residual)
    H, W = residual[0].shape
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    py, px = gy + residual[1], gx + residual[0]
    return tuple((residual[k] + ndi.map_coordinates(prior[k], [py, px], order=1, mode="nearest")).astype(np.float32)
                 for k in range(2))


def _ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def refine(reg, image_oriented, mask_oriented, prior_warp, pixel_um):
    """Return (warp, report).  warp is the full fixed->atlas field replacing prior_warp, or None to keep it."""
    mask = np.asarray(mask_oriented, bool)
    H, W = mask.shape
    if reg.control_points:
        return None, {"accepted": False, "reason": "已有手动修正点，不做深部纹理贴合。"}
    if mask.sum() < 1000:
        return None, {"accepted": False, "reason": "组织太小，不做深部纹理贴合。"}
    density = nuclear_density(image_oriented, mask, pixel_um)

    def labels_for(field):
        probe = R.Registration(**{**reg.__dict__, "nonlinear": field is not None, "control_points": []})
        return R.label_maps(probe, W, H, None if field is None else np.asarray(field))[0]

    lab0 = labels_for(prior_warp)
    synth0 = paint_labels(lab0, density, mask)
    interior = ndi.binary_erosion(mask, iterations=max(1, round(150. / pixel_um)))
    before = {"texture_ncc": _ncc(density, synth0, interior), "dice": R.dice(lab0 > 0, mask)}
    fixed_outline = ndi.gaussian_filter(mask.astype(np.float32), 2)
    field, lab, synth = prior_warp, lab0, synth0
    for _ in range(ROUNDS):
        moving_outline = ndi.gaussian_filter((lab > 0).astype(np.float32), 2)
        candidate = None
        for spacing in GRID_UM:
            try:
                residual = _bspline(density, synth, fixed_outline, moving_outline, max(8., spacing / pixel_um))
            except (RuntimeError, ImportError) as exc:
                return None, {"accepted": False, "reason": "深部纹理贴合失败，保留原结果。", "error": str(exc),
                              "before": before}
            if any(not np.isfinite(v).all() for v in residual):
                continue
            c = compose(residual, field)
            jac = R.min_jacobian(c)
            if np.isfinite(jac) and jac >= MIN_JACOBIAN:
                candidate = c
                break
        if candidate is None:
            break
        field = candidate
        lab = labels_for(field)
        synth = paint_labels(lab, density, mask)
    if field is prior_warp:
        return None, {"accepted": False, "reason": "深部纹理贴合出现折叠，保留原结果。", "before": before}
    after = {"texture_ncc": _ncc(density, synth, interior), "dice": R.dice(lab > 0, mask),
             "min_jacobian": R.min_jacobian(field)}
    if after["texture_ncc"] <= before["texture_ncc"] + .01 or after["dice"] < before["dice"] - .005:
        return None, {"accepted": False, "reason": "深部纹理检查未通过，保留原结果。", "before": before,
                      "candidate": after}
    return field, {"accepted": True, "reason": "深部纹理贴合", "before": before, "after": after}


def texture_score(reg, density, mask, interior, field):
    H, W = mask.shape
    lab = R.label_maps(reg, W, H, None if field is None else np.asarray(field))[0]
    return _ncc(density, paint_labels(lab, density, mask), interior)


def search_plane(reg, image_oriented, mask_oriented, prior_warp, pixel_um, rounds=3):
    """Coordinate search of the cutting plane (AP shift, DV and ML tilt) that best explains the DAPI texture.

    giRAff only searches straight coronal planes.  Here the plane is nudged with the in-plane fit held fixed;
    every candidate is scored by how well the region-painted atlas correlates with the section.
    Returns a dict of the changed Registration fields (possibly empty) and a report."""
    mask = np.asarray(mask_oriented, bool)
    density = nuclear_density(image_oriented, mask, pixel_um)
    interior = ndi.binary_erosion(mask, iterations=max(1, round(150. / pixel_um)))
    best = {"ap_auto": float(reg.ap_auto), "tilt_dv": float(reg.tilt_dv), "tilt_ml": float(reg.tilt_ml)}
    probe = lambda values: R.Registration(**{**reg.__dict__, **values, "control_points": [],
                                            "nonlinear": prior_warp is not None})
    base = best_score = texture_score(probe(best), density, mask, interior, prior_warp)
    steps = {"ap_auto": (4., 2., 1.), "tilt_dv": (2., 1., .5), "tilt_ml": (2., 1., .5)}
    limits = {"ap_auto": 16., "tilt_dv": 8., "tilt_ml": 8.}
    for level in range(rounds):
        for key in best:
            step = steps[key][min(level, 2)]
            for sign in (1., -1.):
                while True:
                    trial = {**best, key: best[key] + sign * step}
                    if abs(trial[key] - float(getattr(reg, key))) > limits[key]:
                        break
                    score = texture_score(probe(trial), density, mask, interior, prior_warp)
                    if score <= best_score + 1e-3:
                        break
                    best, best_score = trial, score
    changed = {k: v for k, v in best.items() if v != float(getattr(reg, k))}
    if best_score < base + .005:
        changed = {}
    return changed, {"before": base, "after": best_score, "changed": changed}
