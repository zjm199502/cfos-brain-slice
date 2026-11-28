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
TEXTURE = "nissl"        # "paint": every region its median section density (self-locks on thin layers that start
                         # off target); "layers": same, but principal cell layers painted bright; "nissl": the
                         # CCFv3-aligned Allen Nissl volume (ccf_data/allen_ccfv3_nissl_25um.npz), falls back to
                         # "paint" when it is not installed.
# Benchmark 2026-11-28 (S01-S03, mean µm, DG / CA / overall P90): old default (paint, flat, no pull) 56 / 74 / 217;
# paint+clahe+pull 57 / 45 / 181; nissl+clahe+pull 33 / 41 / 93.
# Experimental (off by default; benchmark 2026-11-28 in tools/reg_bench): the in-app DAPI ridge detector does not
# yet agree with the benchmark's, so neither the ridge term nor the per-structure check can be trusted to decide.
W_RIDGE = 0.             # weight of the DAPI cell-layer ridge term (atlas DG-sg/CA-sp mask vs detected ridges)
LAYER_GUARD = False      # reject a fit that moves the atlas DG-sg or CA-sp further from the DAPI ridges
LAYER_TOL_UM = 10.       # ... by more than this
SEARCH_TEXTURE = "paint" # the plane search with "layers"/"nissl" scoring jumped AP by up to 200 µm on S02
CELL_LAYERS = (632, 407, 446, 495)   # DG-sg, CA1sp, CA2sp, CA3sp (structure ids)
DG_LAYER = (632,)
HIPPOCAMPUS = 1080


def _indices(ids):
    from .atlas import load_atlas
    from .atlas_choice import CCF
    st = load_atlas(CCF).structures
    return st[st.path.astype(str).apply(lambda p: any(f"/{i}/" in f"/{p}/" for i in ids))]["index"].to_numpy()


_IDX = {}


def indices(name):
    if name not in _IDX:
        _IDX[name] = _indices({"layers": CELL_LAYERS, "dg": DG_LAYER, "ca": CELL_LAYERS[1:],
                               "ca1": (407, 446), "ca3": (495,),
                               "hip": (HIPPOCAMPUS,), "vent": (73,)}[name])
    return _IDX[name]


RIDGE_REACH_UM = 600.    # search for DAPI cell layers this far around the atlas hippocampus
RIDGE_MIN_LEN_UM = 450.  # a cell-layer piece is at least this long ...
RIDGE_MIN_WIDTH_UM = 42. # ... and this thick on average (vessels, fibre edges are thinner and branchy)


def cell_ridges(density, mask, labels, pixel_um):
    """DAPI principal cell layers (DG granule, CA pyramidal) around the atlas hippocampus.

    Bright-ridge filter (Sato, 12–30 µm) on the nuclear density within RIDGE_REACH_UM of the atlas hippocampus,
    kept where the ridge response and the density are both high, in long, thick pieces that are not on a
    ventricle wall.  Wide reach on purpose: dorsal CA1 can sit 300–500 µm from where the affine atlas puts it.
    Returns a bool mask (empty when the section has no hippocampus)."""
    from skimage.filters import sato
    from skimage.morphology import skeletonize
    hip = np.isin(labels, indices("hip"))
    if hip.sum() < 50:
        return np.zeros(mask.shape, bool)
    roi = (ndi.distance_transform_edt(~hip) * pixel_um <= RIDGE_REACH_UM) & \
          (ndi.distance_transform_edt(mask) * pixel_um > 150)
    if roi.sum() < 100:
        return np.zeros(mask.shape, bool)
    ridge = sato(density, sigmas=[v / pixel_um for v in (12., 20., 30.)], black_ridges=False)
    m = (ridge > np.percentile(ridge[roi], 88)) & (density > np.percentile(density[roi], 75)) & roi
    m = ndi.binary_closing(m, iterations=1)
    comp, k = ndi.label(m)
    vent = np.isin(labels, indices("vent"))
    keep = []
    for c in range(1, k + 1):
        sel = comp == c
        length = skeletonize(sel).sum() * pixel_um
        if length < RIDGE_MIN_LEN_UM or sel.sum() * pixel_um ** 2 / length < RIDGE_MIN_WIDTH_UM:
            continue
        if vent[sel].mean() <= .3:
            keep.append(c)
    return np.isin(comp, keep)


def layer_distances(ridges, labels, pixel_um):
    """Mean distance (µm) of DAPI ridge pixels to the atlas DG-sg and CA-sp, each ridge piece assigned to the
    nearer of the two in the given labels.  Returns {"dg": µm or None, "ca": µm or None}."""
    out = {"dg": None, "ca": None}
    if not ridges.any():
        return out
    d = {k: ndi.distance_transform_edt(~np.isin(labels, indices(k))) * pixel_um for k in ("dg", "ca")}
    comp, k = ndi.label(ridges)
    vals = {"dg": [], "ca": []}
    for c in range(1, k + 1):
        m = comp == c
        key = "dg" if np.median(d["dg"][m]) <= np.median(d["ca"][m]) else "ca"
        vals[key].append(d[key][m])
    for key, v in vals.items():
        if v:
            out[key] = float(np.concatenate(v).mean())
    return out


def ridge_classes(ridges, labels):
    """Freeze the DG/CA assignment of each ridge piece (from the labels before refinement)."""
    cls = np.zeros(ridges.shape, np.uint8)
    if not ridges.any():
        return cls
    d = {k: ndi.distance_transform_edt(~np.isin(labels, indices(k))) for k in ("dg", "ca")}
    comp, k = ndi.label(ridges)
    for c in range(1, k + 1):
        m = comp == c
        cls[m] = 1 if np.median(d["dg"][m]) <= np.median(d["ca"][m]) else 2
    return cls


def class_distances(cls, labels, pixel_um):
    out = {}
    for key, c in (("dg", 1), ("ca", 2)):
        sel = cls == c
        if sel.any():
            out[key] = float((ndi.distance_transform_edt(~np.isin(labels, indices(key))) * pixel_um)[sel].mean())
    return out


def pixel_classes(ridges, labels, pixel_um):
    """Per ridge pixel: 1 = DG, 2 = CA1/CA2, 3 = CA3 (nearest atlas layer, smoothed over ~40 µm).
    Per pixel, not per piece: the CA3 pyramidal layer often runs into the DG granule layer as one piece."""
    if not ridges.any():
        return np.zeros(ridges.shape, np.uint8)
    d = np.stack([ndi.distance_transform_edt(~np.isin(labels, indices(k))) for k in ("dg", "ca1", "ca3")])
    near = d.argmin(0)
    vote = np.stack([ndi.gaussian_filter((ridges & (near == i)).astype(np.float32), 40. / pixel_um)
                     for i in range(3)])
    return np.where(ridges, vote.argmax(0) + 1, 0).astype(np.uint8)


def pixel_class_distances(cls, labels, pixel_um):
    out = {}
    for key, c in (("dg", 1), ("ca1", 2), ("ca3", 3)):
        sel = cls == c
        if sel.any():
            out[key] = float((ndi.distance_transform_edt(~np.isin(labels, indices(key))) * pixel_um)[sel].mean())
    return out


_NISSL = {}


def nissl_volume():
    """The CCFv3-aligned Allen Nissl volume (AP, DV, ML at 25 µm, uint8), or None if it is not installed."""
    if "v" not in _NISSL:
        from .resources import allen_atlas_dir
        path = allen_atlas_dir() / "allen_ccfv3_nissl_25um.npz"
        try:
            with np.load(path, allow_pickle=False) as z:
                _NISSL["v"] = z["nissl"]
        except (OSError, KeyError, ValueError):
            _NISSL["v"] = None
    return _NISSL["v"]


def sample_nissl(reg, W, H, field):
    from .atlas import qn_to_asr
    pl = R.adjusted_plane(reg)
    t, s = np.mgrid[0:H, 0:W].astype(np.float32)
    if field is not None:
        s = s + field[0]; t = t + field[1]
    s = (s + .5) / W; t = (t + .5) / H
    p = pl.o[None, None, :] + s[..., None] * pl.u[None, None, :] + t[..., None] * pl.v[None, None, :]
    idx = qn_to_asr(p)
    return ndi.map_coordinates(nissl_volume(), [idx[..., 0], idx[..., 1], idx[..., 2]], order=1, mode="constant")


PREPROC = "clahe"        # DAPI local-contrast step before the density map: "flat" (divide by the BACKGROUND_UM
                         # background), "flat150" (150 µm background), "clahe", "zscore" (local mean/sd, 150 µm)


def _local_mean(x, m, s):
    # Normalised convolution keeps the tissue edge from darkening the background estimate.
    return ndi.gaussian_filter(np.where(m, x, 0), s) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), s), 1e-3)


def nuclear_density(image, mask, pixel_um, preproc=None):
    """Flat-field normalised DAPI density in [0, 1] inside the tissue (local-contrast step: PREPROC)."""
    preproc = preproc or PREPROC
    x = np.asarray(image, np.float32)
    m = np.asarray(mask, bool)
    if preproc == "clahe" and m.any():
        from skimage import exposure
        hi = float(np.percentile(x[m], 99.8)) or 1.
        x = exposure.equalize_adapthist(np.clip(x / hi, 0, 1), kernel_size=max(8, round(200. / pixel_um)),
                                        clip_limit=.02).astype(np.float32) * 255
    d = ndi.gaussian_filter(x, max(.7, DENSITY_SIGMA_UM / pixel_um))
    s = (150. if preproc in ("flat150", "zscore") else BACKGROUND_UM) / pixel_um
    bg = _local_mean(x, m, s)
    if preproc == "zscore":
        sd = np.sqrt(np.maximum(_local_mean(x * x, m, s) - bg * bg, 1e-6))
        z = np.where(m, (d - bg) / sd, 0)
        r = np.clip(z + 3, 0, None).astype(np.float32)
    else:
        r = np.where(m, d / (bg + 1.0), 0).astype(np.float32)
    hi = float(np.percentile(r[m], 99.5)) if m.any() else 1.
    return np.clip(r / max(hi, 1e-6), 0, 1)


def paint_labels(labels, density, mask, min_pixels=8, bright=None):
    """Synthetic DAPI: each atlas label painted with the median section density inside it.

    bright=(label indices, value): those labels get the fixed value instead.  Thin principal cell layers that
    start off target would otherwise be painted with whatever lies under them (often darker than the tissue)
    and never pull towards the real layer."""
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
    if bright is not None:
        out[np.isin(keys, bright[0])] = bright[1]
    synth = np.zeros(lab.shape, np.float32)
    synth[inside] = out[inverse]
    synth[(lab > 0) & ~mask] = fill
    return synth


def _write_points(path, points):
    with open(path, "w", encoding="ascii") as f:
        f.write(f"point\n{len(points)}\n")
        for x, y in points:
            f.write(f"{float(x):.4f} {float(y):.4f}\n")


def _bspline(fixed, moving, fixed2, moving2, grid_px, w_outline=.3, iters=400, points=None, w_points=.5,
             ridge=None, w_ridge=W_RIDGE):
    """B-spline on texture NCC + outline MI (+ landmarks); returns fixed->moving (dx, dy) in pixels.

    points: [((fixed_x, fixed_y), (moving_x, moving_y)), ...] in pixel-index coordinates."""
    import tempfile
    itk = R._itk()
    as_img = lambda a: itk.image_view_from_array(np.ascontiguousarray(a, dtype=np.float32))
    m = itk.ElastixRegistrationMethod[itk.Image[itk.F, 2], itk.Image[itk.F, 2]].New()
    m.SetFixedImage(as_img(fixed)); m.SetMovingImage(as_img(moving))
    metrics = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    weights = [1. - w_outline, w_outline, .05]
    pairs = [(fixed2, moving2)] * 2
    if ridge is not None:
        # DAPI ridges (fixed) vs atlas cell layers (moving), both blurred so the metric sees them from afar.
        metrics.append("AdvancedMeanSquares")
        weights.append(w_ridge)
        pairs.append(ridge)
    if points:
        metrics.append("CorrespondingPointsEuclideanDistanceMetric")
        weights.append(w_points)
        pairs.append((fixed2, moving2))
    # Multi-metric elastix needs one image pair, interpolator, sampler and pyramid per metric,
    # even for the bending penalty and the landmark term, which ignore the images.
    for f2, m2 in pairs:
        m.AddFixedImage(as_img(f2)); m.AddMovingImage(as_img(m2))
    po = itk.ParameterObject.New()
    bs = po.GetDefaultParameterMap("bspline", 3, float(grid_px))
    bs["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    bs["Metric"] = metrics
    for i, w in enumerate(weights):
        bs[f"Metric{i}Weight"] = [str(w)]
    bs["FinalGridSpacingInPhysicalUnits"] = [str(float(grid_px))]
    bs["GridSpacingSchedule"] = ["4", "2", "1"]
    bs["MaximumNumberOfIterations"] = [str(iters)]
    bs["NumberOfSpatialSamples"] = ["6000"]
    n = len(metrics)
    bs["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"] * n
    bs["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"] * n
    bs["Interpolator"] = ["BSplineInterpolator"] * n
    bs["ImageSampler"] = ["RandomCoordinate"] * n
    po.AddParameterMap(bs)
    m.SetParameterObject(po); m.SetLogToConsole(False)
    with tempfile.TemporaryDirectory(prefix="fos-landmarks-") as tmp:
        if points:
            _write_points(f"{tmp}/fixed.txt", [p[0] for p in points])
            _write_points(f"{tmp}/moving.txt", [p[1] for p in points])
            m.SetFixedPointSetFileName(f"{tmp}/fixed.txt")
            m.SetMovingPointSetFileName(f"{tmp}/moving.txt")
        m.UpdateLargestPossibleRegion()
    return R._deformation_field(itk, moving, m.GetTransformParameterObject())


def invert_at(field, target, iters=30):
    """Point q with q + field(q) = target (field is fixed->atlas, pixel-index coordinates)."""
    target = np.asarray(target, float)
    if field is None:
        return target
    q = target.copy()
    for _ in range(iters):
        d = np.array([ndi.map_coordinates(field[k], [[q[1]], [q[0]]], order=1, mode="nearest")[0] for k in range(2)])
        q = target - d
    return q


def displaced(field, point):
    """Atlas-plane position that a section pixel samples: x + field(x)."""
    x = np.asarray(point, float)
    if field is None:
        return x
    return x + np.array([ndi.map_coordinates(field[k], [[x[1]], [x[0]]], order=1, mode="nearest")[0]
                         for k in range(2)])


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


def landmark_error(field, landmarks):
    """Distances (pixels) between where each landmark samples the atlas and where it should."""
    return [float(np.hypot(*(displaced(field, f) - np.asarray(a, float)))) for f, a in landmarks]


LAYER_PULL = True        # after the texture fit, pull atlas CA-sp onto the DAPI CA ridges with DG pinned
PULL_REACH_UM = 500.     # a CA ridge pixel looks this far for the atlas CA-sp it belongs to
PULL_SIGMA_UM = 160.     # smoothness of the pull (100 µm folded on S01/S03; 160 µm: CA 74 -> 58 µm on the benchmark)
PULL_ROUNDS = 25
PULL_STEP_UM = 20.       # largest move per round (keeps each update fold-free)
PULL_STEP_JAC = .6       # each round's own displacement keeps its Jacobian above this


COMBINE = ()                   # e.g. ("nissl", "paint"): textures tried per section; the one closest to the DAPI cell
                               # layers is kept (each finished with layer_pull).  () = TEXTURE only.  Off by default:
                               # on S01-S03 Nissl won every section, so trying both only doubled the time.


def refine(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks=()):
    """Texture fit (texture_refine) + CA1/CA3 layer pull (layer_pull), for each texture in COMBINE; the result
    closest to the DAPI cell layers wins (old region painting and Allen Nissl each win on some sections)."""
    global TEXTURE
    textures = [t for t in COMBINE if t != "nissl" or nissl_volume() is not None] or [TEXTURE]
    if landmarks or reg.fit_atlas != "CCFv3" or len(textures) == 1:
        if len(textures) == 1:
            return _refine_one(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks, textures[0])
        return _refine_one(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks, TEXTURE)
    mask = np.asarray(mask_oriented, bool)
    density = nuclear_density(image_oriented, mask, pixel_um)
    H, W = mask.shape
    labels_for = lambda f: R.label_maps(R.Registration(**{**reg.__dict__, "nonlinear": f is not None,
                                                         "control_points": []}), W, H,
                                        None if f is None else np.asarray(f))[0]
    lab0 = labels_for(prior_warp)
    cls = pixel_classes(cell_ridges(density, mask, lab0, pixel_um), lab0, pixel_um)
    best = None
    scores = {}
    for tex in textures:
        warp, report = _refine_one(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks, tex)
        lab = labels_for(warp if warp is not None else prior_warp)
        score = _layer_score(cls, lab, pixel_um)
        scores[tex] = score
        if best is None or score < best[0]:
            best = (score, warp, report, tex)
    _, warp, report, tex = best
    return warp, {**report, "texture_choice": {"chosen": tex, "layer_um": scores}}


def _layer_score(cls, labels, pixel_um):
    """Mean distance (µm) of all DG/CA1/CA3 ridge pixels to their atlas layer; inf without ridges."""
    total, n = 0., 0
    for key, c in (("dg", 1), ("ca1", 2), ("ca3", 3)):
        sel = cls == c
        if sel.any():
            total += float((ndi.distance_transform_edt(~np.isin(labels, indices(key))) * pixel_um)[sel].sum())
            n += int(sel.sum())
    return total / n if n else float("inf")


def _refine_one(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks, texture):
    global TEXTURE
    saved, TEXTURE = TEXTURE, texture
    try:
        warp, report = texture_refine(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks)
    finally:
        TEXTURE = saved
    report = {**report, "texture": texture}
    if not LAYER_PULL or landmarks or reg.fit_atlas != "CCFv3":
        return warp, report
    base = warp if warp is not None else prior_warp
    mask = np.asarray(mask_oriented, bool)
    density = nuclear_density(image_oriented, mask, pixel_um)
    pulled, pull_report = layer_pull(reg, base, density, mask, pixel_um, class_field=prior_warp)
    report = {**report, "layer_pull": pull_report}
    if pulled is None:
        return warp, report
    reason = report["reason"] + "，CA 细胞层对齐" if report.get("accepted") else "CA 细胞层对齐"
    return pulled, {**report, "accepted": True, "reason": reason}


_SAME = object()


def layer_pull(reg, field, density, mask, pixel_um, class_field=_SAME):
    """Move the atlas CA1 and CA3 pyramidal layers onto the DAPI ridges while the dentate gyrus stays put.

    A smooth texture fit cannot do both when the section's DG–CA1 distance differs from the atlas: fitting DG
    drags dorsal CA1 100–300 µm away (benchmark 2026-11-28).  Here every CA1 (CA3) ridge pixel is paired with
    the nearest atlas CA1 (CA3) sp pixel (within PULL_REACH_UM), DG ridges and the tissue outline are pinned, and the
    residuals are smoothed by normalised Gaussian convolution, a few damped rounds (ICP-like).
    class_field: the field whose labels decide which ridge pixel is DG, CA1 or CA3 (default: field).  Use the
    pre-texture field: a texture fit that drags CA3 onto the DG would otherwise get its error pinned in place.
    Returns (field or None, report); None when it did not help or would fold."""
    from skimage.morphology import skeletonize
    H, W = mask.shape
    probe = lambda f: R.Registration(**{**reg.__dict__, "nonlinear": f is not None, "control_points": []})
    labels_for = lambda f: R.label_maps(probe(f), W, H, None if f is None else np.asarray(f))[0]
    lab = labels_for(field)
    ridges = cell_ridges(density, mask, lab, pixel_um)
    if not ridges.any():
        return None, {"applied": False, "reason": "没有检测到海马细胞层"}
    cls = pixel_classes(ridges, lab if class_field is _SAME else labels_for(class_field), pixel_um)
    dg_r = cls == 1
    if (cls >= 2).sum() * pixel_um < 300:
        return None, {"applied": False, "reason": "没有检测到 CA 锥体层"}
    before = pixel_class_distances(cls, lab, pixel_um)
    sigma = PULL_SIGMA_UM / pixel_um
    floor = np.exp(-2.) / (2 * np.pi * sigma * sigma)        # one sample two sigmas away
    edge = mask ^ ndi.binary_erosion(mask)
    pins = [np.argwhere(dg_r)[::3], np.argwhere(edge)[::6]]
    groups = [(np.argwhere(cls == c)[::2], key) for c, key in ((2, "ca1"), (3, "ca3")) if (cls == c).any()]
    cur = field
    jac_floor = min(MIN_JACOBIAN, R.min_jacobian(field) if field is not None else 1.)
    for _ in range(PULL_ROUNDS):
        lab = labels_for(cur)
        num = np.zeros((2, H, W), np.float32); den = np.zeros((H, W), np.float32)
        for pts, key in groups:
            # each pyramidal-layer pixel looks for its own atlas layer: CA1 ridges for CA1sp, CA3 ridges for CA3sp
            target = np.isin(lab, indices(key))
            if not target.any():
                continue
            dist, (iy, ix) = ndi.distance_transform_edt(~target, return_indices=True)
            q = pts[dist[pts[:, 0], pts[:, 1]] * pixel_um <= PULL_REACH_UM]
            a_y, a_x = iy[q[:, 0], q[:, 1]], ix[q[:, 0], q[:, 1]]
            np.add.at(num[0], (q[:, 0], q[:, 1]), a_x - q[:, 1])
            np.add.at(num[1], (q[:, 0], q[:, 1]), a_y - q[:, 0])
            np.add.at(den, (q[:, 0], q[:, 1]), 1.)
        for pts in pins:
            np.add.at(den, (pts[:, 0], pts[:, 1]), 1.)
        g = [ndi.gaussian_filter(v, sigma) for v in (num[0], num[1], den)]
        scale = 1. / np.maximum(g[2], floor)
        residual = [g[0] * scale, g[1] * scale]
        peak = float(np.hypot(*residual).max()) * pixel_um
        if peak < 2:
            break
        if peak > PULL_STEP_UM:
            residual = [v * (PULL_STEP_UM / peak) for v in residual]
        if not R.min_jacobian(tuple(residual)) >= PULL_STEP_JAC:
            residual = [v * .5 for v in residual]
            if not R.min_jacobian(tuple(residual)) >= PULL_STEP_JAC:
                break
        nxt = compose(residual, cur)
        if not R.min_jacobian(nxt) >= jac_floor:
            break
        cur = nxt
    if cur is field:
        return None, {"applied": False, "reason": "CA 对齐会折叠或无需移动", "before": before}
    after = pixel_class_distances(cls, labels_for(cur), pixel_um)
    worse = [k for k in after if after[k] > before[k] + LAYER_TOL_UM]
    gain = sum(before[k] - after[k] for k in ("ca1", "ca3") if k in after)
    ok = not worse and gain > LAYER_TOL_UM
    report = {"applied": bool(ok), "before": before, "after": after, "min_jacobian": R.min_jacobian(cur)}
    return (cur if ok else None), report


def texture_refine(reg, image_oriented, mask_oriented, prior_warp, pixel_um, landmarks=()):
    """Return (warp, report).  warp is the full fixed->atlas field replacing prior_warp, or None to keep it.

    landmarks: [((section_x, section_y), (atlas_x, atlas_y)), ...] in pixel-index coordinates of the
    oriented section and of the affine atlas plane: the atlas content at atlas_xy belongs at section_xy.
    Control points are the caller's business; reg.control_points is ignored here."""
    mask = np.asarray(mask_oriented, bool)
    H, W = mask.shape
    landmarks = [(np.asarray(f, float), np.asarray(a, float)) for f, a in landmarks]
    if mask.sum() < 1000:
        return None, {"accepted": False, "reason": "组织太小，不做深部纹理贴合。"}
    density = nuclear_density(image_oriented, mask, pixel_um)

    def labels_for(field):
        probe = R.Registration(**{**reg.__dict__, "nonlinear": field is not None, "control_points": []})
        return R.label_maps(probe, W, H, None if field is None else np.asarray(field))[0]

    lab0 = labels_for(prior_warp)
    texture = TEXTURE if TEXTURE != "nissl" or nissl_volume() is not None else "paint"
    ridges = cell_ridges(density, mask, lab0, pixel_um)
    bright = None
    if texture == "layers" and ridges.any():
        bright = (indices("layers"), float(np.median(density[ridges])))

    def synth_for(field, lab):
        if texture == "nissl":
            nis = sample_nissl(reg, W, H, None if field is None else np.asarray(field)).astype(np.float32)
            return nuclear_density(nis, lab > 0, pixel_um)
        return paint_labels(lab, density, mask, bright=bright)

    blur = max(1., 40. / pixel_um)
    fixed_ridge = ndi.gaussian_filter(ridges.astype(np.float32), blur) if ridges.any() else None
    moving_ridge = lambda lab: ndi.gaussian_filter(np.isin(lab, indices("layers")).astype(np.float32), blur)
    cls = ridge_classes(ridges, lab0)
    synth0 = synth_for(prior_warp, lab0)
    interior = ndi.binary_erosion(mask, iterations=max(1, round(150. / pixel_um)))
    before = {"texture_ncc": _ncc(density, synth0, interior), "dice": R.dice(lab0 > 0, mask),
              "layers_um": class_distances(cls, lab0, pixel_um), "texture": texture}
    if landmarks:
        before["landmark_um"] = [e * pixel_um for e in landmark_error(prior_warp, landmarks)]
    fixed_outline = ndi.gaussian_filter(mask.astype(np.float32), 2)
    field, lab, synth = prior_warp, lab0, synth0
    for _ in range(ROUNDS):
        moving_outline = ndi.gaussian_filter((lab > 0).astype(np.float32), 2)
        candidate = None
        # Landmarks in the current moving frame: the point that the current field sends to the atlas target.
        points = [(f, invert_at(field, a)) for f, a in landmarks]
        for spacing in GRID_UM:
            try:
                residual = _bspline(density, synth, fixed_outline, moving_outline, max(8., spacing / pixel_um),
                                    points=points,
                                    ridge=None if fixed_ridge is None or W_RIDGE <= 0 else (fixed_ridge, moving_ridge(lab)))
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
        synth = synth_for(field, lab)
    if field is prior_warp:
        return None, {"accepted": False, "reason": "深部纹理贴合出现折叠，保留原结果。", "before": before}
    after = {"texture_ncc": _ncc(density, synth, interior), "dice": R.dice(lab > 0, mask),
             "min_jacobian": R.min_jacobian(field), "layers_um": class_distances(cls, lab, pixel_um)}
    worse = [k for k, v in after["layers_um"].items() if v > before["layers_um"].get(k, np.inf) + LAYER_TOL_UM]
    if landmarks:
        after["landmark_um"] = [e * pixel_um for e in landmark_error(field, landmarks)]
        # Landmarks are the user's explicit decision; only reject if they are not honoured or the outline breaks.
        ok = max(after["landmark_um"]) <= max(3 * pixel_um, 40.) and after["dice"] >= before["dice"] - .01
        if not ok:
            return None, {"accepted": False, "reason": "标志点没有贴合到位，保留原结果。", "before": before,
                          "candidate": after}
        return field, {"accepted": True, "reason": "深部纹理贴合（含标志点）", "before": before, "after": after}
    if worse and LAYER_GUARD:
        return None, {"accepted": False, "reason": "深部纹理贴合让海马细胞层（" + "、".join(
            {"dg": "DG", "ca": "CA"}[k] for k in worse) + "）离 DAPI 更远，保留原结果。", "before": before,
                      "candidate": after}
    improved = LAYER_GUARD and any(after["layers_um"][k] < before["layers_um"][k] - LAYER_TOL_UM
                                   for k in after["layers_um"] if k in before["layers_um"])
    if (after["texture_ncc"] <= before["texture_ncc"] + .01 and not improved) or after["dice"] < before["dice"] - .005:
        return None, {"accepted": False, "reason": "深部纹理检查未通过，保留原结果。", "before": before,
                      "candidate": after}
    return field, {"accepted": True, "reason": "深部纹理贴合", "before": before, "after": after}


def texture_score(reg, density, mask, interior, field, texture="paint", bright=None, pixel_um=8.):
    H, W = mask.shape
    f = None if field is None else np.asarray(field)
    if texture == "nissl":
        lab = R.label_maps(reg, W, H, f)[0]
        synth = nuclear_density(sample_nissl(reg, W, H, f).astype(np.float32), lab > 0, pixel_um)
    else:
        synth = paint_labels(R.label_maps(reg, W, H, f)[0], density, mask, bright=bright)
    return _ncc(density, synth, interior)


def texture_setup(reg, density, mask, field, pixel_um, texture=None):
    """(texture, bright) for the given texture (default TEXTURE), as refine() and search_plane() use them."""
    texture = texture or TEXTURE
    texture = texture if texture != "nissl" or nissl_volume() is not None else "paint"
    bright = None
    if texture == "layers":
        H, W = mask.shape
        lab = R.label_maps(R.Registration(**{**reg.__dict__, "nonlinear": field is not None, "control_points": []}),
                           W, H, None if field is None else np.asarray(field))[0]
        ridges = cell_ridges(density, mask, lab, pixel_um)
        if ridges.any():
            bright = (indices("layers"), float(np.median(density[ridges])))
    return texture, bright


def search_plane(reg, image_oriented, mask_oriented, prior_warp, pixel_um, rounds=3, ap_bounds=None):
    """Coordinate search of the cutting plane (AP shift, DV and ML tilt) that best explains the DAPI texture.

    giRAff only searches straight coronal planes.  Here the plane is nudged with the in-plane fit held fixed;
    every candidate is scored by how well the synthetic atlas texture (see TEXTURE) correlates with the section.
    ap_bounds=(lo, hi) limits the AP change (voxels, relative to the current ap_auto), e.g. to stay where the
    slice order of the series allows; the default is the symmetric AP limit below.
    Returns a dict of the changed Registration fields (possibly empty) and a report."""
    mask = np.asarray(mask_oriented, bool)
    density = nuclear_density(image_oriented, mask, pixel_um)
    interior = ndi.binary_erosion(mask, iterations=max(1, round(150. / pixel_um)))
    best = {"ap_auto": float(reg.ap_auto), "tilt_dv": float(reg.tilt_dv), "tilt_ml": float(reg.tilt_ml)}
    probe = lambda values: R.Registration(**{**reg.__dict__, **values, "control_points": [],
                                            "nonlinear": prior_warp is not None})
    texture, bright = texture_setup(reg, density, mask, prior_warp, pixel_um, SEARCH_TEXTURE)
    score = lambda r: texture_score(r, density, mask, interior, prior_warp, texture=texture, bright=bright,
                                    pixel_um=pixel_um)
    base = best_score = score(probe(best))
    steps = {"ap_auto": (4., 2., 1.), "tilt_dv": (2., 1., .5), "tilt_ml": (2., 1., .5)}
    limits = {"ap_auto": 16., "tilt_dv": 8., "tilt_ml": 8.}
    lo, hi = ap_bounds if ap_bounds is not None else (-limits["ap_auto"], limits["ap_auto"])
    bounds = {"ap_auto": (max(lo, -limits["ap_auto"]), min(hi, limits["ap_auto"])),
              "tilt_dv": (-limits["tilt_dv"], limits["tilt_dv"]), "tilt_ml": (-limits["tilt_ml"], limits["tilt_ml"])}
    for level in range(rounds):
        for key in best:
            step = steps[key][min(level, 2)]
            for sign in (1., -1.):
                while True:
                    trial = {**best, key: best[key] + sign * step}
                    delta = trial[key] - float(getattr(reg, key))
                    if not bounds[key][0] - 1e-9 <= delta <= bounds[key][1] + 1e-9:
                        break
                    value = score(probe(trial))
                    if value <= best_score + 1e-3:
                        break
                    best, best_score = trial, value
    changed = {k: v for k, v in best.items() if v != float(getattr(reg, k))}
    if best_score < base + .005:
        changed = {}
    report = {"before": base, "after": best_score, "changed": changed, "texture": texture}
    if ap_bounds is not None:
        report["ap_bounds"] = [float(v) for v in bounds["ap_auto"]]
    return changed, report
