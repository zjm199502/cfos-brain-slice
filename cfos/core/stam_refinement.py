"""Conservative residual B-spline fitting of an already aligned, native STAM section."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import registration as registration


def tissue_mask(reference, pixel_um=32., threshold=8):
    """Extract the bilateral tissue outline from the STAM Nissl reference."""
    iterations = max(1, round(96. / pixel_um))
    # Preserve tissue where a physical slice meets the field-of-view boundary.
    foreground = np.pad(reference > threshold, iterations, mode="edge")
    foreground = ndi.binary_closing(foreground, iterations=iterations)
    foreground = foreground[iterations:-iterations, iterations:-iterations]
    components, number = ndi.label(foreground)
    if not number:
        return foreground
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return ndi.binary_fill_holes(components == sizes.argmax())


def outline_metrics(a, b, pixel_um):
    """Overlap and bidirectional surface distances; HD95 is the larger directional P95."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    edge_a = a ^ ndi.binary_erosion(a)
    edge_b = b ^ ndi.binary_erosion(b)
    ab = ndi.distance_transform_edt(~edge_b)[edge_a] * pixel_um
    ba = ndi.distance_transform_edt(~edge_a)[edge_b] * pixel_um
    return {"dice": float(2 * np.sum(a & b) / (a.sum() + b.sum())),
            "hd95_um": float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
            "assd_um": float((ab.sum() + ba.sum()) / (ab.size + ba.size))}


def resample(image, warp, order):
    if warp is None:
        return image
    y, x = np.indices(image.shape, dtype=np.float32)
    return ndi.map_coordinates(image, (y + warp[1], x + warp[0]), order=order, mode="constant")


def refine(image, mask, reference, tissue, pixel_um=None):
    """Return a fixed-to-moving (dx,dy) residual in input pixels, or None to retain the affine fit."""
    pixel_um = float(pixel_um or 32.)
    if not np.isfinite(pixel_um) or pixel_um <= 0:
        raise ValueError("像素尺寸必须是有限正数。")
    image, reference = np.asarray(image), np.asarray(reference)
    mask, tissue = np.asarray(mask, bool), np.asarray(tissue, bool)
    if not (image.ndim == 2 and image.shape == mask.shape == tissue.shape == reference.shape):
        raise ValueError("脑片、参考图和组织掩膜的二维尺寸必须一致。")
    if min(image.shape) < 32 or min(mask.sum(), tissue.sum()) < 100:
        return None, {"accepted": False, "reason": "组织掩膜不足，保留仿射配准。"}
    before = outline_metrics(mask, tissue, pixel_um)
    radius = max(1., 64. / pixel_um)
    depth = ndi.distance_transform_edt(mask) * pixel_um
    interior = (depth >= 192.) & tissue
    # Compare every candidate on the same pixels, including the baseline.
    if interior.sum() < 100:
        return None, {"accepted": False, "reason": "缺少可评估的内部组织。", "before": before}
    before["interior_mi"] = registration.similarity(reference, image, interior)
    base = {"accepted": False, "before": before, "after": before, "reason": "保留仿射配准"}
    if np.all(mask == tissue):
        return None, {**base, "reason": "外轮廓已经重合。"}
    fixed = ndi.gaussian_filter(mask.astype("float32"), radius)
    moving = ndi.gaussian_filter(tissue.astype("float32"), radius)
    def texture(arr):
        blurred = ndi.gaussian_filter(arr.astype("float32"), max(.7, radius / 2))
        return blurred / max(float(blurred.max()), 1.)
    try:
        raw = registration._elastix(fixed, moving, texture(image), texture(reference),
                                    grid=max(12., 1000. / pixel_um))
    except (RuntimeError, ImportError) as exc:
        return None, {**base, "reason": "非线性拟合失败，保留仿射配准。", "error": str(exc)}
    if any(not np.isfinite(v).all() for v in raw):
        return None, {**base, "reason": "位移场含无效值，保留仿射配准。"}
    # Concentrate corrections near the tissue edge; keep the affine solution in deep tissue.
    frac = np.clip(depth / 1000., 0., 1.).astype("float32")
    weight = 1. - frac * frac * (3. - 2. * frac)
    warp = tuple(np.asarray(v * weight, np.float32) for v in raw)
    jac = registration.min_jacobian(warp)
    if not np.isfinite(jac) or jac < .15:
        return None, {**base, "reason": "位移折叠或局部压缩过强，保留仿射配准。", "min_jacobian": jac}
    aligned_mask = resample(tissue.astype(np.uint8), warp, 0).astype(bool)
    if not aligned_mask.any():
        return None, {**base, "reason": "变换后组织移出视野，保留仿射配准。"}
    after = outline_metrics(mask, aligned_mask, pixel_um)
    after["interior_mi"] = registration.similarity(resample(reference, warp, 1), image, interior)
    if (after["dice"] <= before["dice"] or after["assd_um"] >= before["assd_um"]
            or after["hd95_um"] > before["hd95_um"] or after["interior_mi"] < .95 * before["interior_mi"]):
        return None, {**base, "reason": "轮廓或内部纹理检查未通过，保留仿射配准。", "candidate": after}
    return warp, {"accepted": True, "reason": "局部非线性贴合", "before": before, "after": after,
                  "min_jacobian": float(jac)}
