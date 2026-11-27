"""Candidate adapter for the FosFolio 0.9 registration worker."""
import numpy as np
from .surface_refine import refine_registration


def preview_hippocampal_alignment(reg, oriented_channel, prior_field, pixel_um, ids, render):
    """Return a candidate and QC. The caller owns preview, persistence and acceptance.

    render is cfos.core.registration.label_maps(reg, W, H, field).
    prior_field is the existing nonlinear field on the oriented image grid.
    """
    if prior_field is None or not reg.nonlinear:
        return prior_field, {"applied": False, "reason": "requires prealigned nonlinear atlas"}
    if reg.control_points:
        return prior_field, {"applied": False, "reason": "manual control points present"}
    if oriented_channel.ndim != 2:
        raise ValueError("Choose a single oriented structural channel")
    h, w = oriented_channel.shape
    labels, hemis = render(reg, w, h, prior_field)
    def render_labels(field):
        return render(reg, w, h, field)[0]
    candidate, report, _ = refine_registration(oriented_channel, labels, hemis, prior_field,
                                               pixel_um, ids, render_labels)
    if report["applied"] and (candidate.shape != (2,h,w) or candidate.dtype != np.float32):
        raise ValueError("Unexpected candidate field format")
    return candidate, report
