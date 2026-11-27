"""Display curves between categorical atlas labels, in pixel-corner (row, column) coordinates."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _crossings(rows):
    """Find transitions in each row and the narrower adjoining run at each transition."""
    row, pos = np.nonzero(rows[:, :-1] != rows[:, 1:])
    pos += 1
    width = np.minimum(pos, rows.shape[1] - pos)
    gap = np.where(row[1:] == row[:-1], np.diff(pos), rows.shape[1])
    np.minimum(width[:-1], gap, out=width[:-1])
    np.minimum(width[1:], gap, out=width[1:])
    return row, pos, width


def _boundary_graph(labels):
    # Each interface is one set of unit segments, even when both sides are
    # nonzero labels. Edges clipped by the image frame terminate at the frame.
    stride = labels.shape[1] + 1
    neighbours = defaultdict(list)
    clearance = {}
    for horizontal in (False, True):
        row, pos, width = _crossings(labels.T if horizontal else labels)
        start = pos * stride + row if horizontal else row * stride + pos
        offset = 1 if horizontal else stride
        for a, gap in zip(start.tolist(), width.tolist()):
            b = a + offset
            neighbours[a].append(b)
            neighbours[b].append(a)
            clearance[a] = min(clearance.get(a, gap), gap)
            clearance[b] = min(clearance.get(b, gap), gap)
    return neighbours, clearance


def _chains(neighbours):
    # Split at every true junction, including four-way diagonal contacts.
    # Consume both incidences so a shared border can never be drawn twice.
    stops = {v for v, edges in neighbours.items() if len(edges) != 2}

    def walk(start):
        chain = [start]
        v = start
        while neighbours[v]:
            w = neighbours[v].pop()
            neighbours[w].remove(v)
            chain.append(w)
            v = w
            if v in stops or v == start:
                break
        return chain

    for start in sorted(stops):
        while neighbours[start]:
            yield walk(start), False
    for start, edges in neighbours.items():
        if edges:
            yield walk(start), True


def _thin(points):
    """Decimate nearly straight portions with subpixel error; keep the endpoints."""
    for _ in range(4):
        if len(points) <= 6:
            break
        odd = np.arange(1, len(points) - 1, 2)
        midpoint = (points[odd - 1] + points[odd + 1]) * 0.5
        redundant = np.max(np.abs(midpoint - points[odd]), axis=1) < 0.015
        if not redundant.any():
            break
        keep = np.ones(len(points), dtype=bool)
        keep[odd[redundant]] = False
        points = points[keep]
    return points


def _curve(vertices, clearance, stride, voxel_px, ring):
    ids = np.asarray(vertices)
    corners = np.column_stack(np.divmod(ids, stride)).astype(np.float32)
    if len(corners) == 2:
        return corners
    # Smooth at quarter-pixel intervals on a uniform arclength grid.
    # Pins at junctions (and image edges) preserve registration coordinates.
    # Short open chains can themselves return to their pinned junction.
    subdivisions = 4
    t = np.arange(subdivisions, dtype=np.float32) / subdivisions
    samples = corners[:-1, None, :] + (corners[1:] - corners[:-1])[:, None, :] * t[None, :, None]
    points = np.concatenate((samples.reshape(-1, 2), corners[-1:]))
    gaps = np.array([clearance[v] for v in vertices], np.float32)
    limit = np.r_[np.repeat(np.minimum(gaps[:-1], gaps[1:]), subdivisions), gaps[-1]]
    limit[::subdivisions] = gaps
    # Displacement is limited by both atlas resolution and local label
    # thickness. Adjacent sides of a narrow ribbon retain separate traces.
    limit *= 0.30
    np.minimum(limit, 0.55 * voxel_px, out=limit)
    sigma = min(subdivisions * 1.2 * voxel_px, max(1.8, (len(points) - 1) / 9))
    if ring:
        points, limit = points[:-1], limit[:-1]
        smooth = gaussian_filter1d(points, sigma, axis=0, mode="wrap")
    else:
        smooth = gaussian_filter1d(points, sigma, axis=0, mode="nearest")
        # Taper the displacement over the first and last few samples so short
        # branches approach a shared junction without rounding it away.
        distance = np.arange(len(points))
        taper = np.minimum(distance, distance[::-1]) / max(1, sigma)
        limit *= np.minimum(taper, 1)
    delta = smooth - points
    norm = np.hypot(delta[:, 0], delta[:, 1])
    points += delta * np.minimum(1, limit / np.maximum(norm, 1e-9))[:, None]
    # The pointwise clearance constraint can leave small corners at its active
    # limit. Fair those corners on a shorter scale with a smooth bounded step.
    fair_sigma = min(.40 * subdivisions * voxel_px, (len(points)-1)/10)
    if fair_sigma > .5:
        fair = gaussian_filter1d(points, fair_sigma, axis=0, mode="wrap" if ring else "nearest") - points
        bound = np.minimum(.22 * voxel_px, .65 * limit)
        length = np.linalg.norm(fair, axis=1)
        points += fair * (bound / np.hypot(bound, np.maximum(length, 1e-9)))[:,None]
    if ring:
        points = np.concatenate((points, points[:1]))
    return _thin(points)


def boundary_paths(labels: np.ndarray, voxel_px: float = 3.0) -> list[np.ndarray]:
    """Return display-only shared interfaces; each original pixel spans [i, i+1].

    voxel_px is the size of an atlas voxel projected into the preview label
    map. No outline is invented where a region simply meets the image frame.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("The registration preview needs a two-dimensional label map.")
    if not labels.size:
        return []
    voxel_px = float(np.clip(voxel_px, 1.0, 24.0)) if np.isfinite(voxel_px) else 3.0
    neighbours, clearance = _boundary_graph(labels)
    stride = labels.shape[1] + 1
    return [_curve(ids, clearance, stride, voxel_px, ring) for ids, ring in _chains(neighbours)]


def registration_paths(reg, width, height, warp, pixel_um):
    """Sample the physical plane more finely, then smooth its shared interfaces."""
    from . import registration as R
    from .edge_drag import is_fine
    from dataclasses import replace
    base = replace(reg, control_points=[p for p in reg.control_points if not is_fine(p)])
    # Typically 8 -> 4 µm. Use tiled sampling and cap label pixels on large slides.
    factor = min(2.0, max(1., pixel_um / 4.), max(1., np.sqrt(6_000_000 / (width*height))))
    nw, nh = round(width*factor), round(height*factor)
    fx, fy = nw/width, nh/height
    lab = R.sample_grid(base, width, height, warp, (np.arange(nw)+.5)/fx,
                        (np.arange(nh)+.5)/fy, ("annotation",))["annotation"]
    plane = R.adjusted_plane(base)
    voxel_px = min(nw/max(np.linalg.norm(plane.u),1e-6), nh/max(np.linalg.norm(plane.v),1e-6))
    scale = np.array([fy,fx], np.float32)
    if reg.fit_atlas == "STAM":
        voxel_px *= 40 / 25
    return [p/scale for p in boundary_paths(lab,voxel_px)]
