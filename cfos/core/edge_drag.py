"""Invertible, compact edge edits in continuous section pixel-corner coordinates."""
import numpy as np


def is_fine(point):
    return len(point) >= 6 and point[5] == 2 and point[4] > 0


def landmark_targets(points, width, height):
    """Current positions of protected landmarks after replaying saved edits."""
    anchors = np.empty((0, 2))
    for point in points:
        if not is_fine(point):
            continue
        if len(anchors):
            anchors = np.column_stack(transport(anchors[:, 0], anchors[:, 1], point, width, height))
        if len(point) > 6 and point[6] == 1:
            anchors = np.vstack((anchors, np.asarray(point[2:4]) * [width, height]))
    return anchors


def protected_radius(anchors, start, end, radius):
    """Limit the new edit's swept support so it fixes the earlier targets."""
    if not len(anchors):
        return radius
    v = np.subtract(end, start)
    t = np.clip((anchors - start) @ v / max(v @ v, 1e-16), 0, 1)
    d = np.linalg.norm(anchors - (start + t[:, None] * v), axis=1)
    return min(radius, float(d.min()))


def landmark_radius(points, start, end, width, height, radius):
    return protected_radius(landmark_targets(points, width, height), start, end, radius)


def transport(x, y, point, width, height, inverse=False):
    """Map a stroke in either direction. The support is its swept radius.

    Along each line parallel to the drag, translate in a coordinate which
    tends to infinity at either end. This fixes both ends with unit derivative
    and has an exact inverse even when the drag is longer than its radius.
    """
    sx, sy, dx, dy, radius = np.array(point[:5]) * [width, height, width, height, width]
    vx, vy = dx - sx, dy - sy
    length = np.hypot(vx, vy)
    if length < 1e-9 or radius <= 0:
        return x, y
    vx, vy = vx / length, vy / length
    hit = ((x >= min(sx, dx) - radius) & (x <= max(sx, dx) + radius)
           & (y >= min(sy, dy) - radius) & (y <= max(sy, dy) + radius))
    if not hit.any():
        return x, y
    xx = x[hit].astype(float) - sx
    yy = y[hit].astype(float) - sy
    along = vx * xx + vy * yy
    cross = vy * xx - vx * yy
    falloff = np.maximum(0., 1. - (cross / radius) ** 2)
    margin = radius * np.sqrt(falloff)
    span = length + 2 * margin
    use = (margin > 1e-7) & (along > -margin) & (along < length + margin)
    if not use.any():
        return x, y

    def axis(u):
        return 1 / (1 - u) - 1 / u

    a, b = margin[use], span[use]
    center = a / b
    moved = (a + length * falloff[use] ** 2) / b
    shift = axis(moved) - axis(center)
    z = axis((along[use] + a) / b) + (-shift if inverse else shift)
    delta = np.zeros_like(along)
    delta[use] = b * (.5 + .5 * z / (2 + np.hypot(2, z))) - a - along[use]
    out_x, out_y = x.copy(), y.copy()
    out_x[hit] += vx * delta
    out_y[hit] += vy * delta
    return out_x, out_y


def deform_paths(paths, point, width, height):
    """Subdivide only segments that intersect the brush before bending them."""
    src = np.asarray(point[:2]) * (width, height)
    dst = np.asarray(point[2:4]) * (width, height)
    if np.array_equal(src, dst):
        return paths
    rad = point[4] * width
    low, high = np.minimum(src, dst) - rad, np.maximum(src, dst) + rad
    result = []
    for path in paths:
        p = path[:, ::-1]
        a, b = p[:-1], p[1:]
        near = ((np.maximum(a, b) >= low) & (np.minimum(a, b) <= high)).all(axis=1)
        if not near.any():
            result.append(path)
            continue
        n = np.ones(len(a), int)
        n[near] = np.maximum(1, np.ceil(np.linalg.norm(b[near]-a[near], axis=1) / (rad/8))).astype(int)
        seg = np.repeat(np.arange(len(a)), n)
        fraction = (np.arange(len(seg)) - np.repeat(np.cumsum(n)-n, n)) / n[seg]
        coords = np.vstack((a[seg] + (b-a)[seg] * fraction[:, None], p[-1]))
        x, y = transport(coords[:, 0], coords[:, 1], point, width, height)
        result.append(np.column_stack((y, x)).astype(np.float32))
    return result


def snap_path(paths, x, y, tolerance):
    """Project a pointer onto the nearest displayed segment, in corner pixels."""
    cursor = np.array([y, x])
    best = tolerance ** 2
    snapped = None
    for path in paths:
        if np.any(path.min(axis=0) > cursor+tolerance) or np.any(path.max(axis=0) < cursor-tolerance):
            continue
        start = path[:-1]
        v = path[1:] - start
        t = np.sum((cursor-start)*v, axis=1) / np.maximum(np.sum(v*v, axis=1), 1e-20)
        q = start + np.clip(t, 0, 1)[:, None] * v
        d2 = np.sum((cursor-q)**2, axis=1)
        i = d2.argmin()
        if d2[i] <= best:
            best, snapped = d2[i], (float(q[i, 1]), float(q[i, 0]))
    return snapped
