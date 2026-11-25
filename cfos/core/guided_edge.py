"""Follow a fluorescence edge near an ordered, hand-drawn guide."""
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from .edge_drag import landmark_targets, protected_radius, transport


DOT_UM = 10.


def _arclength(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def fluorescence_edge(slide, clicks, channel, bbox):
    """Return a level-zero xy curve. Read small windows even on non-pyramidal slides."""
    xy = np.asarray(clicks, dtype=float)
    arc = _arclength(xy)
    um = slide.pixel_size_um
    if len(xy) < 2 or np.any(np.diff(arc) <= 0) or arc[-1] * um < DOT_UM:
        raise ValueError("请沿同一条边界依次点至少两个相隔 10 µm 的位置。")
    stations = np.linspace(0, arc[-1], max(3, int(np.ceil(arc[-1]*um/5)) + 1))
    guide = PchipInterpolator(arc, xy, axis=0)(stations)
    tangent = np.gradient(guide, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    # The centre is the hand guide. Extra margin supports the normal derivative.
    offsets = np.arange(-36., 37., 3.)
    coords = guide[:, None] + normal[:, None] * (offsets[None, :, None] / um)
    lv = slide.level_for_pixel_size(3.)
    ds = slide.downsample(lv)
    H, W = slide.level_shapes[lv]
    left, top, width, height = bbox
    inside = ((coords[..., 0] >= left) & (coords[..., 0] < left+width) &
              (coords[..., 1] >= top) & (coords[..., 1] < top+height) &
              (coords[..., 0] >= 0) & (coords[..., 0] < W*ds) &
              (coords[..., 1] >= 0) & (coords[..., 1] < H*ds))
    sigma = max(.6, 6./(um*ds))
    pad = int(np.ceil(4*sigma)) + 2
    ribbon = np.empty(coords.shape[:2], np.float32)
    for i in range(0, len(guide), 16):
        local = coords[i:i+16] / ds
        x0, y0 = np.maximum(0, np.floor(local.min(axis=(0, 1))).astype(int)-pad)
        x1, y1 = np.minimum([W, H], np.ceil(local.max(axis=(0, 1))).astype(int)+pad+1)
        # slide.read floors the origin at the selected pyramid level.
        bx, by = int(np.ceil(x0*ds)), int(np.ceil(y0*ds))
        roi = (bx, by, int(np.ceil(x1*ds))-bx, int(np.ceil(y1*ds))-by)
        img = slide.read(lv, roi, [channel])[0].astype(np.float32)
        smooth = ndi.gaussian_filter(img, sigma)
        ribbon[i:i+16] = ndi.map_coordinates(smooth, [local[..., 1]-y0, local[..., 0]-x0],
                                              order=1, mode="nearest")
    grad = np.abs(np.gradient(ribbon, offsets, axis=1))
    valid = inside[:, 1:-1] & inside[:, :-2] & inside[:, 2:]
    # Up to 24 µm either side. Penalize displacement and sudden changes in offset.
    idx = np.flatnonzero(np.abs(offsets) <= 24.)
    strength = grad[:, idx]
    allowed = valid[:, idx-1]
    values = strength[allowed]
    if not values.size or not allowed.any(axis=1).all():
        raise ValueError("点列超出脑片，请把点移到脑片范围内。")
    scale = np.percentile(values, 90)
    if scale <= 1e-6:
        raise ValueError("所选通道在点列附近缺少可辨边缘，请换通道或重新布点。")
    offsets = offsets[idx]
    score = np.minimum(strength/scale, 2.) - .45*(offsets/24.)**2
    score[~allowed] = -np.inf
    transitions = .08*((offsets[:, None]-offsets[None, :])/3.)**2
    total = score[0]
    prev = np.empty(score.shape, int)
    for j in range(1, len(score)):
        options = total[:, None] - transitions
        prev[j] = np.argmax(options, axis=0)
        total = options[prev[j], np.arange(len(offsets))] + score[j]
    track = np.empty(len(guide), int)
    track[-1] = np.argmax(total)
    for j in range(len(guide)-1, 0, -1):
        track[j-1] = prev[j, track[j]]
    # Smooth small fluctuations from individual bright somata along the strip.
    shift = ndi.gaussian_filter1d(offsets[track], 1.6, mode="nearest")
    smooth = guide + normal * (shift[:, None] / um)
    bounds = np.array([min(left+width, W*ds), min(top+height, H*ds)])
    out = ((smooth < [max(left, 0), max(top, 0)]) | (smooth >= bounds)).any(axis=1)
    smooth[out] = guide[out] + normal[out] * (offsets[track[out], None] / um)
    return smooth


def _match_outline(paths, targets, tol, pixel_um):
    """Choose a single atlas interface supported along the whole guide."""
    best, source = np.inf, None
    for path in paths:
        xy = path[:, ::-1]
        if np.any(xy.max(axis=0)+tol < targets.max(axis=0)) or np.any(xy.min(axis=0)-tol > targets.min(axis=0)):
            continue
        arc = _arclength(xy)
        if arc[-1] == 0:
            continue
        step = max(1./pixel_um, .1)
        positions = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1]/step))+1))
        curve = np.column_stack([np.interp(positions, arc, col) for col in xy.T])
        distances, indices = cKDTree(curve).query(targets)
        if distances.max() > tol:
            continue
        along = np.diff(positions[indices])
        if np.allclose(xy[0], xy[-1]):
            along = (along + arc[-1]/2) % arc[-1] - arc[-1]/2
        hops = np.linalg.norm(np.diff(targets, axis=0), axis=1)
        if np.any(np.abs(along) > 3*hops + 10./pixel_um) or abs(along.sum()) < .35*hops.sum():
            continue
        cost = distances.mean() + .4*distances.max() + np.mean(np.abs(np.sign(along.sum())*along-hops))
        if cost < best:
            best, source = cost, curve[indices]
    if source is None:
        raise ValueError("附近没有贯穿整段点列的图谱边界。请在脑区交汇处分段点选，或调整影响范围。")
    return source


def contour_edits(paths, curve, W, H, pixel_um, radius_um, rotated, existing):
    """Fit a matched interface using the same invertible edits used by label lookup and export."""
    arc = _arclength(curve)
    count = max(2, int(np.ceil(arc[-1]*pixel_um/26.))+1)
    samples = np.linspace(0, arc[-1], count)
    targets = np.column_stack([np.interp(samples, arc, col) for col in curve.T])
    shape = np.array([W, H])
    sources = _match_outline(paths, targets, min(200., radius_um)/pixel_um, pixel_um)
    if rotated:
        sources, targets = shape-sources, shape-targets
    group = 1 + max((p[7] for p in existing if len(p) > 7), default=0)
    edits = []
    anchors = landmark_targets(existing, W, H)
    # Displace broad stretches first, then correct the gaps with short residual edits.
    broad = list(range(0, count, 3))
    if broad[-1] != count-1:
        broad.append(count-1)
    for i in dict.fromkeys([*broad, *range(count)]):
        target = targets[i]
        start = sources[i]
        dist = np.linalg.norm(start-target)
        radius = min(radius_um/pixel_um, max(85./pixel_um, dist*2))
        radius = protected_radius(anchors, start, target, radius)
        if radius*pixel_um < 4.:
            if dist*pixel_um > 2.:
                raise ValueError("这段边缘与已有修正点冲突。可缩短点列或撤销上一段。")
            continue
        point = [*(start/shape).tolist(), *(target/shape).tolist(), radius/W, 2, 1, group]
        edits.append(point)
        anchors = np.vstack((anchors, target))
        x, y = transport(sources[:, 0], sources[:, 1], point, W, H)
        sources = np.column_stack((x, y))
    return edits
