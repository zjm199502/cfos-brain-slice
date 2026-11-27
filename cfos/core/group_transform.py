"""Invertible similarity edits of atlas coordinates within an oriented selection rectangle."""
from functools import lru_cache

import numpy as np


def similarity(xy, center, offset, angle, scale):
    c, s = np.cos(angle), np.sin(angle)
    matrix = scale * np.array([[c, -s], [s, c]])
    return (np.asarray(xy) - center) @ matrix.T + center + offset


def make_edit(box, shift, angle, scale, width, height, whole, existing):
    """The first eight slots follow the existing ordered, grouped manual-edit history."""
    dims = np.array([width, height])
    center = box.mean(axis=0)
    group = 1 + max((p[7] for p in existing if len(p) > 7), default=0)
    return [*(center/dims).tolist(), *((center+shift)/dims).tolist(), 1., 3, 0, group,
            *(box/dims).ravel().tolist(), float(angle), float(scale), bool(whole)]


@lru_cache(maxsize=128)
def _program(edit, width, height):
    dims = np.array([width, height])
    box = np.array(edit[8:16]).reshape(4, 2) * dims
    center = box.mean(axis=0)
    shift = (np.array(edit[2:4])-edit[:2]) * dims
    angle, scale, whole = edit[16:]
    stages = []
    # Each one-axis operation has an exact inverse on every scanline. The
    # surrounding band reconnects interfaces that cross the selection border.
    margin = max(1., min(np.linalg.norm(np.roll(box,-1,axis=0)-box,axis=1)) / 4)
    def add(axis, gain, shear, move):
        if (gain, shear, move) == (1., 0., 0.):
            return
        bias = center[axis]*(1-gain) - center[1-axis]*shear + move
        after = box.copy()
        after[:,axis] = gain*box[:,axis] + shear*box[:,1-axis] + bias
        stages.append((axis, box.copy(), gain, shear, bias, after.copy(), margin))
        box[:] = after

    if not whole:
        turns = max(1, int(np.ceil(abs(angle)/(np.pi/2))))
        step = angle/turns
        for _ in range(turns):
            add(0, 1., -np.tan(step/2), 0.)
            add(1, 1., np.sin(step), 0.)
            add(0, 1., -np.tan(step/2), 0.)
        for axis in range(2):
            add(axis, scale, 0., 0.)
        for axis in range(2):
            add(axis, 1., 0., shift[axis])
    return center, shift, angle, scale, whole, stages


def _scanline(a, b, stage, inverse):
    axis, box, gain, shear, bias, after, margin = stage
    across = 1-axis
    bmin, bmax = box[:,across].min(), box[:,across].max()
    left = min(box[:,axis].min(),after[:,axis].min())-margin
    right = max(box[:,axis].max(),after[:,axis].max())+margin
    hit = (b>bmin-margin) & (b<bmax+margin) & (a>left) & (a<right)
    if not hit.any():
        return a
    u, v = a[hit], b[hit]
    vc = np.clip(v, bmin, bmax)
    lo, hi = np.full_like(vc,np.inf), np.full_like(vc,-np.inf)
    for p,q in zip(box,np.roll(box,-1,axis=0)):
        span = q[across]-p[across]
        if abs(span) < 1e-12:
            continue
        t = (vc-p[across])/span
        on = (t >= -1e-9) & (t <= 1+1e-9)
        intercept = p[axis]+t*(q[axis]-p[axis])
        lo = np.where(on,np.minimum(lo,intercept),lo)
        hi = np.where(on,np.maximum(hi,intercept),hi)
    blend = np.maximum(0., 1.-abs(v-vc)/margin)
    slope = 1.+(gain-1.)*blend
    offset = (shear*vc+bias)*blend
    new_lo, new_hi = slope*lo+offset, slope*hi+offset
    start, end = np.minimum(lo,new_lo)-margin, np.maximum(hi,new_hi)+margin
    if inverse:
        lo, new_lo, hi, new_hi = new_lo, lo, new_hi, hi
        offset, slope = -offset/slope, 1./slope
    inside = (u>=start) & (u<=end)
    moved = np.where(u<lo, start+(u-start)*(new_lo-start)/(lo-start),
                     np.where(u>hi, end+(u-end)*(end-new_hi)/(end-hi), slope*u+offset))
    out = a.copy()
    out[hit] = np.where(inside,moved,u)
    return out


def transport_group(x, y, edit, width, height, inverse=False):
    center, shift, angle, scale, whole, stages = _program(tuple(edit), width, height)
    if whole:
        xy = np.stack(np.broadcast_arrays(x,y),axis=-1)
        if inverse:
            out = similarity(xy,center+shift,-shift,-angle,1/scale)
        else:
            out = similarity(xy,center,shift,angle,scale)
        return out[...,0], out[...,1]
    coords = list(np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float)))
    for stage in reversed(stages) if inverse else stages:
        axis = stage[0]
        coords[axis] = _scanline(coords[axis],coords[1-axis],stage,inverse)
    return tuple(coords)


def group_paths(paths, edit, width, height):
    if not paths:
        return []
    # Atlas curves are simplified into long straight segments. Add samples
    # before a local edit so the middle of a segment can follow the mouse.
    if not edit[-1]:
        box = np.array(edit[8:16]).reshape(4,2)*[width,height]
        stride = max(.5, min(np.linalg.norm(np.roll(box,-1,axis=0)-box,axis=1))/12)
        prepared = []
        for path in paths:
            diff = np.diff(path,axis=0)
            n = np.maximum(1,np.ceil(np.linalg.norm(diff,axis=1)/stride).astype(int))
            if n.max(initial=1) > 1:
                index = np.repeat(np.arange(len(n)), n)
                t = (np.arange(len(index)) - np.repeat(n.cumsum()-n,n))/n[index]
                path = np.vstack((path[index]+t[:,None]*diff[index],path[-1]))
            prepared.append(path)
        paths = prepared
    lengths = np.fromiter(map(len,paths),int)
    y,x = np.concatenate(paths).T
    x,y = transport_group(x,y,edit,width,height)
    return np.split(np.column_stack((y,x)).astype(np.float32),lengths.cumsum()[:-1])
