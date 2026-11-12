"""同一只动物的连续切片：按前后位置（Bregma）排序，并联合校正配准。

DeepSlice 对每张切片单独估计切面角度，误差在几度左右；而同一个脑子的切片是用同一个角度切的。
所以把同一动物所有切片的角度取中位数，再把每张切片转到这个角度（保持各自的前后位置不变），
可以去掉单张切片的角度误差。若知道切片间距（例如每隔 4 张取 1 张、40 µm 厚 → 160 µm），
还可以让前后位置按等间距排列，进一步去掉单张切片的前后位置误差。
"""
from __future__ import annotations

import numpy as np

from .atlas import Plane
from .registration import Registration, _rot, adjusted_plane


def _normal(pl: Plane) -> np.ndarray:
    n = np.cross(pl.u, pl.v)
    n = n / (np.linalg.norm(n) + 1e-12)
    return -n if n[1] < 0 else n


def angles(reg: Registration) -> tuple[float, float]:
    """切面相对标准冠状面的倾斜（度）：(左右方向, 背腹方向)。"""
    n = _normal(adjusted_plane(reg))
    return float(np.degrees(np.arctan2(n[0], n[1]))), float(np.degrees(np.arctan2(n[2], n[1])))


def _rotate_plane(pl: Plane, n_from, n_to) -> Plane:
    axis = np.cross(n_from, n_to)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return pl
    deg = np.degrees(np.arctan2(s, float(np.dot(n_from, n_to))))
    R = _rot(axis, deg)
    c = pl.center()
    u, v = R @ pl.u, R @ pl.v
    return Plane(c - 0.5 * u - 0.5 * v, u, v)


def harmonize(regs: list[Registration], spacing_um: float = 0.0, fix_angle: bool = True) -> list[dict]:
    """原地修改 regs（只改基础平面 plane，手动调整的滑块数值保留）。返回每张切片校正前后的对照。"""
    regs = [r for r in regs if r.done]
    if len(regs) < 2:
        raise ValueError("同一只动物至少需要 2 张已配准的切片才能联合校正。")
    for r in regs:
        if not r.plane_raw:
            r.plane_raw = list(r.plane)
    before = [(adjusted_plane(r).ap_mm(), *angles(r)) for r in regs]
    base = [Plane.from_list(r.plane_raw) for r in regs]
    normals = np.array([_normal(p) for p in base])
    n_med = np.median(normals, axis=0); n_med /= np.linalg.norm(n_med)
    new = [_rotate_plane(p, _normal(p), n_med) if fix_angle else Plane(p.o.copy(), p.u.copy(), p.v.copy())
           for p in base]
    if spacing_um and spacing_um > 0:
        step = spacing_um / 25.0  # 体素
        pos = np.array([float(np.dot(p.center(), n_med)) for p in new])
        order = np.argsort(pos)
        idx = np.empty(len(pos)); idx[order] = np.arange(len(pos))
        b = np.median(pos - idx * step)
        for k, p in enumerate(new):
            d = (b + idx[k] * step) - pos[k]
            new[k] = Plane(p.o + n_med * d, p.u, p.v)
    for r, p in zip(regs, new):
        r.plane = p.to_list()
        r.status = "联合校正" + ("（角度" + ("+间距" if spacing_um else "") + "）")
    after = [(adjusted_plane(r).ap_mm(), *angles(r)) for r in regs]
    return [{"校正前 Bregma (mm)": round(b_[0], 2), "校正后 Bregma (mm)": round(a_[0], 2),
             "校正前 左右倾斜 (°)": round(b_[1], 1), "校正后 左右倾斜 (°)": round(a_[1], 1),
             "校正前 背腹倾斜 (°)": round(b_[2], 1), "校正后 背腹倾斜 (°)": round(a_[2], 1)}
            for b_, a_ in zip(before, after)]


def undo(reg: Registration) -> bool:
    if reg.plane_raw:
        reg.plane = list(reg.plane_raw); reg.plane_raw = []
        reg.status = "自动配准（已撤销联合校正）"
        return True
    return False


def animal_sections(project) -> dict[str, list[tuple]]:
    """{动物编号: [(image, section), ...]}，按前后位置从前到后排序。没填动物编号的按图片分组。"""
    out: dict[str, list] = {}
    for im in project.images:
        for s in im.sections:
            if not s.enabled:
                continue
            a = (project.samples.get(f"{im.key}/{s.name}") or {}).get("animal") or im.key
            out.setdefault(a, []).append((im, s))
    for a, lst in out.items():
        def ap(item):
            r = Registration.from_dict(item[1].registration)
            return -adjusted_plane(r).ap_mm() if r.done else 1e9
        lst.sort(key=ap)
    return out
