"""同一只动物的连续切片：按前后位置（Bregma）排序，并联合校正配准。

配准支持逐片微调切面角度；而同一个脑子的切片是用同一个角度切的。
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
        # 自动校正的前后平移（ap_auto）仍然叠加在平面上，这里把它算进当前位置，校正后的最终位置才是等间距的
        pos = np.array([float(np.dot(p.center(), n_med)) + r.ap_auto for p, r in zip(new, regs)])
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


AUTO_MIN_SECTIONS = 3    # 同一只动物至少有这么多张脑片才自动统一角度（2 张时中位数没有意义）


def unify_new(new_regs: list[Registration], context_regs: list[Registration] = ()) -> bool:
    """自动配准时用（v0.8.5）：把同一只动物新配准的脑片的切面角度统一成中位数（前后位置不变）。
    context_regs 是这只动物以前已经配准好的脑片：只参与求中位数，本身不改。
    新旧加起来不足 AUTO_MIN_SECTIONS 张时什么也不做，返回 False。"""
    new_regs = [r for r in new_regs if r.done]
    ctx = [r for r in context_regs if r.done]
    if not new_regs or len(new_regs) + len(ctx) < AUTO_MIN_SECTIONS:
        return False
    base = [Plane.from_list(r.plane_raw or r.plane) for r in new_regs]
    normals = [_normal(p) for p in base] + [_normal(Plane.from_list(r.plane_raw or r.plane)) for r in ctx]
    n_med = np.median(np.array(normals), axis=0); n_med /= np.linalg.norm(n_med)
    for r, p in zip(new_regs, base):
        if not r.plane_raw:
            r.plane_raw = list(r.plane)
        r.plane = _rotate_plane(p, _normal(p), n_med).to_list()
    return True


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


# ------------------------------------------------------------------ 按切片顺序联合定位前后位置
VOXEL_UM = 25.0          # CCFv3 与 STAM 的编辑坐标都是 25 µm 一格
AP_TOL_MIN_UM = 100.0    # 偏离顺序预测超过这么多才算异常（giRAff 单张的前后误差常在 ±100 µm 内）
AP_TOL_MAX_UM = 300.0    # 离散度很大时容差也不超过这么多，免得掩盖 400 µm 这样的跳动
AP_SEARCH_UM = 400.0     # 纹理微调前后位置的默认范围（与 deep_refine.search_plane 一致）


def fit_order(positions, spacing_vox: float = 0.0) -> dict | None:
    """按切片顺序排好的前后位置（体素）→ 稳健直线拟合 pos_i ≈ b + s·i，并找出偏离直线的切片。

    间距已知时 s = ±间距（方向取所有两两斜率的中位数的符号）；未知时 s 用 Theil–Sen（两两斜率的中位数）。
    异常：残差超过 tol = clip(3 × 1.4826 × MAD, 100 µm, max(100 µm, min(300 µm, 半个间距)))。异常的切片不参与第二次拟合。
    间距未知时只有 3 张无法判断谁错（任意两张都能连成直线），至少要 4 张；已知间距时 3 张即可。
    consistent=False 表示多数切片都对不上（切片顺序可能不是切的顺序），这时不应据此挪动切片。"""
    x = np.asarray(positions, float)
    n = len(x)
    if n < (3 if spacing_vox > 0 else 4):
        return None
    idx = np.arange(n, dtype=float)

    def line(keep):
        i, p = idx[keep], x[keep]
        a, b_ = np.triu_indices(len(p), 1)
        slopes = (p[b_] - p[a]) / (i[b_] - i[a])
        ts = float(np.median(slopes))
        if spacing_vox > 0:
            sign = np.sign(ts) or (1.0 if np.sum(slopes > 0) >= np.sum(slopes < 0) else -1.0)
            s = sign * spacing_vox
        else:
            s = ts
        return s, float(np.median(p - s * i))

    s, b = line(np.ones(n, bool))
    res = x - (b + s * idx)
    sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    # 偏离超过半个间距 = 已经落到相邻切片的位置上，无论离散度多大都算异常
    cap = max(AP_TOL_MIN_UM / VOXEL_UM, min(AP_TOL_MAX_UM / VOXEL_UM, 0.5 * abs(s)))
    tol = float(np.clip(3 * sigma, AP_TOL_MIN_UM / VOXEL_UM, cap))
    out = np.abs(res) > tol
    consistent = int(out.sum()) <= (n - 1) // 2 and (n - out.sum()) >= (2 if spacing_vox > 0 else 3)
    if consistent and out.any():
        s, b = line(~out)
        res = x - (b + s * idx)
        out = np.abs(res) > tol
        consistent = int(out.sum()) <= (n - 1) // 2
    pred = b + s * idx
    return {"pred": pred, "resid": res, "slope": s, "intercept": b, "tol": tol,
            "outliers": out, "consistent": bool(consistent)}


def _series_key(project, image, section) -> str:
    return (project.samples.get(f"{image.key}/{section.name}") or {}).get("animal") or image.key


def _position(reg: Registration, n_med) -> float:
    return float(np.dot(adjusted_plane(reg).center(), n_med))


def _shift_base(reg: Registration, n_med, d: float):
    if not reg.plane_raw:
        reg.plane_raw = list(reg.plane)
    p = Plane.from_list(reg.plane)
    reg.plane = Plane(p.o + n_med * d, p.u, p.v).to_list()


def position_series(project, new: dict, spacing_um: float | None = None) -> dict:
    """自动配准的第二步（giRAff 逐张定位之后、非线性贴合之前）：同一动物的脑片按切片顺序联合定位前后位置。

    new: {(image.source_id, section.name): Registration}，本次新定位的脑片（原地修改）。
    同一动物以前已配准的脑片只作参照，不改动。切片顺序 = 项目里的图片顺序，图片内按脑片顺序（拆分时的编号）；
    方向（从前到后还是从后到前）由数据决定。
    - 前后位置偏离顺序预测超过容差的新脑片：基础平面挪到预测位置（可在“排序和联合校正”里撤销），并加 ⚠ 提示；
    - 返回每张新脑片的纹理微调范围 {key: (lo, hi)}（体素，相对当前 ap_auto），保证微调后不会再跳出容差；
    - 顺序与定位整体对不上（多数切片偏离）时不挪动，只给偏离的脑片加 ⚠。
    返回 {"bounds": {...}, "notes": {key: [提示, ...]}, "groups": [每组的报告]}。"""
    spacing_um = project.section_spacing_um if spacing_um is None else spacing_um
    step = spacing_um / VOXEL_UM if spacing_um and spacing_um > 0 else 0.0
    groups: dict[str, list] = {}
    for im in project.images:
        for s in im.sections:
            key = (im.source_id, s.name)
            if key in new:
                reg = new[key]
            elif s.enabled:
                reg = Registration.from_dict(s.registration)
                if reg.fit_atlas != project.registration_atlas:
                    continue
            else:
                continue
            if reg.done:
                groups.setdefault(_series_key(project, im, s), []).append((key, reg, key in new, s.name))
    bounds, notes, reports = {}, {}, []
    lim = AP_SEARCH_UM / VOXEL_UM
    for animal, items in groups.items():
        if not any(is_new for _, _, is_new, _ in items):
            continue
        n_med = np.median([_normal(adjusted_plane(r)) for _, r, _, _ in items], axis=0)
        n_med /= np.linalg.norm(n_med)
        pos = [_position(r, n_med) for _, r, _, _ in items]
        fit = fit_order(pos, step)
        if fit is None:
            # 张数太少、判断不了谁错：不挪动，只把纹理微调限制在半个间距内（至少 100 µm），切片顺序不会被微调打乱
            if len(items) >= 2:
                gap = step or float(np.median(np.abs(np.diff(pos))))
                half = float(np.clip(0.5 * gap, AP_TOL_MIN_UM / VOXEL_UM, AP_TOL_MAX_UM / VOXEL_UM))
                for key, reg, is_new, _ in items:
                    if is_new:
                        bounds[key] = (-half, half)
                reports.append({"animal": animal, "consistent": True, "spacing_um": round(gap * VOXEL_UM, 1),
                                "tol_um": round(half * VOXEL_UM), "moved": [], "flagged": [], "too_few": True})
            continue
        report = {"animal": animal, "consistent": fit["consistent"], "spacing_um": round(abs(fit["slope"]) * VOXEL_UM, 1),
                  "tol_um": round(fit["tol"] * VOXEL_UM), "moved": [], "flagged": []}
        for k, (key, reg, is_new, name) in enumerate(items):
            if not is_new:
                continue
            got, want = pos[k], float(fit["pred"][k])
            n_reg = _normal(adjusted_plane(reg))
            cos = max(float(np.dot(n_reg, n_med)), 0.5)
            if fit["outliers"][k]:
                off = (got - want) * VOXEL_UM
                if fit["consistent"]:
                    _shift_base(reg, n_med, want - got)
                    reg.status = f"{reg.status} · 按切片顺序校正前后位置"
                    notes.setdefault(key, []).append(
                        f"前后位置与切片顺序不符：giRAff 给出的位置偏离相邻切片推算的位置 {off:+.0f} µm，"
                        f"已移到推算位置，请检查")
                    report["moved"].append({"section": name, "offset_um": round(off)})
                    got = want
                else:
                    notes.setdefault(key, []).append(
                        f"前后位置与切片顺序不符（偏离 {off:+.0f} µm），同一动物多数切片也对不上，未自动挪动，请检查切片顺序")
                    report["flagged"].append({"section": name, "offset_um": round(off)})
                    continue
            lo = (want - fit["tol"] - got) / cos
            hi = (want + fit["tol"] - got) / cos
            bounds[key] = (min(max(lo, -lim), 0.0), max(min(hi, lim), 0.0))
        reports.append(report)
    return {"bounds": bounds, "notes": notes, "groups": reports}


AP_PRIOR_UM = 2.0        # 每偏离 giRAff（及上一步顺序校正）的位置 1 格，细胞层评分加这么多 µm（分数相近时少挪动）
AP_MIN_GAP_UM = 50.0     # 间距未知时相邻切片至少相隔这么多（约一张切片的厚度），顺序不能颠倒


def choose_ap(project, new: dict, profiles: dict, spacing_um: float | None = None) -> dict:
    """按细胞层评分（deep_refine.ap_profile）选每张新脑片的前后位置，同一动物内保持切片顺序。

    profiles: {key: {"offsets": [...], "scores": [...]}}（体素，相对当前 ap_auto；µm）。没有评分的新脑片和
    以前配准好的脑片位置固定，只作为顺序的约束。同一动物里在两个方向（从前到后、从后到前）各做一次动态规划，
    取总分低的：相邻切片间距不小于 AP_MIN_GAP_UM；已知间距时限制在间距 ± 100 µm 内。
    返回 {"offsets": {key: 体素}, "notes": {key: [提示]}, "groups": [...]}；不修改 new。"""
    spacing_um = project.section_spacing_um if spacing_um is None else spacing_um
    step = spacing_um / VOXEL_UM if spacing_um and spacing_um > 0 else 0.0
    groups: dict[str, list] = {}
    for im in project.images:
        for s in im.sections:
            key = (im.source_id, s.name)
            if key in new:
                reg = new[key]
            elif s.enabled:
                reg = Registration.from_dict(s.registration)
                if reg.fit_atlas != project.registration_atlas:
                    continue
            else:
                continue
            if reg.done:
                groups.setdefault(_series_key(project, im, s), []).append((key, reg))
    offsets, notes, reports = {}, {}, []
    for animal, items in groups.items():
        if not any(key in profiles for key, _ in items):
            continue
        n_med = np.median([_normal(adjusted_plane(r)) for _, r in items], axis=0)
        n_med /= np.linalg.norm(n_med)
        cand = []   # 每张切片：[(偏移体素, 位置, 代价)]
        for key, reg in items:
            pos = _position(reg, n_med)
            prof = profiles.get(key)
            if prof is None:
                cand.append([(0.0, pos, 0.0)])
                continue
            o = np.asarray(prof["offsets"], float)
            sc = np.asarray(prof["scores"], float)
            ok = np.isfinite(sc)
            cos = max(float(np.dot(_normal(adjusted_plane(reg)), n_med)), 0.5)
            grid = np.arange(np.ceil(o[ok].min()), np.floor(o[ok].max()) + 1)
            cost = np.interp(grid, o[ok], sc[ok]) + AP_PRIOR_UM * np.abs(grid)
            cand.append([(float(g), pos + g * cos, float(c)) for g, c in zip(grid, cost)])
        lo_gap = max(AP_MIN_GAP_UM / VOXEL_UM, step - AP_TOL_MIN_UM / VOXEL_UM) if step else AP_MIN_GAP_UM / VOXEL_UM
        hi_gap = step + AP_TOL_MIN_UM / VOXEL_UM if step else np.inf
        best = None
        for direction in (1.0, -1.0):
            total = [c for _, _, c in cand[0]]
            back = []
            for k in range(1, len(cand)):
                nxt, arg = [], []
                for _, p, c in cand[k]:
                    options = [(total[j] + c, j) for j, (_, q, _) in enumerate(cand[k - 1])
                               if np.isfinite(total[j]) and lo_gap - 1e-6 <= direction * (p - q) <= hi_gap + 1e-6]
                    v, j = min(options) if options else (np.inf, -1)
                    nxt.append(v)
                    arg.append(j)
                total, back = nxt, back + [arg]
            j = int(np.argmin(total))
            if np.isfinite(total[j]) and (best is None or total[j] < best[0]):
                path = [j]
                for arg in reversed(back):
                    path.append(arg[path[-1]])
                best = (total[j], direction, path[::-1])
        report = {"animal": animal, "moved": []}
        if best is None:
            # 顺序约束无解（以前的切片或间距与评分矛盾）：各自取最低分，并提示
            path = [int(np.argmin([c for _, _, c in cs])) for cs in cand]
            report["unordered"] = True
        else:
            path = best[2]
            report["direction"] = "前后位置随切片顺序增大" if best[1] > 0 else "前后位置随切片顺序减小"
        for (key, _), cs, j in zip(items, cand, path):
            if key not in profiles:
                continue
            off = cs[j][0]
            offsets[key] = off
            if off:
                report["moved"].append({"section": key[1], "offset_um": round(off * VOXEL_UM)})
            if best is None:
                notes.setdefault(key, []).append("按细胞层选前后位置时与切片顺序对不上，已各自取最佳位置，请检查切片顺序")
            elif abs(off) >= max(abs(o) for o in profiles[key]["offsets"]):
                notes.setdefault(key, []).append(
                    f"按海马细胞层把前后位置挪了 {off * VOXEL_UM:+.0f} µm，到了搜索范围的边缘，请检查")
        reports.append(report)
    return {"offsets": offsets, "notes": notes, "groups": reports}
