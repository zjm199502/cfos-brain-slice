"""自动配准的完整流程（v0.8.5）：界面上的“自动配准”和批量处理都走这里。

1. DeepSlice 估计每张脑片的切面（同时判断是否倒放）；
2. 同一只动物有 3 张以上脑片时，把切面角度统一成中位数（同一个脑子是按同一个角度切的）；
3. 自动校正前后位置（图谱模板与 DAPI 最像的位置）；
4. 非线性贴合；
5. 找出需要人工重点检查的脑片，原因写进 Registration.warnings，界面上用 ⚠ 标出。
"""
from __future__ import annotations

import traceback

import numpy as np

from . import registration as R
from . import series
from .atlas import sample_plane
from .log import log
from .slide_io import open_slide

DICE_WARN = 0.90          # 最终轮廓重合度低于它时提示
NEIGHBOR_WARN_MM = 0.5    # 与同一张图片里最近的另一张脑片的前后位置相差超过它时提示


def _animal(project, im, s) -> str:
    return (project.samples.get(f"{im.key}/{s.name}") or {}).get("animal") or im.key


def neighbor_warnings(regs: dict[str, R.Registration]) -> dict[str, str]:
    """同一张图片（玻片）上的脑片通常是相邻的。某一张与离它最近的另一张相差很远时，多半是它（或对方）定位错了。"""
    ap = {n: R.adjusted_plane(r).ap_mm() for n, r in regs.items() if r.done}
    out = {}
    if len(ap) < 2:
        return out
    for n, a in ap.items():
        d = min(abs(a - b) for m, b in ap.items() if m != n)
        if d > NEIGHBOR_WARN_MM:
            out[n] = f"与同一张图片里其他脑片的前后位置相差 {d:.1f} mm 以上"
    return out


def run(project, targets, nuc, nonlinear: bool = True, refine: bool = True, unify: bool = True,
        progress=None, cancelled=None, only: dict | None = None, slides: dict | None = None) -> dict:
    """targets：要配准的图片；only={图片 key: {脑片名}} 时只配准这些脑片（批量处理里“已配准的跳过”）。
    slides={图片路径: 已打开的 Slide} 时直接用、不关闭。
    返回 {图片路径: {脑片名: Registration}}，并把非线性位移场存到 results/registration。不改 project。"""
    say = progress or (lambda d, t, txt: None)

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError

    # ---- 1. DeepSlice
    work = []     # (im, s, reg, img, mask)
    n_im = max(len(targets), 1)
    for i, im in enumerate(targets):
        check()
        say(i, n_im * 10, f"正在定位 {im.key} 的脑片…")
        secs = [s for s in im.sections if s.enabled and (only is None or s.name in only.get(im.key, ()))]
        if not secs:
            continue
        sl = (slides or {}).get(im.path)
        own = sl is None
        if own:
            sl = open_slide(im.path)
        try:
            regs, thumbs = R.auto_register(sl, secs, nuc, str(project.results_dir), return_thumbs=True)
        finally:
            if own:
                sl.close()
        for s in secs:
            img, mask = thumbs[s.name]
            work.append((im, s, regs[s.name], img, mask))

    # ---- 2. 同一只动物统一切面角度
    unified = set()
    if unify and work:
        new_keys = {(im.key, s.name) for im, s, *_ in work}
        groups: dict[str, list] = {}
        for im, s, reg, *_ in work:
            groups.setdefault(_animal(project, im, s), []).append((im, s, reg))
        for animal, items in groups.items():
            ctx = [R.Registration.from_dict(s.registration) for im in project.images for s in im.sections
                   if s.enabled and (im.key, s.name) not in new_keys and _animal(project, im, s) == animal]
            if series.unify_new([r for _, _, r in items], ctx):
                unified.update((im.key, s.name) for im, s, _ in items)

    # ---- 3 + 4. 前后位置、非线性贴合
    out: dict[str, dict] = {}
    total = max(len(work), 1)
    for k, (im, s, reg, img, mask) in enumerate(work):
        check()
        mo, io = R.orient(mask, reg.orientation), R.orient(img, reg.orientation)
        notes, warns = [], []
        if (im.key, s.name) in unified:
            notes.append("已统一切面角度")
        if refine:
            say(n_im + int(9 * n_im * k / total), n_im * 10, f"正在校正 {im.key} {s.name} 的前后位置…")
            try:
                res = R.refine_ap(reg, mo, io, cancelled)
                um = res["shift_um"]
                if um:
                    notes.append(f"前后位置自动向{'前' if um > 0 else '后'}校正了 {abs(um):.0f} µm")
                    if abs(um) >= R.AP_BIG_UM:
                        warns.append(f"前后位置自动校正了 {abs(um):.0f} µm，幅度较大"
                                     + ("（已到搜索范围的边上，真实位置可能更远）" if res["edge"] else ""))
            except InterruptedError:
                raise
            except Exception:  # noqa: BLE001  校正失败时保留 DeepSlice 的位置
                reg.ap_auto = 0.0
                log.warning("前后位置自动校正失败 %s %s\n%s", im.key, s.name, traceback.format_exc())
        H, W = mo.shape
        warp = None
        if nonlinear:
            say(n_im + int(9 * n_im * (k + 0.6) / total), n_im * 10, f"正在非线性贴合 {im.key} 的 {s.name}…")
            try:
                w = R.nonlinear_fit(reg, mo, io)
                np.save(R.warp_path(project.results_dir, im.key, s.name), np.stack(w))
                reg.nonlinear = True
                warp = w
                if reg.nl_method == R.CONTOUR:
                    warns.append("内部结构的贴合被放弃，只对齐了外轮廓")
            except Exception:  # noqa: BLE001  非线性贴合失败时保留线性配准
                log.warning("非线性贴合失败 %s %s\n%s", im.key, s.name, traceback.format_exc())
                warns.append("非线性贴合失败，只做了整体对齐")
        d = R.dice(sample_plane(R.adjusted_plane(reg), W, H, warp=warp)["annotation"] > 0, mo)
        if d < DICE_WARN:
            warns.append(f"轮廓重合度偏低（{d:.2f}）")
        reg.status = f"自动配准（轮廓重合度 {d:.2f}" + "".join("；" + n for n in notes) + "）"
        reg.warnings = warns
        out.setdefault(im.path, {})[s.name] = reg

    # ---- 5. 同一张图片里前后位置跳变（把这张图片里以前配准好的脑片也算上）
    for im in targets:
        regs = out.get(im.path)
        if not regs:
            continue
        allr = {s.name: R.Registration.from_dict(s.registration) for s in im.sections if s.enabled}
        allr.update(regs)
        for name, msg in neighbor_warnings(allr).items():
            if name in regs:
                regs[name].warnings = list(regs[name].warnings) + [msg]
    say(n_im * 10, n_im * 10, "配准完成")
    return out
