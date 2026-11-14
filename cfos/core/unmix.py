"""通道串色校正（不需要单染对照的“盲解混”）。

模型（线性串色）：
    看到的通道 i = 真实通道 i + Σ_j a[i, j] × 真实通道 j        （j ≠ i）
写成矩阵就是  观测 = M · 真实，M 的对角线是 1，a[i, j] 是“通道 j 漏进通道 i 的比例”。
知道 M 后，真实 = M⁻¹ · 观测。

系数 a 直接从片子本身估算（只看“源通道亮”的像素，按源通道亮度分段）：

1. 主峰法（默认）
   源通道亮的像素里，大多数并没有目标通道自己的信号（比如 cFos 只在少数神经元里），
   这些像素的目标通道亮度 = a × 源 + 背景，构成散点图里最密的那条“主带”。
   每段取目标通道亮度的主峰位置，拟合直线，斜率就是 a。真共标的细胞在主带上方，
   只要它们是少数，就不会把斜率拉高。

2. 下包络法（偏保守，作为下限参考）
   每段取目标通道的 5% 低分位数拟合斜率。噪声会让它略偏低。

3. 互信息最小化（PICASSO 的思路，Seo et al. Nat Commun 2022）
   试一系列 a，选使“源通道”与“扣除后的目标通道”互信息最小的那个。
   缺点：真实共表达也会产生互信息，可能扣过头，只作为参考值显示。

所有计算只用 numpy，在 CPU 上几秒钟完成（在 0.5–1 µm/像素的随机小块上估算）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class PairEstimate:
    source: int               # 漏出的通道
    sink: int                 # 被串进的通道
    a_envelope: float         # 下包络法估计的系数（偏保守，可看作下限）
    a_ridge: float            # 主峰法估计的系数（默认采用）
    a_mi: float               # 互信息最小化估计的系数
    r2: float                 # 下包络直线拟合优度（越接近 1 越可信）
    n_pixels: int             # 参与估算的像素数
    bins_x: np.ndarray = field(repr=False, default=None)   # 画图用：分段中心
    bins_y: np.ndarray = field(repr=False, default=None)   # 画图用：每段的低分位数
    bins_mode: np.ndarray = field(repr=False, default=None)  # 画图用：每段的主峰位置


@dataclass
class UnmixModel:
    channel_names: list[str]
    matrix: np.ndarray          # M（C×C），观测 = M · 真实
    offsets: np.ndarray         # 每个通道的暗背景（扣除后再解混，解混后加回）
    pairs: list[PairEstimate]
    method: str = "ridge"

    def to_dict(self) -> dict:
        return {
            "channel_names": list(self.channel_names),
            "matrix": np.asarray(self.matrix, float).round(5).tolist(),
            "offsets": np.asarray(self.offsets, float).round(3).tolist(),
            "method": self.method,
        }

    @staticmethod
    def from_dict(d: dict) -> "UnmixModel":
        return UnmixModel(list(d["channel_names"]), np.asarray(d["matrix"], float),
                          np.asarray(d["offsets"], float), [], d.get("method", "ridge"))


# ---------------------------------------------------------------- 估算

def _local_background(ch: np.ndarray, radius_px: float) -> np.ndarray:
    """大尺度背景（最小值滤波 + 平滑），用来去掉组织自发荧光的缓慢变化。"""
    f = max(1, int(radius_px // 6))
    small = ch[::f, ::f]
    r = max(1, int(radius_px / f))
    bg = ndi.minimum_filter(small, size=2 * r + 1)
    bg = ndi.uniform_filter(bg, size=2 * r + 1)
    bg = ndi.zoom(bg, (ch.shape[0] / bg.shape[0], ch.shape[1] / bg.shape[1]), order=1)
    out = np.empty_like(ch)
    h, w = min(bg.shape[0], ch.shape[0]), min(bg.shape[1], ch.shape[1])
    out[:] = bg[-1:, -1:].mean() if bg.size else 0
    out[:h, :w] = bg[:h, :w]
    return out


def _mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    lo_x, hi_x = np.percentile(x, [0.5, 99.5])
    lo_y, hi_y = np.percentile(y, [0.5, 99.5])
    if hi_x <= lo_x or hi_y <= lo_y:
        return 0.0
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[lo_x, hi_x], [lo_y, hi_y]])
    p = h / max(h.sum(), 1)
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def _mode(v: np.ndarray) -> float:
    """一组数的主峰位置（平滑直方图的最高点）。"""
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        return float(np.median(v))
    nb = 60
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    h = np.convolve(h, np.ones(5) / 5, mode="same")
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def estimate_pair(src: np.ndarray, snk: np.ndarray, q: float = 0.05, n_bins: int = 24,
                  a_max: float = 1.0) -> PairEstimate | None:
    """src、snk：已扣背景的一维像素数组（只含组织内的像素）。"""
    if src.size < 2000:
        return None
    # 只看源通道“真正有信号”的像素：亮度位于组织内前 10% 到 99.9%
    lo, hi = np.percentile(src, [90, 99.9])
    if hi <= lo or hi <= 0:
        return None
    sel = (src >= lo) & (src <= hi)
    x, y = src[sel], snk[sel]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    cx, cy, cm, cn = [], [], [], []
    for k in range(n_bins):
        m = (x >= edges[k]) & (x < edges[k + 1]) if k < n_bins - 1 else (x >= edges[k])
        if m.sum() < 30:
            continue
        cx.append(float(np.median(x[m])))
        cy.append(float(np.quantile(y[m], q)))
        cm.append(_mode(y[m]))
        cn.append(int(m.sum()))
    if len(cx) < 4:
        return None
    cx, cy, cm = np.asarray(cx), np.asarray(cy), np.asarray(cm)
    A = np.stack([cx, np.ones_like(cx)], 1)
    (slope_m, _), *_ = np.linalg.lstsq(A, cm, rcond=None)
    a_ridge = float(np.clip(slope_m, 0, a_max))
    (slope, icpt), *_ = np.linalg.lstsq(A, cy, rcond=None)
    pred = A @ np.array([slope, icpt])
    ss = ((cy - cy.mean()) ** 2).sum()
    r2 = float(1 - ((cy - pred) ** 2).sum() / ss) if ss > 0 else 0.0
    a_env = float(np.clip(slope, 0, a_max))

    # 互信息最小化（PICASSO 思路），只在源通道有信号的像素上算
    rng = np.random.default_rng(0)
    idx = rng.choice(x.size, size=min(x.size, 200_000), replace=False)
    xs, ys = x[idx], y[idx]
    grid = np.linspace(0, a_max, 101)
    mis = [_mutual_info(xs, ys - a * xs) for a in grid]
    a_mi = float(grid[int(np.argmin(mis))])
    return PairEstimate(-1, -1, a_env, a_ridge, a_mi, r2, int(x.size), cx, cy, cm)


# 各通道的大致发射波长（nm），用来判断“短波长漏进长波长”的物理方向
_WAVE_WORDS = {"dapi": 460, "hoechst": 460, "blue": 460, "蓝": 460,
               "fitc": 520, "gfp": 510, "green": 520, "绿": 520, "cy3": 570, "tritc": 570,
               "far": 680, "cy5": 670, "远红": 680, "red": 610, "红": 610}


def wavelength(name: str) -> float | None:
    n = name.lower()
    m = re.search(r"(\d{3})", n)
    if m and 380 <= int(m.group(1)) <= 900:
        return float(m.group(1))
    for k in sorted(_WAVE_WORDS, key=len, reverse=True):
        if k in n:
            return float(_WAVE_WORDS[k])
    return None


def default_pairs(channel_names: list[str], adjacent_only: bool = False) -> list[tuple[int, int]]:
    """默认只估算“短波长 → 长波长”的串色（荧光发射光谱的长尾只往红端拖）。

    DAPI 通道不作为被串进的通道：所有细胞都有细胞核，其他通道亮的地方 DAPI 必然也亮，
    无法区分“串色”和“生物学上本来就在一起”。
    认不出波长时按通道顺序当作从短到长。
    """
    C = len(channel_names)
    w = [wavelength(n) for n in channel_names]
    w = [x if x is not None else 400 + 100 * i for i, x in enumerate(w)]
    order = sorted(range(C), key=lambda i: w[i])
    pairs = []
    for a, j in enumerate(order):
        for b, i in enumerate(order):
            if b <= a or (adjacent_only and b != a + 1):
                continue
            if w[i] <= 470:   # DAPI/Hoechst 不作为目标
                continue
            pairs.append((j, i))
    return pairs


def _signal(ch: np.ndarray, pixel_um: float, bg_radius_um: float) -> np.ndarray:
    return ch - _local_background(ch, bg_radius_um / pixel_um)


def estimate(img: np.ndarray, mask: np.ndarray | None, pixel_um: float,
             channel_names: list[str], pairs: list[tuple[int, int]] | None = None,
             method: str = "ridge", min_coef: float = 0.01,
             bg_radius_um: float = 30.0, offsets: np.ndarray | None = None) -> UnmixModel:
    """img：(C, H, W)，或若干小块组成的列表 [(C, h, w), ...]（mask 同样给列表）。

    建议用 0.5–1 µm/像素的分辨率，单个细胞要能分辨出来。
    pairs：要估算的 (源, 目标) 通道对；不给就用 default_pairs()。
    小于 min_coef 的系数当作 0（避免把噪声当串色）。
    """
    tiles = img if isinstance(img, (list, tuple)) else [img]
    masks = mask if isinstance(mask, (list, tuple)) else [mask]
    C = tiles[0].shape[0]
    if offsets is None:
        offsets = np.zeros(C, np.float32)
        for c in range(C):
            outs = [t[c][~m] for t, m in zip(tiles, masks) if m is not None and (~m).sum() > 1000]
            allv = np.concatenate([t[c].ravel() for t in tiles])
            offsets[c] = float(np.median(np.concatenate(outs))) if outs else float(np.percentile(allv, 1))
    vals, sat = [], []
    for t, m in zip(tiles, masks):
        f = t.astype(np.float32)
        if m is None:
            m = np.ones(f.shape[1:], bool)
        top = 255 if t.dtype == np.uint8 else (65535 if t.dtype == np.uint16 else np.inf)
        sat.append((f >= top).any(0)[m])
        vals.append(np.stack([_signal(f[c], pixel_um, bg_radius_um) for c in range(C)])[:, m])
    vals = np.concatenate(vals, 1)
    vals = vals[:, ~np.concatenate(sat)]      # 去掉过曝像素

    if pairs is None:
        pairs = default_pairs(channel_names)
    M = np.eye(C)
    ests = []
    for (j, i) in pairs:
        e = estimate_pair(vals[j], vals[i])
        if e is None:
            continue
        e.source, e.sink = j, i
        ests.append(e)
        a = {"envelope": e.a_envelope, "ridge": e.a_ridge, "mi": e.a_mi}[method]
        if a >= min_coef:
            M[i, j] = a
    return UnmixModel(list(channel_names), M, np.asarray(offsets, np.float32), ests, method)


def sample_tiles(slide, sections=None, pixel_um: float = 0.5, tile_um: float = 250.0,
                 n_tiles: int = 48, seed: int = 0):
    """在组织内随机取若干小块（默认 48 块 250 µm，0.5 µm/像素）。

    返回 (tiles, masks, 实际像素µm, 每通道暗背景)。暗背景取载玻片上组织外区域的中位数。
    """
    from .sections import tissue_mask
    mlv = slide.level_for_pixel_size(8.0)
    mds = slide.downsample(mlv)
    mpix = slide.pixel_size_um * mds
    small = slide.read(mlv)
    tm = tissue_mask(small, mpix)
    off_area = ~ndi.binary_dilation(tm, iterations=max(1, int(300 / mpix)))
    offsets = np.array([float(np.median(c[off_area])) if off_area.sum() > 100 else float(np.percentile(c, 1))
                        for c in small.astype(np.float32)], np.float32)
    tm = ndi.binary_erosion(tm, iterations=max(1, int(150 / mpix)))
    if sections:
        keep = np.zeros_like(tm)
        for sec in sections:
            x, y, w, h = sec.bbox0
            keep[int(y / mds):int((y + h) / mds) + 1, int(x / mds):int((x + w) / mds) + 1] = True
        tm &= keep
    ys, xs = np.nonzero(tm)
    if ys.size == 0:
        raise ValueError("没有找到组织区域")
    rng = np.random.default_rng(seed)
    lv = slide.level_for_pixel_size(pixel_um)
    pix = slide.pixel_size_um * slide.downsample(lv)
    size0 = int(tile_um / slide.pixel_size_um)
    tiles, masks = [], []
    for k in rng.choice(ys.size, size=min(n_tiles, ys.size), replace=False):
        cx, cy = xs[k] * mds, ys[k] * mds
        t = slide.read(lv, (int(cx - size0 / 2), int(cy - size0 / 2), size0, size0))
        tiles.append(t)
        masks.append(np.ones(t.shape[1:], bool))
    return tiles, masks, pix, offsets


# ---------------------------------------------------------------- 应用

def apply(img: np.ndarray, model: UnmixModel, channels: list[int] | None = None) -> np.ndarray:
    """对 (C, H, W) 图做解混，返回与输入相同 dtype。channels：只返回这些通道（仍用全部通道计算）。"""
    C = img.shape[0]
    M = np.asarray(model.matrix, np.float64)
    if np.allclose(M, np.eye(C)):
        return img if channels is None else img[channels]
    Minv = np.linalg.inv(M).astype(np.float32)
    off = np.asarray(model.offsets, np.float32)[:, None, None]
    f = img.astype(np.float32) - off
    np.maximum(f, 0, out=f)
    rows = range(C) if channels is None else channels
    out = np.stack([np.tensordot(Minv[r], f, axes=1) for r in rows])
    out += off[list(rows)]
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    else:
        np.maximum(out, 0, out=out)
    return out.astype(img.dtype)


def describe(model: UnmixModel) -> list[str]:
    """中文说明每对通道的串色比例。"""
    n = model.channel_names
    lines = []
    for e in model.pairs:
        used = model.matrix[e.sink, e.source]
        lines.append(f"{n[e.source]} → {n[e.sink]}：主峰法 {e.a_ridge * 100:.1f}%，"
                     f"下包络法 {e.a_envelope * 100:.1f}%（R²={e.r2:.2f}），互信息法 {e.a_mi * 100:.1f}%，"
                     f"采用 {used * 100:.1f}%")
    return lines


# ---------------------------------------------------------------- 接入识别流程（v0.8.2）
# 项目里的设置（Project.unmix）：
#   enabled  是否在识别前去除串色（新项目默认 True；v0.8.2 以前的旧项目为 False，原来的结果不变）
#   auto     自动估算的串色比例 {"源,目标": 比例}
#   coef     实际使用的比例（滑块微调后的值；没调过时等于 auto）
#   offsets  每个通道的暗背景
#   image    用哪张图片估算的
#   names    估算时各通道的名称（通道名称改了以后用来提示重新估算）

def _key(j: int, i: int) -> str:
    return f"{j},{i}"


def _pairs_from_key(k: str) -> tuple[int, int]:
    a, b = k.split(",")
    return int(a), int(b)


def model_from_settings(d: dict, channel_names: list[str]) -> UnmixModel | None:
    """项目设置 → 解混模型；未开启或还没估算时返回 None。"""
    if not d or not d.get("enabled") or "coef" not in d:
        return None
    C = len(channel_names)
    M = np.eye(C)
    for k, a in d.get("coef", {}).items():
        j, i = _pairs_from_key(k)
        if j < C and i < C and a > 0:
            M[i, j] = float(a)
    off = np.asarray((d.get("offsets") or [0.0] * C)[:C] + [0.0] * max(0, C - len(d.get("offsets") or [])), np.float32)
    return UnmixModel(list(channel_names), M, off, [], d.get("method", "ridge"))


def signature(d: dict) -> dict | None:
    """进入“原始识别结果”签名的部分（改了比例就需要重新识别）。未开启时返回 None，不影响旧项目的签名。"""
    if not d or not d.get("enabled") or "coef" not in d:
        return None
    return {"coef": {k: round(float(v), 4) for k, v in sorted(d["coef"].items())},
            "off": [round(float(v), 2) for v in d.get("offsets", [])]}


def estimate_project(project, image=None, slide=None) -> dict:
    """用一张图片（默认第一张有脑片的图片）估算串色比例，写进 project.unmix（保留开关）。返回新的设置。"""
    from .slide_io import open_slide
    im = image or next((x for x in project.images if x.sections), None) or (project.images[0] if project.images else None)
    if im is None:
        raise ValueError("项目里还没有图片。")
    own = slide is None
    sl = slide or open_slide(im.path)
    try:
        names = list(project.channel_names)
        used = [c for c in range(len(names)) if c not in getattr(project, "unused_channels", [])]
        pairs = [(j, i) for (j, i) in default_pairs([names[c] for c in used])]
        pairs = [(used[j], used[i]) for (j, i) in pairs]
        secs = [s for s in im.sections if s.enabled] or None
        tiles, masks, pix, off = sample_tiles(sl, secs)
        m = estimate(tiles, masks, pix, names, pairs=pairs, offsets=off)
    finally:
        if own:
            sl.close()
    auto = {_key(e.source, e.sink): round(float(m.matrix[e.sink, e.source]), 4) for e in m.pairs}
    d = dict(project.unmix or {})
    d.update(enabled=d.get("enabled", True), auto=auto, coef=dict(auto), offsets=[round(float(v), 3) for v in m.offsets],
             image=im.key, names=names, method="ridge",
             detail=describe(m))
    project.unmix = d
    project.save()
    return d


def prepare(project, slide=None, image=None) -> UnmixModel | None:
    """识别前调用：开启了但还没估算时，先用这张图片估算；把模型交给识别参数（project.detection.unmix）。"""
    d = project.unmix or {}
    if d.get("enabled") and "coef" not in d:
        estimate_project(project, image=image, slide=slide)
    m = model_from_settings(project.unmix, project.channel_names)
    project.detection.unmix = m
    project.detection.unmix_sig = signature(project.unmix) if m is not None else None
    return m


def describe_settings(d: dict, channel_names: list[str]) -> str:
    """写进 Excel“分析参数”表的一行文字。"""
    if not d or not d.get("enabled"):
        return "未使用"
    if "coef" not in d:
        return "已开启，尚未估算"
    parts = []
    for k, a in sorted(d["coef"].items()):
        j, i = _pairs_from_key(k)
        auto = d.get("auto", {}).get(k)
        nm = lambda c: channel_names[c] if c < len(channel_names) else f"通道{c + 1}"
        t = f"{nm(j)}→{nm(i)} {a * 100:.1f}%"
        if auto is not None and abs(auto - a) > 1e-4:
            t += f"（自动估算 {auto * 100:.1f}%，手动调整）"
        parts.append(t)
    src = f"；用“{d.get('image', '')}”估算" if d.get("image") else ""
    return ("识别前去除通道串色：" + "；".join(parts) if parts else "已开启，未发现明显串色") + src
