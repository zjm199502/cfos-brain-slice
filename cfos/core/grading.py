"""阳性细胞强度分级（弱 / 中 / 强）。

强度用“倍背景”表示：细胞内平均亮度 ÷ 该脑片组织的背景亮度（中位数）。这样不同脑片、不同图片之间可以比较。
分界值对整个项目统一（同一批实验必须用同一组分界值），保存在项目里。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADE_NAMES = {1: "弱", 2: "中", 3: "强"}


def fold(cells: pd.DataFrame, c: int) -> np.ndarray:
    if f"int_ch{c}" not in cells.columns:
        return np.full(len(cells), np.nan)
    bg = cells[f"bg_ch{c}"].to_numpy(float) if f"bg_ch{c}" in cells.columns else np.ones(len(cells))
    return cells[f"int_ch{c}"].to_numpy(float) / np.maximum(bg, 1.0)


def add_grades(cells: pd.DataFrame, channel_names, grading: dict) -> pd.DataFrame:
    """grading: {通道名: [弱/中分界, 中/强分界]}（倍背景）。只给设置了分界值的通道分级。"""
    if not grading or cells is None or cells.empty:
        return cells
    cells = cells.copy()
    for c, nm in enumerate(channel_names):
        cuts = grading.get(nm)
        if not cuts or len(cuts) != 2 or f"pos_ch{c}" not in cells.columns:
            continue
        f = fold(cells, c)
        g = np.where(f < cuts[0], 1, np.where(f < cuts[1], 2, 3))
        g = np.where(cells[f"pos_ch{c}"].to_numpy(bool) & np.isfinite(f), g, 0)
        cells[f"grade_ch{c}"] = g
    return cells


def suggest_cuts(values: np.ndarray) -> list[float]:
    """按全部阳性细胞的强度三等分（33% / 67% 分位数），四舍五入到 0.1。"""
    v = values[np.isfinite(values)]
    if len(v) < 10:
        return [3.0, 6.0]
    a, b = np.percentile(v, [33.3, 66.7])
    return [round(float(a), 1), round(float(b), 1)]


def project_folds(project, c: int, max_cells: int = 200000) -> np.ndarray:
    """项目中所有图片、所有启用脑片里通道 c 阳性细胞的倍背景强度（用于直方图和建议分界值）。"""
    from .coloc import merged_cells
    from .pipeline import drop_excluded
    out = []
    for im in project.images:
        df = project.load_cells(im)
        if df is None or df.empty or "merged_id" not in df.columns:
            continue
        en = {s.name for s in im.sections if s.enabled}
        cells = merged_cells(drop_excluded(df[df["section"].isin(en)], im), len(project.channel_names))
        if f"pos_ch{c}" in cells.columns:
            out.append(fold(cells[cells[f"pos_ch{c}"].astype(bool)], c))
    v = np.concatenate(out) if out else np.zeros(0)
    if len(v) > max_cells:
        v = np.random.default_rng(0).choice(v, max_cells, replace=False)
    return v


def histogram_png(values: np.ndarray, cuts, title: str, path):
    from .stats import _plt
    plt = _plt()
    v = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    hi = np.percentile(v, 99.5) if len(v) else 10
    ax.hist(v, bins=80, range=(0, hi), color="#888")
    for x, lab in zip(cuts or [], ("弱|中", "中|强")):
        ax.axvline(x, color="#d62728", lw=1.5)
        ax.text(x, ax.get_ylim()[1] * 0.97, f"{lab}\n{x}", color="#d62728", fontsize=8, va="top", ha="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    if cuts and len(v):
        n = [(v < cuts[0]).mean(), ((v >= cuts[0]) & (v < cuts[1])).mean(), (v >= cuts[1]).mean()]
        title += f"（弱 {n[0]:.0%} / 中 {n[1]:.0%} / 强 {n[2]:.0%}）"
    ax.set_title(title, fontsize=9); ax.set_xlabel("细胞平均亮度 ÷ 组织背景（倍）"); ax.set_ylabel("细胞数")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
