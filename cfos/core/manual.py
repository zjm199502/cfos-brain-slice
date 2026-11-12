"""手动校正细胞：学生在图上补加漏掉的细胞、删除误识别的细胞。

校正记录按位置保存（results/<图片>_manual.json），与识别结果分开：
- 以后改阈值、重新识别后，校正仍然有效（删除记录会去掉 3 µm 内最近的同通道细胞）；
- 每张脑片每个通道补加 / 删除了多少个，会写进 Excel，说明人工改动的比例。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ADD_RADIUS_UM = 4.0      # 补加细胞的默认半径
MATCH_UM = 3.0           # 删除记录与细胞的匹配距离


def manual_path(project, image) -> Path:
    return project.results_dir / f"{image.key}_manual.json"


def load_manual(project, image) -> dict:
    p = manual_path(project, image)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return {"added": d.get("added", []), "removed": d.get("removed", [])}
        except (OSError, ValueError):
            pass
    return {"added": [], "removed": []}


def save_manual(project, image, d: dict):
    p = manual_path(project, image)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def section_of(image, x, y) -> str:
    """点所在的脑片（按方框判断），不在任何脑片里返回空字符串。"""
    for s in image.sections:
        bx, by, bw, bh = s.bbox0
        if s.enabled and bx <= x < bx + bw and by <= y < by + bh:
            return s.name
    return ""


def measure_added(slide, x, y, px_um) -> tuple[list[float], list]:
    """补加细胞：在原图上取半径 ADD_RADIUS_UM 的圆，测各通道平均亮度；返回 (means, contour)。"""
    r = ADD_RADIUS_UM / px_um
    x0, y0 = int(x - r - 1), int(y - r - 1)
    a = slide.read(0, (max(x0, 0), max(y0, 0), int(2 * r + 3), int(2 * r + 3))).astype(np.float32)
    yy, xx = np.mgrid[0:a.shape[1], 0:a.shape[2]]
    m = (xx + max(x0, 0) - x) ** 2 + (yy + max(y0, 0) - y) ** 2 <= r * r
    means = [float(a[c][m].mean()) if m.any() else 0.0 for c in range(a.shape[0])]
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    contour = np.round(np.stack([x + r * np.cos(t), y + r * np.sin(t)], 1), 1).tolist()
    return means, contour


def apply_manual(df: pd.DataFrame, manual: dict, px_um: float, n_ch: int) -> pd.DataFrame:
    """把校正记录应用到（已按阈值筛选的）细胞表上。增加列 manual：0 = 自动识别，1 = 手动补加。"""
    df = df.copy()
    df["manual"] = 0
    if not manual or (not manual["added"] and not manual["removed"]):
        return df
    keep = np.ones(len(df), bool)
    tol = MATCH_UM / px_um
    for c in {int(r["channel"]) for r in manual["removed"]}:
        sel = np.nonzero(df["channel"].to_numpy() == c)[0]
        if not len(sel):
            continue
        tree = cKDTree(df.iloc[sel][["x", "y"]].to_numpy(float))
        pts = np.array([[r["x"], r["y"]] for r in manual["removed"] if int(r["channel"]) == c], float)
        d, k = tree.query(pts, distance_upper_bound=tol)
        keep[sel[k[np.isfinite(d)]]] = False
    df = df[keep]
    add = []
    for r in manual["added"]:
        row = {"section": r["section"], "channel": int(r["channel"]), "x": r["x"], "y": r["y"],
               "area_um2": np.pi * ADD_RADIUS_UM ** 2, "contour": r["contour"], "manual": 1,
               "background": np.nan, "contrast": np.nan}
        for k in range(n_ch):
            row[f"mean_ch{k}"] = r["means"][k] if k < len(r["means"]) else np.nan
        add.append(row)
    if add:
        df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
    return df.reset_index(drop=True)


def correction_counts(manual: dict) -> pd.DataFrame:
    """每张脑片每个通道：补加数、删除数。"""
    rows = [(r["section"], int(r["channel"]), "补加") for r in manual["added"]] + \
           [(r.get("section", ""), int(r["channel"]), "删除") for r in manual["removed"]]
    if not rows:
        return pd.DataFrame(columns=["section", "channel", "补加", "删除"])
    t = pd.DataFrame(rows, columns=["section", "channel", "kind"])
    return t.pivot_table(index=["section", "channel"], columns="kind", aggfunc="size", fill_value=0).reset_index() \
        .reindex(columns=["section", "channel", "补加", "删除"], fill_value=0)
