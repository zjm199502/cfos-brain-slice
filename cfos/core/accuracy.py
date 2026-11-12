"""识别准确率检查：在一个小方框里人工点出所有真实的细胞，与软件识别结果一一配对，算出
查准率（软件找到的细胞里有多少是真的）、查全率（真的细胞里软件找到了多少）和 F1。

配对规则：人工点与软件细胞中心的距离不超过 MATCH_UM 才算同一个细胞，每个细胞只能配一次
（匈牙利算法，总距离最小）。只统计中心落在方框内的细胞。用的是软件自动识别的结果（不含手动补加的）。"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

MATCH_UM = 6.0


def match(manual_xy: np.ndarray, auto_xy: np.ndarray, max_dist: float):
    """返回 (配上的 [(i_manual, j_auto)], 未配上的人工点下标, 未配上的软件细胞下标)。"""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    nm, na = len(manual_xy), len(auto_xy)
    if nm == 0 or na == 0:
        return [], list(range(nm)), list(range(na))
    d = cdist(manual_xy, auto_xy)
    big = max_dist * 1000.0
    cost = np.where(d <= max_dist, d, big)
    r, c = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if d[i, j] <= max_dist]
    mi = {i for i, _ in pairs}; aj = {j for _, j in pairs}
    return pairs, [i for i in range(nm) if i not in mi], [j for j in range(na) if j not in aj]


def check(cells: pd.DataFrame, channel: int, box0, manual_xy, px_um: float, match_um: float = MATCH_UM) -> dict:
    """box0 = (x, y, w, h)（第 0 层像素）；manual_xy = 人工点的 (x, y)（第 0 层像素）。"""
    x, y, w, h = box0
    man = np.asarray(manual_xy, float).reshape(-1, 2)
    if len(man):
        inb = (man[:, 0] >= x) & (man[:, 0] < x + w) & (man[:, 1] >= y) & (man[:, 1] < y + h)
        man = man[inb]
    g = cells[(cells["channel"] == channel) & (cells["x"] >= x) & (cells["x"] < x + w)
              & (cells["y"] >= y) & (cells["y"] < y + h)]
    if "manual" in g.columns:
        g = g[g["manual"] == 0]
    auto = g[["x", "y"]].to_numpy(float)
    pairs, fn, fp = match(man, auto, match_um / px_um)
    tp = len(pairs)
    prec = tp / len(auto) if len(auto) else float("nan")
    rec = tp / len(man) if len(man) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) else float("nan")
    return {"n_manual": int(len(man)), "n_auto": int(len(auto)), "tp": tp, "fn": len(fn), "fp": len(fp),
            "precision": prec, "recall": rec, "f1": f1,
            "missed_xy": man[fn].tolist(), "false_xy": auto[fp].tolist(), "manual_xy": man.tolist()}


def record(project, image, section: str, channel: int, box0, res: dict, model: str) -> dict:
    r = {"时间": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "图片": image.key, "脑片": section,
         "通道": project.channel_names[channel] if channel < len(project.channel_names) else str(channel),
         "模型": model or "通用模型", "方框": [float(v) for v in box0],
         "人工计数": res["n_manual"], "软件识别": res["n_auto"], "配对成功": res["tp"],
         "漏检": res["fn"], "误检": res["fp"],
         "查准率": round(res["precision"], 4) if res["precision"] == res["precision"] else None,
         "查全率": round(res["recall"], 4) if res["recall"] == res["recall"] else None,
         "F1": round(res["f1"], 4) if res["f1"] == res["f1"] else None,
         "人工点": [[round(a, 1), round(b, 1)] for a, b in res["manual_xy"]]}
    project.accuracy.append(r)
    return r


def summary(project) -> pd.DataFrame:
    """按通道汇总所有检查记录（把各方框的计数加起来再算，比平均各框的百分比更稳）。"""
    if not project.accuracy:
        return pd.DataFrame()
    df = pd.DataFrame(project.accuracy)
    rows = []
    for ch, g in df.groupby("通道"):
        tp, nm, na = g["配对成功"].sum(), g["人工计数"].sum(), g["软件识别"].sum()
        p = tp / na if na else float("nan"); r = tp / nm if nm else float("nan")
        f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
        rows.append({"通道": ch, "检查方框数": len(g), "人工计数": int(nm), "软件识别": int(na), "配对成功": int(tp),
                     "查准率": p, "查全率": r, "F1": f})
    return pd.DataFrame(rows)
