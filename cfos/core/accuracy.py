"""识别准确率检查：在一个小方框里人工点出所有真实的细胞，与软件识别结果一一配对，算出
查准率（软件找到的细胞里有多少是真的）、查全率（真的细胞里软件找到了多少）和 F1。

配对规则：人工点与软件细胞中心的距离不超过 MATCH_UM 才算同一个细胞，每个细胞只能配一次
（匈牙利算法，总距离最小）。只统计中心落在方框内的细胞。用的是软件自动识别的结果（不含手动补加的）。"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

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
    f1 = 2 * tp / (len(man) + len(auto)) if len(man) + len(auto) else float("nan")
    return {"n_manual": int(len(man)), "n_auto": int(len(auto)), "tp": tp, "fn": len(fn), "fp": len(fp),
            "precision": prec, "recall": rec, "f1": f1,
            "missed_xy": man[fn].tolist(), "false_xy": auto[fp].tolist(), "manual_xy": man.tolist()}


def record(project, image, section: str, channel: int, box0, res: dict, model: str, cells=None) -> dict:
    r = {"时间": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "图片": image.key, "脑片": section,
         "通道": project.channel_names[channel] if channel < len(project.channel_names) else str(channel),
         "模型": model or "模型未记录", "方框": [float(v) for v in box0],
         "人工计数": res["n_manual"], "软件识别": res["n_auto"], "配对成功": res["tp"],
         "漏检": res["fn"], "误检": res["fp"],
         "查准率": round(res["precision"], 4) if res["precision"] == res["precision"] else None,
         "查全率": round(res["recall"], 4) if res["recall"] == res["recall"] else None,
         "F1": round(res["f1"], 4) if res["f1"] == res["f1"] else None,
         "人工点": [[round(a, 1), round(b, 1)] for a, b in res["manual_xy"]]}
    if cells is None:
        cells = project.load_cells(image)
    source = cells.attrs.get("section_settings", {}).get(section) if cells is not None else None
    if source and channel in source["filters"]:
        snapshot = {"候选": json.loads(source["inference_settings"]),
                    "通道筛选": source["filters"][channel], "投影": source["z_mode"]}
        if source.get("model_identity") is not None:
            snapshot["识别引擎"] = source["model_identity"]
        hardware = cells.attrs.get("section_devices", {}).get(section, {})
        if channel in hardware:
            snapshot["实际计算设备"] = hardware[channel]
    else:
        # Historical tables can be evaluated, but cannot share a configuration ID
        # with current settings or unrelated sections of unknown provenance.
        snapshot = {"识别参数": "历史结果未记录", "图片": image.key, "脑片": section, "通道": channel}
    encoded = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    r["检查时参数"] = encoded
    r["参数编号"] = hashlib.sha256(encoded.encode()).hexdigest()[:12]
    r["原图"] = image.path
    r["视野类型"] = res.get("field_type", "常规")
    r["检查面积 mm²"] = res.get("field_area_mm2")
    r["动物"] = project.samples.get(f"{image.key}/{section}", {}).get("animal", "")
    r["视野编号"] = json.dumps([image.key, section, list(box0)], ensure_ascii=False)
    r["漏检线索"] = res.get("miss_details", [])
    project.accuracy.append(r)
    return r


def summary(project) -> pd.DataFrame:
    """按通道汇总所有检查记录（把各方框的计数加起来再算，比平均各框的百分比更稳）。"""
    if not project.accuracy:
        return pd.DataFrame()
    df = pd.DataFrame(project.accuracy)
    rows = []
    for key in ("模型", "参数编号", "视野类型"):
        if key not in df:
            df[key] = "历史未记录"
        df[key] = df[key].fillna("历史未记录")
    keys = ["通道", "模型", "参数编号", "视野类型"]
    if "视野编号" in df:
        identified = df["视野编号"].notna()
        df = pd.concat([df[~identified], df[identified].drop_duplicates(keys + ["视野编号"], keep="last")])
    for (ch, model, config, field), g in df.groupby(keys):
        tp, nm, na = g["配对成功"].sum(), g["人工计数"].sum(), g["软件识别"].sum()
        p = tp / na if na else float("nan"); r = tp / nm if nm else float("nan")
        f = 2 * tp / (nm + na) if nm + na else float("nan")
        area = g["检查面积 mm²"].sum(min_count=len(g)) if "检查面积 mm²" in g else np.nan
        rows.append({"通道": ch, "模型": model, "视野类型": field, "计数偏差": int(na - nm),
                     "误检数/mm²": (na - tp) / area if area > 0 else np.nan, "参数编号": config, "检查方框数": len(g), "人工计数": int(nm), "软件识别": int(na), "配对成功": int(tp),
                     "查准率": p, "查全率": r, "F1": f})
    return pd.DataFrame(rows)

def miss_details(project, image, channel, points, px_um, cells=None):
    """Explain nearby candidates with saved filtering settings whenever available."""
    from .manual import section_of
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    if not len(points):
        return []
    sections = {s.name: s for s in image.sections if s.enabled}
    cache = {}
    out = []
    from .field_review import effective_settings
    for x, y in points:
        name = section_of(image, x, y)
        if name not in cache:
            settings = cells.attrs.get("section_settings", {}).get(name, {}) if cells is not None else {}
            saved = settings.get("filters", {}).get(channel)
            if saved is not None:
                p = SimpleNamespace(**saved)
            else:
                p = (effective_settings(project.detection, image, sections[name]) if name in sections
                     else project.detection).channels[channel]
            raw = project.load_part(image, sections[name]) if name in sections else None
            candidates = raw[raw["channel"] == channel] if raw is not None else None
            tree = cKDTree(candidates[["x", "y"]]) if candidates is not None and len(candidates) else None
            tiles = raw.attrs.get("dark_tiles") if raw is not None else None
            cache[name] = candidates, tree, tiles, p, saved is not None
        candidates, tree, tiles, p, recorded = cache[name]
        nearby = tree.query_ball_point([x, y], MATCH_UM / px_um) if tree is not None else []
        if nearby:
            reasons = set()
            for i in nearby:
                row = candidates.iloc[i]
                failed = []
                if not p.min_area_um2 <= row.area_um2 <= p.max_area_um2:
                    failed.append("面积")
                if row[f"mean_ch{channel}"] < max(p.min_intensity, p.bg_fold * row.get("background", 0)):
                    failed.append("强度")
                if p.min_contrast > 0 and row.get("contrast", np.nan) < p.min_contrast:
                    failed.append("对比度")
                reasons.add("、".join(failed) if failed else "通过筛选，请查配对及人工编辑")
            source = "结果保存的阈值" if recorded else "当前阈值，历史阈值未记录"
            reason = f"附近候选（按{source}）：" + " / ".join(sorted(reasons))
        elif candidates is None:
            reason = "缺少对应脑片候选缓存"
        elif tiles is None:
            reason = "无邻近候选，旧缓存缺少暗块记录"
        elif any(c == channel and x0 <= x < x1 and y0 <= y < y1 for c, x0, y0, x1, y1 in tiles):
            reason = "所在分块因亮度过低跳过"
        else:
            reason = "无邻近原始候选（含基础面积和组织范围筛选）"
        out.append({"x": float(x), "y": float(y), "线索": reason})
    return out
