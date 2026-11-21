"""按脑片和通道保存人工盲评参数，在同一标注框比较两种设置。"""
from dataclasses import replace
import json

import numpy as np

from . import accuracy, detection


def section_key(section):
    return f"{section.name}@{','.join(map(str, section.bbox0))}"


def effective_settings(settings, image, section):
    overrides = image.review_settings.get(section_key(section), {})
    if not overrides:
        return settings
    channels = dict(settings.channels)
    for c, p in channels.items():
        values = overrides.get(str(c))
        if values:
            channels[c] = replace(p, min_contrast=values["min_contrast"], deconv=values["deconv"])
    return replace(settings, channels=channels)


def compare(slide, section, settings, channel, box, manual, candidate, cancel, norm_cache=None):
    """两版共享全脑片归一化范围，并在评估框外留出分块边界。"""
    p = settings.channels[channel]
    norms = detection.preview_norms(slide, section, {channel: p}, settings.unmix, norm_cache, cancel)
    x, y, w, h = box
    result = []
    raw = None
    raw_key = None
    for param in (p, replace(p, **candidate)):
        if cancel():
            raise InterruptedError
        local = replace(settings, channels={c: replace(param if c == channel else v, detect=c == channel)
                                             for c, v in settings.channels.items()})
        key = detection.raw_signature(local)
        if key != raw_key:
            raw = detection.detect_section(slide, section, local, cancel=cancel, norms=norms,
                                           prefilter=False, field_box=box)
            raw = raw[(raw.x >= x) & (raw.x < x+w) & (raw.y >= y) & (raw.y < y+h)]
            raw_key = key
        df = detection.apply_filters(raw, local)
        score = accuracy.check(df, channel, box, manual, slide.pixel_size_um)
        result.append((score, df))
    old, new = result[0][1], result[1][1]
    limit = accuracy.MATCH_UM / slide.pixel_size_um
    shared, lost, gained = accuracy.match(old[["x","y"]].to_numpy(), new[["x","y"]].to_numpy(), limit)
    pairs, _, _ = accuracy.match(np.asarray(result[1][0]["manual_xy"]).reshape(-1,2),
                                 new[["x","y"]].to_numpy(), limit)
    confirmed = len(set(gained) & {j for _,j in pairs})
    return result, lost, gained, confirmed

def pooled_reviews(records, current):
    """同一脑片、通道和计算参数的重复检查框只取最近一次。"""
    keys = ("位置标识", "通道", "当前", "候选", "识别条件")
    def config(record):
        return json.dumps([record.get(k) for k in keys], sort_keys=True)
    wanted = config(current)
    fields = {tuple(r["视野"]): r for r in records if config(r) == wanted}
    lines = []
    for label in ("当前", "候选"):
        tp, fp, fn = (sum(r[label+"计数"][k] for r in fields.values()) for k in ("tp","fp","fn"))
        den = 2*tp + fp + fn
        score = f"{2*tp/den:.2f}" if den else "—"
        negative = [r for r in fields.values() if not r["人工点"]]
        area = sum(r["面积mm2"] for r in negative)
        rate = sum(r[label+"计数"]["fp"] for r in negative)/area if area else None
        lines.append(f"{label} F1 {score}" + (f"，阴性区误检/mm² {rate:.1f}" if rate is not None else ""))
    return f"同参数 {len(fields)} 个不同框：" + "；".join(lines)
