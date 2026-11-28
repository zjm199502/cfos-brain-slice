"""⑤ 共标统计页的汇总表：按图片缓存，读细胞结果和汇总都可以放在后台线程。"""
from __future__ import annotations

import pandas as pd

from .coloc import merged_cells, summarize


def image_signature(project, im, channel_names, require_nucleus):
    """汇总结果依赖的全部输入；细胞文件没变、脑片和排除区域没改时，签名不变，可以直接用缓存。"""
    for path in (project.cells_path(im), project._old_cells_path(im)):
        try:
            st = path.stat()
        except OSError:
            continue
        enabled = tuple(sorted(s.name for s in im.sections if s.enabled))
        exclusions = tuple(tuple(tuple(float(v) for v in pt) for pt in poly) for poly in im.exclusions)
        outlines = tuple(sorted(im.changed_outlines.items()))
        return (im.key, str(path), st.st_size, st.st_mtime_ns, enabled, exclusions, outlines,
                tuple(channel_names), bool(require_nucleus))
    return None


def image_summary(project, im, channel_names, require_nucleus):
    """一张图片按脑片汇总的共标表；没有共标结果时返回 None。"""
    from .pipeline import drop_excluded
    df = project.load_cells(im)
    if df is None or df.empty or "merged_id" not in df.columns:
        return None
    if "combo" in df.columns:
        df = df[df["combo"].notna()]      # 共标结果已清除的脑片显示为待计算，不列入汇总
    enabled = {s.name for s in im.sections if s.enabled}
    cells = merged_cells(drop_excluded(df[df["section"].isin(enabled)], im), len(channel_names))
    cells.insert(0, "图片", im.key)
    return summarize(cells, channel_names, ("图片", "section"), require_nucleus=require_nucleus)


def combine(frames):
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True).rename(columns={"section": "脑片"}) if frames else pd.DataFrame()


class LRUDict(dict):
    """最多保留 maxsize 条、按最近使用淘汰的字典。用于细胞结果缓存：大项目不把所有图片的细胞常驻内存。"""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().pop(key)
        super().__setitem__(key, value)  # 移到末尾 = 最近使用
        return value

    def __setitem__(self, key, value):
        super().pop(key, None)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            super().pop(next(iter(self)))
