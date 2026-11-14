"""细胞轮廓的紧凑存储（v0.8.1）。

以前每个细胞的轮廓是 Python 列表 [[x, y], ...] 存在表格里，一张脑片几万个细胞时
内存和文件都很大、打开项目也慢。现在：
- 内存里：每个细胞的轮廓是 float32 的 (N, 2) 数组（共用一块连续内存的视图）；
- 文件里：所有轮廓拼成一个 float32 数组 + 每个细胞的起点（offsets），和表格的其余列一起保存。
旧格式（Python 列表）照样能读，读入时自动转换。"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORMAT = 2


def as_array(c) -> np.ndarray:
    a = np.asarray(c, np.float32)
    return a.reshape(-1, 2) if a.size else np.zeros((0, 2), np.float32)


def pack(df: pd.DataFrame) -> dict:
    """表格 → {"format", "cells"（不含 contour 列）, "contour_xy", "contour_off"}。"""
    if df is None or "contour" not in df.columns:
        return {"format": FORMAT, "cells": df, "contour_xy": None, "contour_off": None}
    arrs = [as_array(c) for c in df["contour"].to_numpy()]
    lens = np.fromiter((len(a) for a in arrs), np.int64, len(arrs))
    off = np.zeros(len(arrs) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    xy = np.concatenate(arrs) if arrs else np.zeros((0, 2), np.float32)
    return {"format": FORMAT, "cells": df.drop(columns="contour"), "contour_xy": xy.astype(np.float32),
            "contour_off": off}


def unpack(d: dict) -> pd.DataFrame:
    df = d["cells"]
    xy, off = d.get("contour_xy"), d.get("contour_off")
    if df is None or xy is None:
        return df
    df = df.copy()
    col = np.empty(len(df), object)
    for i in range(len(df)):
        col[i] = xy[off[i]:off[i + 1]]
    df["contour"] = col
    return df


def normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """旧格式（Python 列表）的轮廓列 → float32 数组。"""
    if df is None or "contour" not in df.columns or not len(df):
        return df
    first = df["contour"].iloc[0]
    if isinstance(first, np.ndarray) and first.dtype == np.float32:
        return df
    return unpack(pack(df))
