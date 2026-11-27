"""Allen CCFv3（25 µm）图谱：读取、按任意斜切平面切片、脑区查询。

坐标约定：
- 图谱数组（brainglobe "asr"）：axis0 前→后（AP），axis1 上→下（DV），axis2 右→左（LR）。
- 切片平面使用 QuickNII / DeepSlice 的锚定方式：o, u, v 三个向量（单位为 25 µm 体素，
  QuickNII 坐标系 x: 左→右，y: 后→前，z: 下→上）。图像上归一化坐标 (s, t)（s 向右、t 向下，0..1）
  对应的图谱位置为 o + s·u + t·v。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .resources import allen_atlas_dir
from . import atlas_choice, stam

SHAPE_ASR = (528, 320, 456)


@dataclass
class Atlas:
    annotation: np.ndarray   # uint16，0 为背景，其余为 structures 表中的 index
    reference: np.ndarray    # uint8，平均模板
    hemispheres: np.ndarray  # uint8，1/2
    structures: pd.DataFrame

    @property
    def n_regions(self):
        return len(self.structures)


def load_atlas(name=None) -> Atlas:
    return _load_atlas(atlas_choice.current() if name is None else name)


@lru_cache(maxsize=2)
def _load_atlas(name):
    if name == atlas_choice.STAM:
        from .region_sets import ZH
        reference, labels = stam.volumes()
        structures = pd.read_csv(stam.DATA / "structures.csv", keep_default_na=False)
        structures["name_zh"] = structures["acronym"].map(ZH).fillna("")
        return Atlas(labels, reference, np.where(labels > 0, 2, 0).astype(np.uint8), structures)
    if name != atlas_choice.CCF:
        raise ValueError(f"未知图谱：{name}")
    d = allen_atlas_dir()
    z = np.load(d / "allen_ccfv3_25um.npz")
    st = pd.read_csv(d / "structures.csv")
    zh = d / "structures_zh.csv"
    if zh.exists():
        z2 = pd.read_csv(zh)
        st = st.merge(z2[["id", "name_zh"]], on="id", how="left")
    if "name_zh" not in st.columns:
        st["name_zh"] = ""
    st["name_zh"] = st["name_zh"].fillna("")
    return Atlas(z["annotation"], z["reference"], z["hemispheres"], st)


def native_mm(edit_coords):
    return np.asarray(edit_coords)[..., (0, 2, 1)] * np.array([.025, -.025, -.025])


def edit_coords(native):
    return np.asarray(native)[..., (0, 2, 1)] * np.array([40., -40., -40.])


def qn_to_asr(p: np.ndarray) -> np.ndarray:
    """QuickNII 体素坐标 (x, y, z) → 图谱数组下标 (a, s, r)。最后一维为坐标。"""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([SHAPE_ASR[0] - 1 - y, SHAPE_ASR[1] - 1 - z, SHAPE_ASR[2] - 1 - x], -1)


@dataclass
class Plane:
    o: np.ndarray
    u: np.ndarray
    v: np.ndarray
    atlas: str = atlas_choice.CCF

    def to_list(self):
        return [float(x) for x in np.concatenate([self.o, self.u, self.v])]

    @staticmethod
    def from_list(a):
        a = np.asarray(a, float)
        return Plane(a[0:3], a[3:6], a[6:9])

    # ---- 便于手动调整的参数化 ----
    def center(self):
        return self.o + 0.5 * self.u + 0.5 * self.v

    def ap_mm(self):
        """平面中心的前后位置（相对 Bregma，mm；Bregma 约在 QuickNII y=540·… 这里用常用近似值）。"""
        if self.atlas == atlas_choice.STAM:
            return float(5.230 - native_mm(self.center())[2])
        y = self.center()[1]
        return (y - 332.0) * 0.025  # 约定：QuickNII y≈332 对应 Bregma 0（Allen CCF 常用换算）


def sample_plane(plane: Plane, width: int, height: int, which=("annotation",), order_ref=1,
                 warp: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """按平面切出 height×width 的图谱图像。warp=(dx, dy) 为每个像素的额外位移（像素单位，非线性修正）。"""
    at = load_atlas(plane.atlas)
    t, s = np.mgrid[0:height, 0:width].astype(np.float32)
    if warp is not None:
        s = s + warp[0]
        t = t + warp[1]
    s = (s + 0.5) / width
    t = (t + 0.5) / height
    p = plane.o[None, None, :] + s[..., None] * plane.u[None, None, :] + t[..., None] * plane.v[None, None, :]
    if plane.atlas == atlas_choice.STAM:
        coords = np.moveaxis(native_mm(p)[..., ::-1], -1, 0)
        return {key: ndi.map_coordinates(getattr(at, key), coords / (.032 if key == "reference" else .040),
                                         order=order_ref if key == "reference" else 0, mode="constant")
                for key in which}
    idx = qn_to_asr(p)
    coords = [idx[..., 0], idx[..., 1], idx[..., 2]]
    out = {}
    for w in which:
        vol = getattr(at, w)
        order = order_ref if w == "reference" else 0
        out[w] = ndi.map_coordinates(vol, coords, order=order, mode="constant", cval=0)
    return out


def region_boundaries(label: np.ndarray) -> np.ndarray:
    """脑区边界掩膜。"""
    b = np.zeros(label.shape, bool)
    b[:, 1:] |= label[:, 1:] != label[:, :-1]
    b[1:, :] |= label[1:, :] != label[:-1, :]
    return b


def region_colors(label: np.ndarray) -> np.ndarray:
    at = load_atlas()
    lut = np.zeros((at.n_regions + 1, 3), np.uint8)
    lut[1:] = at.structures[["r", "g", "b"]].to_numpy()
    return lut[label]
