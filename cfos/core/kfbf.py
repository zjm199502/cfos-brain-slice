"""读取江丰生物（KFBIO）扫描仪的荧光全片格式 .kfbf（纯 Python，不需要厂家 SDK）。

格式是根据实际文件分析出来的（2026-11，样本：DAPI/520/670 三通道，20x）：

文件头（小端）
  0x00  4B  块标记 f1 01 ee ee
  0x04  4B  "KFBF"
  0x10  u32 图块总数（所有缩放层合计）
  0x14  u32 最高倍率层的高（像素）
  0x18  u32 最高倍率层的宽（像素）
  0x1c  u32 最高倍率（如 20）
  0x20  4B  "JPEG"（图块编码）
  0x34  u32 缩略图块位置
  0x3c  u32 标签图块位置
  0x44  u64 图块索引表起点
  0x4c  f32 最高倍率层每像素微米数（如 0.3263）
  0x5c 起是 ff 01 ee ee 块：u32 条目数，然后是 TLV（标签 u32、长度 u32、值）：0x4b = 通道数，
        0x4d = 通道名表的位置（每个名字 40 字节，以 0 结尾），0x4c = 滤光片编号表（每个 20 字节）。
图块索引表：每条 64 字节，以 f1 04 ee ee 开头，其后
  i32 x, i32 y, i32 w, i32 h（该缩放层内的像素坐标）, f32 倍率,
  i32, i32, i32 第 1 通道字节数, u64 → 各通道 JPEG 起点表（每通道 u64）,
  u64 → 各通道 JPEG 字节数表（每通道 u64）
每个通道是一张独立的 8 位灰度 JPEG（256×256，边缘可能更小）。
只存有组织的图块，没有存的图块视为 0（黑）。
各缩放层的倍率是 20、10、5、2.5 …（每层减半）。
"""
from __future__ import annotations

import io
import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TILE_MARK = b"\xf1\x04\xee\xee"


def _decode_jpeg(buf: bytes) -> np.ndarray:
    try:
        import imagecodecs
        return np.asarray(imagecodecs.jpeg8_decode(buf))
    except Exception:  # noqa: BLE001
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(buf)))


@dataclass
class KfbfLevel:
    scale: float
    shape: tuple[int, int]                       # (H, W)
    tiles: dict = field(default_factory=dict)    # (x, y) -> (w, h, offsets, sizes)
    tile: int = 256


class KfbfFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mm
        if mm[4:8] != b"KFBF":
            raise ValueError("不是 KFBF 文件")
        n_tiles, H, W, mag = struct.unpack_from("<4I", mm, 0x10)
        self.magnification = float(mag)
        idx0 = struct.unpack_from("<Q", mm, 0x44)[0]
        self.pixel_size_um = float(struct.unpack_from("<f", mm, 0x4c)[0])
        if not (0.05 < self.pixel_size_um < 5):
            self.pixel_size_um = 10.0 / max(self.magnification, 1) * 0.5
        self.channel_names, self.n_channels = self._read_channel_info()
        # 图块索引
        if mm[idx0:idx0 + 4] != TILE_MARK:
            idx0 = mm.find(TILE_MARK)
        levels: dict[float, KfbfLevel] = {}
        C = self.n_channels
        p = idx0
        for _ in range(n_tiles):
            if mm[p:p + 4] != TILE_MARK:
                break
            x, y, w, h, sc, _a, _b, _s0, po, pz = struct.unpack_from("<4if3iQQ", mm, p + 4)
            offs = struct.unpack_from(f"<{C}Q", mm, po)
            szs = struct.unpack_from(f"<{C}Q", mm, pz)
            lv = levels.get(sc)
            if lv is None:
                lv = levels[sc] = KfbfLevel(sc, (0, 0))
            lv.tiles[(x, y)] = (w, h, offs, szs)
            lv.shape = (max(lv.shape[0], y + h), max(lv.shape[1], x + w))
            p += 64
        if not levels:
            raise ValueError("KFBF 文件里没有找到图块索引")
        top = max(levels)
        # 各层尺寸按最高层等比推算（只存了组织图块，最大坐标可能偏小）
        for sc, lv in levels.items():
            f = top / sc
            lv.shape = (int(np.ceil(H / f)), int(np.ceil(W / f)))
        self.levels = [levels[k] for k in sorted(levels, reverse=True)]
        if self.magnification != top:
            self.magnification = float(top)

    def _read_channel_info(self):
        mm = self._mm
        p = 0x5c
        n_ch, names_at = None, None
        # 先找 ff01eeee 后的 TLV 段
        start = mm.find(b"\xff\x01\xee\xee", 0, 0x200)
        if start >= 0:
            p = start + 8          # 跳过块标记和条目数
            end = mm.find(b"\xf1\x02\xee\xee", p, 0x1000)
            end = end if end > 0 else p + 0x200
            while p + 8 <= end:
                tag, ln = struct.unpack_from("<II", mm, p)
                if ln > 64:
                    break
                if tag == 0x4b and ln == 4:
                    n_ch = struct.unpack_from("<I", mm, p + 8)[0]
                elif tag == 0x4d and ln == 8:
                    names_at = struct.unpack_from("<Q", mm, p + 8)[0]
                p += 8 + ln
        if not n_ch or n_ch > 8:
            n_ch = 3
        names = []
        if names_at:
            for i in range(n_ch):
                raw = bytes(mm[names_at + 40 * i: names_at + 40 * (i + 1)]).split(b"\0")[0]
                names.append(raw.decode("utf-8", "replace") or f"通道{i + 1}")
        if len(names) != n_ch:
            names = [f"通道{i + 1}" for i in range(n_ch)]
        return names, n_ch

    # ------------------------------------------------------------------ 读取
    def read(self, level: int, x0: int, y0: int, w: int, h: int,
             channels: list[int] | None = None) -> np.ndarray:
        """读取第 level 层（0 = 最高倍率）里的一块，坐标是该层的像素坐标，返回 (C, h, w) uint8。"""
        lv = self.levels[level]
        chs = list(range(self.n_channels)) if channels is None else list(channels)
        out = np.zeros((len(chs), h, w), np.uint8)
        T = lv.tile
        tx0, ty0 = (x0 // T) * T, (y0 // T) * T
        for ty in range(ty0, y0 + h, T):
            for tx in range(tx0, x0 + w, T):
                t = lv.tiles.get((tx, ty))
                if t is None:
                    continue
                tw, th, offs, szs = t
                sx0, sy0 = max(x0, tx), max(y0, ty)
                sx1, sy1 = min(x0 + w, tx + tw), min(y0 + h, ty + th)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                for k, c in enumerate(chs):
                    a = _decode_jpeg(self._mm[offs[c]:offs[c] + szs[c]])
                    if a.ndim == 3:
                        a = a[..., 0]
                    out[k, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        return out

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


class _KfbfLevelArray:
    """把一个缩放层包装成 arr[c, ys, xs] 形式，供 slide_io.Slide 使用。"""

    def __init__(self, kf: KfbfFile, level: int):
        self.kf, self.level = kf, level
        H, W = kf.levels[level].shape
        self.shape = (kf.n_channels, H, W)
        self.ndim = 3
        self.dtype = np.dtype(np.uint8)

    def __getitem__(self, key):
        c, ys, xs = key
        H, W = self.shape[1:]
        y0, y1, _ = ys.indices(H)
        x0, x1, _ = xs.indices(W)
        cs = [c] if isinstance(c, (int, np.integer)) else list(range(self.shape[0]))[c]
        a = self.kf.read(self.level, x0, y0, x1 - x0, y1 - y0, cs)
        return a[0] if isinstance(c, (int, np.integer)) else a


def guess_color(name: str, idx: int) -> tuple[int, int, int]:
    n = name.lower()
    try:
        wl = int("".join(ch for ch in n if ch.isdigit()) or 0)
    except ValueError:
        wl = 0
    if "dapi" in n or (400 <= wl < 480):
        return (0, 0, 255)
    if 480 <= wl < 560:
        return (0, 255, 0)
    if 560 <= wl < 640:
        return (255, 0, 0)
    if wl >= 640:
        return (255, 0, 255)
    return [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)][idx % 4]
