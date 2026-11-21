"""读取奥林巴斯 / Evident（cellSens、VS120、VS200）的 .vsi 荧光图像（纯 Python，不需要 Java / Bio-Formats）。

一次扫描由两部分组成，缺一不可：
  名字.vsi                          很小，只有元数据（通道名、像素尺寸、维度顺序）和一张缩略图
  _名字_/stackN/frame_t*.ets        真正的像素（每幅图一个 stackN 文件夹）
所以拷贝图片时必须把 .vsi 和同名的 “_名字_” 文件夹一起拷走。

.vsi 的元数据由 vsi_meta（取自 bioio-vsi，BSD-3）解析；.ets 的图块在这里读：
  主文件头 64 字节  <4sIIIQIIQIIQII：SIS\\0、头长、版本、坐标维数 ndim、第二文件头位置和长度、…、图块表位置、图块数
  第二文件头        ETS\\0、版本、像素类型（2 = 8 位，4 = 16 位）、每像素分量数、颜色空间、压缩方式
                    （0 不压缩，2 JPEG，3 JPEG 2000，5 无损 JPEG，8 PNG，9 BMP）、质量、图块宽、高、深；
                    偏移 172/176 是拼接扫描的原点偏移（0 或小的负数）；偏移 184 起是各维大小
  图块表            每条 4 × (ndim + 5) 字节：u32、ndim 个 i32 坐标、u64 文件位置、u32 字节数、u32
  坐标 = (图块列, 图块行, 维度 0, 维度 1, …, 金字塔层)；各维度是 Z / T / 通道中的哪一个由 .vsi 说明，
  同一台显微镜的不同文件顺序也可能不同，不能假定。
没有存的图块按 0（黑）处理。
"""
from __future__ import annotations

import io
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAIN = struct.Struct("<4sIIIQIIQIIQII")
_PIXEL = {2: np.dtype(np.uint8), 4: np.dtype(np.uint16)}
COMPRESSION_NAMES = {0: "不压缩", 2: "JPEG", 3: "JPEG 2000", 5: "无损 JPEG", 8: "PNG", 9: "BMP"}
# .vsi 里每幅图的类型（2005/2074）：这些不是要分析的主扫描图
_AUX_TYPES = {1: "预览图", 4: "对焦图", 256: "标签", 1024: "对焦点", 8388608: "样本掩膜"}


class VsiReadError(ValueError):
    """读不了这个 .vsi，文字是给用户看的中文说明。"""


@dataclass
class _EtsHeader:
    path: str
    ndim: int
    dtype: np.dtype
    ncomp: int
    compression: int
    tile_w: int
    tile_h: int
    width: int                   # 声明的图像宽、高（第 0 层）
    height: int
    slot_sizes: tuple            # 宽高之后各维的大小（文件头里写的）
    shift_x: int                 # 图像从存储网格的 (shift_x, shift_y) 像素开始（拼接扫描，通常是 0）
    shift_y: int
    tabpos: int
    ntiles: int


@dataclass
class EtsIndex(_EtsHeader):
    tiles: dict                  # 坐标元组 -> (文件位置, 字节数)
    cmax: tuple                  # 每个坐标的最大值
    n_levels: int


def _ets_header(path, f) -> _EtsHeader:
    size = os.fstat(f.fileno()).st_size
    main = f.read(64)
    if len(main) < 64 or main[:4] != b"SIS\x00":
        raise VsiReadError("像素文件 .ets 的文件头不对（可能没有拷贝完整）")
    h = _MAIN.unpack(main)
    ndim, addpos, addsize, tabpos, ntiles = h[3], h[4], h[5], h[7], h[8]
    f.seek(addpos)
    sec = f.read(addsize)
    if len(sec) < 188 or sec[:4] != b"ETS\x00":
        raise VsiReadError("像素文件 .ets 的第二文件头不对（可能没有拷贝完整）")
    _ver, ptype, ncomp, _cs, comp, _q, tw, th, _td = struct.unpack_from("<9I", sec, 4)
    if ptype not in _PIXEL:
        raise VsiReadError(f"暂不支持的像素类型（代码 {ptype}），目前支持 8 位和 16 位")
    n_ext = struct.unpack_from("<I", sec, 184)[0]
    if n_ext < 2 or len(sec) < 188 + 4 * n_ext:
        raise VsiReadError("像素文件 .ets 里的图像尺寸信息不完整")
    ext = struct.unpack_from(f"<{n_ext}I", sec, 188)
    ox, oy = struct.unpack_from("<ii", sec, 172)
    rec = 4 * (ndim + 5)
    if ntiles <= 0 or tw <= 0 or th <= 0 or ndim < 3:
        raise VsiReadError("像素文件 .ets 里没有图块")
    if tabpos + rec * ntiles > size:
        raise VsiReadError("像素文件 .ets 不完整（图块表超出了文件末尾），请重新拷贝")
    if not ext[0] or not ext[1]:
        raise VsiReadError("像素文件 .ets 里的图像宽高必须大于零")
    return _EtsHeader(str(path), ndim, _PIXEL[ptype], ncomp, comp, tw, th, ext[0], ext[1],
                      tuple(ext[2:]), max(0, -ox), max(0, -oy), tabpos, ntiles)


def read_ets_index(path: str | os.PathLike) -> EtsIndex:
    with open(path, "rb") as f:
        header = _ets_header(path, f)
        size = os.fstat(f.fileno()).st_size
        ndim = header.ndim
        f.seek(header.tabpos)
        tab = np.frombuffer(f.read(4 * (ndim + 5) * header.ntiles), dtype=np.dtype(
            [("u0", "<u4"), ("c", "<i4", (ndim,)), ("off", "<u8"), ("len", "<u4"), ("u1", "<u4")]))
    coords = tab["c"]
    ok = (tab["off"].astype(np.int64) + tab["len"] <= size) & (coords >= 0).all(axis=1)
    tiles = {}
    offs, lens = tab["off"].tolist(), tab["len"].tolist()
    for i, c in enumerate(coords.tolist()):
        if ok[i]:
            tiles[tuple(c)] = (offs[i], lens[i])
    if not tiles:
        raise VsiReadError("像素文件 .ets 不完整（图块都在文件末尾之外），请重新拷贝")
    cmax = tuple(int(v) for v in coords[ok].max(axis=0))
    return EtsIndex(**vars(header), tiles=tiles, cmax=cmax, n_levels=cmax[-1] + 1)


def _decode(raw, compression: int, dtype: np.dtype, shape: tuple) -> np.ndarray:
    """解码一个图块，返回 (tile_h, tile_w) 的数组（边缘图块已补齐到整块大小）。"""
    if compression == 0:
        a = np.frombuffer(raw, dtype=dtype.newbyteorder("<"), count=shape[0] * shape[1]).reshape(shape)
        return a.astype(dtype, copy=False)
    try:
        import imagecodecs
        if compression == 2:
            a = imagecodecs.jpeg8_decode(raw) if dtype == np.uint8 else imagecodecs.jpeg_decode(raw)
        elif compression == 3:
            a = imagecodecs.jpeg2k_decode(raw)
        elif compression == 5:
            a = imagecodecs.ljpeg_decode(raw)
        elif compression == 8:
            a = imagecodecs.png_decode(raw)
        elif compression == 9:
            a = imagecodecs.bmp_decode(raw)
        else:
            raise VsiReadError(f"暂不支持的压缩方式（代码 {compression}）")
    except ImportError:
        if compression not in (2, 8, 9):
            raise VsiReadError("解码 JPEG 2000 图块需要 imagecodecs 组件，请重新安装完整安装包") from None
        from PIL import Image
        a = np.asarray(Image.open(io.BytesIO(bytes(raw))))
    a = np.asarray(a)
    if a.ndim == 3:            # 灰度图被存成多分量时取第一个分量
        a = a[..., 0]
    if a.dtype != dtype:
        a = a.astype(dtype)
    if a.shape != shape:       # 个别文件的边缘图块没有补齐
        b = np.zeros(shape, dtype)
        hh, ww = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
        b[:hh, :ww] = a[:hh, :ww]
        a = b
    return np.ascontiguousarray(a)


@dataclass
class _Level:
    shape: tuple          # (H, W)


def companion_dir(vsi_path: Path) -> Path:
    return vsi_path.parent / f"_{vsi_path.stem}_"


@dataclass(frozen=True)
class VsiScene:
    scene_id: int                 # The on-disk stack number, stable when the file is moved.
    name: str
    width: int
    height: int


def _scene_candidates(path, scene_id=None, notes=None):
    from . import vsi_meta

    folder = companion_dir(path)
    stacks = {}
    if folder.is_dir():
        for child in folder.iterdir():
            number = child.name[5:]
            if child.name[:5].lower() != "stack" or not number.isascii() or not number.isdigit() or not child.is_dir():
                continue
            # Split time series can have multiple ETS files in one stack. Use one
            # time point, matching the metadata reader, never a new scene per frame.
            frames = sorted(p for p in child.iterdir() if p.name.lower().startswith("frame_t") and p.suffix.lower() == ".ets")
            if frames:
                stacks[int(number)] = frames[0]
    if not stacks:
        raise VsiReadError(f"没有找到像素文件夹“{folder.name}”里的 stack…/frame_t….ets。"
                           "请将 .vsi 和配套的像素文件夹一起拷贝到同一目录。")

    why = None
    missing_external = []
    try:
        tree = vsi_meta.load_tree(str(path))
        metas, problems = vsi_meta.parse_all_images(tree, missing_external=missing_external)
        types = vsi_meta.stack_image_types(tree) if problems else {}
        if problems and not metas:
            why = problems[0]
    except Exception as exc:  # Some cellSens versions have unreadable metadata but intact ETS scans.
        metas, types, why = [], {}, str(exc)
    by_id = {m.stack_id: m for m in metas}
    candidates, errors = [], []
    for number in missing_external:
        if number in stacks or types.get(number) in _AUX_TYPES or scene_id not in (None, number):
            continue
        issue = f"扫描区 stack{number} 声明了外部像素文件，但未找到 {folder.name}/stack{number}/frame_t*.ets。"
        errors.append(issue)
        if notes is not None:
            notes.append(issue)
    for number, ets in sorted(stacks.items()):
        if scene_id is not None and number != scene_id:
            continue
        meta = by_id.get(number)
        if (meta.image_type if meta else types.get(number)) in _AUX_TYPES:
            continue
        try:
            with ets.open("rb") as f:
                header = _ets_header(ets, f)
            if header.ncomp != 1:
                errors.append(f"stack{number} 是明场 RGB 彩色扫描，本软件只分析荧光图。")
                continue
            if header.compression not in COMPRESSION_NAMES:
                raise VsiReadError(f"暂不支持的压缩方式（代码 {header.compression}）")
        except (OSError, VsiReadError) as exc:
            issue = f"未能读取扫描区 stack{number}（{ets.name}）：{exc}"
            errors.append(issue)
            if notes is not None:
                notes.append(issue)
            continue
        if notes is not None and meta is None:
            notes.append(f"扫描区 stack{number} 已按 ETS 数据读取，请手动核对扫描区、通道名和像素尺寸。")
        name = (meta.name if meta else None) or f"stack{number}"
        scene = VsiScene(number, name, header.width, header.height)
        candidates.append((scene, meta, ets))
    if not candidates:
        if errors:
            raise VsiReadError("；".join(errors))
        if scene_id is not None:
            raise VsiReadError(f"未找到可分析的扫描区 stack{scene_id}。请检查 .vsi 及配套的像素文件夹。")
        raise VsiReadError("这个 .vsi 没有可分析的荧光扫描区。")
    # Legacy open without a selector prefers a metadata-backed main scan, then
    # the largest area. The stack number breaks ties independently of filenames.
    def rank(candidate):
        scene, meta, _ = candidate
        area = (meta.width * meta.height) if meta else (scene.width * scene.height)
        return meta is None, -area, scene.scene_id

    candidates.sort(key=rank)
    return candidates, why


def list_vsi_scenes(path: str | Path, *, notes: list[str] | None = None) -> list[VsiScene]:
    """List fluorescence scenes in legacy selection order from headers alone."""
    candidates, _ = _scene_candidates(Path(path), notes=notes)
    return [scene for scene, _, _ in candidates]


class VsiFile:
    """读取一个稳定 stack ID 对应的荧光扫描区。旧项目未记录 ID 时选择原来的最大主扫描图。"""

    def __init__(self, path: str | Path, z_mode: str = "max", scene_id: int | None = None,
                 channel_order: list[str] | None = None):
        self.path = Path(path)
        self.z_mode = z_mode
        self.notes: list[str] = []
        candidates, why = _scene_candidates(self.path, scene_id)
        errors = []
        for scene, meta, ets in candidates:
            try:
                idx = read_ets_index(ets)
                break
            except (VsiReadError, OSError) as exc:
                errors.append(f"stack{scene.scene_id}：{exc}")
        else:
            raise VsiReadError("；".join(errors))
        self.scene_id = scene.scene_id
        if meta is None:
            self.notes.append("没能读懂这个扫描区的 .vsi 说明信息" + (f"（{why}）" if why else "")
                              + "，请手动核对扫描区、通道名和像素尺寸。")
        if scene_id is None and len(candidates) > 1:
            self.notes.append(f"这个 .vsi 里有 {len(candidates)} 幅扫描图，当前使用“{scene.name}”（stack{scene.scene_id}）。")
        self.index = idx
        self.dtype = idx.dtype
        self._layout(meta, idx)
        self._order_channels(channel_order)
        self.pixel_size_um = None
        self.image_name = None
        if meta is not None:
            self.image_name = meta.name
            ps = meta.pixel_size_um
            if ps is not None and 0.01 < float(ps.x) < 100:
                self.pixel_size_um = float(ps.x)
        self.levels = []
        for lv in range(idx.n_levels):
            hh, ww = idx.height >> lv, idx.width >> lv
            if hh < 1 or ww < 1:
                break
            self.levels.append(_Level((hh, ww)))
        self._fh = open(idx.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        from .tileread import TileCache
        self.cache = TileCache()

    # ------------------------------------------------------------------ 选图、定维度
    def _layout(self, meta, idx: EtsIndex):
        """确定通道 / Z / 时间各在第几个坐标上。"""
        n_slots = idx.ndim - 3
        self.slot = {}                       # "c" / "z" / "t" -> 坐标下标
        names, emis = [], []
        if meta is not None:
            kinds = [s.kind for s in meta.slots if s.kind is not None]
            for i, k in enumerate(kinds[:n_slots]):
                self.slot.setdefault(k.lower(), 2 + i)
            for s in meta.slots:
                if s.kind == "C":
                    chs = sorted(s.channels, key=lambda c: c.index)
                    names = [(c.name or c.alt_name or "").strip() for c in chs]
                    emis = [c.emission_nm for c in chs]
        else:
            # 没有元数据时的常见顺序：只有一维 = 通道；两维 = (Z, 通道)；三维 = (时间, Z, 通道)
            order = {1: "c", 2: "zc", 3: "tzc"}.get(n_slots, "")
            for i, k in enumerate(order):
                self.slot[k] = 2 + i
        size = lambda k: (idx.cmax[self.slot[k]] + 1) if k in self.slot else 1  # noqa: E731
        self.n_channels, self.n_z, self.n_t = size("c"), size("z"), size("t")
        if self.n_t > 1:
            self.notes.append(f"这是时间序列（{self.n_t} 个时间点），只使用第 1 个时间点。")
        self.channel_names, self.channel_ids, self.emission_nm = [], [], []
        for i in range(self.n_channels):
            name = names[i] if i < len(names) else ""
            emission = emis[i] if i < len(emis) else None
            label = name or f"通道{i + 1}"
            # A wavelength can identify an unnamed channel even when its position changes.
            identity = f"{emission:g} nm" if not name and emission else label
            self.channel_names.append(label)
            self.channel_ids.append(identity)
            self.emission_nm.append(emission)

    def _order_channels(self, expected):
        self._channel_indices = list(range(self.n_channels))
        if not expected:
            return
        raw = [name.strip().casefold() for name in self.channel_ids]
        wanted = [name.strip().casefold() for name in expected]
        if raw == wanted:
            return
        if len(raw) != len(wanted) or set(raw) != set(wanted) or len(set(raw)) != len(raw):
            raise VsiReadError(f"扫描区 stack{self.scene_id} 无法按通道名唯一匹配。"
                               f"当前通道为 {', '.join(self.channel_ids)}，同一原图的参照顺序为 {', '.join(expected)}。")
        self._channel_indices = [raw.index(name) for name in wanted]
        self.channel_ids = list(expected)
        self.channel_names = [self.channel_names[i] for i in self._channel_indices]
        self.emission_nm = [self.emission_nm[i] for i in self._channel_indices]

    # ------------------------------------------------------------------ 读取
    def _tile(self, level: int, c: int, z: int, tx: int, ty: int):
        key = (level, c, z, tx, ty)
        a = self.cache.get(key)
        if a is not None:
            return a
        idx = self.index
        coord = [0] * idx.ndim
        coord[0], coord[1], coord[-1] = tx, ty, level
        if "c" in self.slot:
            coord[self.slot["c"]] = c
        if "z" in self.slot:
            coord[self.slot["z"]] = z
        t = idx.tiles.get(tuple(coord))
        if t is None:
            return None
        off, ln = t
        a = _decode(self._mm[off:off + ln], idx.compression, idx.dtype, (idx.tile_h, idx.tile_w))
        self.cache.put(key, a)
        return a

    def _plane(self, level: int, c: int, z: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """一个通道、一个 z 层里的一块，(h, w)。"""
        from .tileread import pmap
        idx = self.index
        out = np.zeros((h, w), idx.dtype)
        if w <= 0 or h <= 0:
            return out
        tw, th = idx.tile_w, idx.tile_h
        sx, sy = x0 + (idx.shift_x >> level), y0 + (idx.shift_y >> level)   # 存储网格里的坐标
        jobs = [(tx, ty) for ty in range(sy // th, (sy + h - 1) // th + 1)
                for tx in range(sx // tw, (sx + w - 1) // tw + 1)]
        tiles = pmap(lambda j: self._tile(level, c, z, j[0], j[1]), jobs)
        for (tx, ty), a in zip(jobs, tiles):
            if a is None:
                continue
            xa, ya = max(sx, tx * tw), max(sy, ty * th)
            xb, yb = min(sx + w, (tx + 1) * tw), min(sy + h, (ty + 1) * th)
            out[ya - sy:yb - sy, xa - sx:xb - sx] = a[ya - ty * th:yb - ty * th, xa - tx * tw:xb - tx * tw]
        return out

    def read(self, level: int, x0: int, y0: int, w: int, h: int,
             channels: list[int] | None = None) -> np.ndarray:
        """读取第 level 层里的一块（该层的像素坐标），返回 (C, h, w)；z-stack 按 z_mode 合成。"""
        from .slide_io import project_z
        chs = list(range(self.n_channels)) if channels is None else list(channels)
        out = np.zeros((len(chs), h, w), self.dtype)
        mode = self.z_mode or "max"
        for k, logical in enumerate(chs):
            c = self._channel_indices[logical]
            if self.n_z <= 1:
                out[k] = self._plane(level, c, 0, x0, y0, w, h)
            elif mode.startswith("z") and mode[1:].isdigit():
                out[k] = self._plane(level, c, min(int(mode[1:]) - 1, self.n_z - 1), x0, y0, w, h)
            elif mode == "max":        # 逐层取最大值，不把所有层同时放进内存
                acc = self._plane(level, c, 0, x0, y0, w, h)
                for z in range(1, self.n_z):
                    np.maximum(acc, self._plane(level, c, z, x0, y0, w, h), out=acc)
                out[k] = acc
            else:
                st = np.stack([self._plane(level, c, z, x0, y0, w, h) for z in range(self.n_z)])
                out[k] = project_z(st, mode)
        return out

    def close(self):
        self.cache.clear()
        try:
            self._mm.close()
        finally:
            self._fh.close()
