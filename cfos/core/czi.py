"""Read single-file ZEISS fluorescence CZI with scene-local, level-zero coordinates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from ._czifile import CziFile
from .tileread import TileCache


@dataclass(frozen=True)
class CziScene:
    scene_id: int
    name: str


def _axis(entry, dim):
    """Absolute half-open coordinates in the original CZI dimension."""
    if dim not in entry.dims:
        return range(1)
    i = entry.dims.index(dim)
    return range(entry.start[i], entry.start[i] + entry.shape[i])


def _catalogue(file):
    if file.header.update_pending:
        raise ValueError("CZI 文件仍在写入，请等 ZEN 完成保存后重新打开。")
    # Use the complete directory to detect external parts, including those
    # which contain no local full-resolution tiles.
    entries = file.subblock_directory
    if file.header.file_part or any(e.file_part for e in entries):
        raise ValueError("暂不支持多文件分卷 CZI，请在 ZEN 导出为单文件 CZI 或 OME-TIFF。")
    if not entries:
        raise ValueError("CZI 中没有图像数据。")
    base = [e for e in entries if e.pyramid_type == 0 and not e.is_pyramid]
    unmarked = [e for e in entries if e.pyramid_type == 0 and e.is_pyramid]
    if unmarked:
        _verify_redundant_overviews(base, unmarked)
        entries = [e for e in entries if e.pyramid_type or not e.is_pyramid]
    if not base:
        raise ValueError("CZI 缺少可读取的原始分辨率灰度平面。")
    ids = {e.scene_index for e in base}
    if len(ids) > 1 and -1 in ids:
        raise ValueError("CZI 同时包含有场景编号和无编号的原始图块，无法安全区分扫描区。")
    scenes = defaultdict(list)
    implicit = ids == {-1}
    for e in entries:
        if e.scene_index in ids:
            scenes[0 if implicit else e.scene_index].append(e)
    names = {}
    xml = file.xml_element
    if xml is not None:
        for el in xml.findall(".//Information/Image/Dimensions/S/Scenes/Scene"):
            idx = el.get("Index", "")
            if idx.isdecimal() and el.get("Name"):
                names[int(idx)] = el.get("Name")
    channels = sorted({c for e in base for c in _axis(e, "C")})
    size_c = xml.findtext(".//Information/Image/SizeC") if xml is not None else None
    if size_c and int(size_c) != len(channels):
        raise ValueError("CZI 标注的通道数量与实际图块不符，请在 ZEN 核实文件是否完整。")
    return scenes, names, channels


def list_czi_scenes(path: str | Path, *, notes: list[str] | None = None) -> list[CziScene]:
    with CziFile(path) as file:
        scenes, names, _ = _catalogue(file)
        if notes is not None:
            for i, entries in sorted(scenes.items()):
                times = _time_points(entries)
                if len(times) > 1:
                    notes.append(_time_note(i, times))
        return [CziScene(i, names.get(i, f"Scene {i}")) for i in sorted(scenes)]


def _time_points(entries):
    return sorted({t for e in entries if e.pyramid_type == 0 for t in _axis(e, "T")})


def _time_note(scene, times):
    return f"CZI 场景 S={scene} 共 {len(times)} 个时间点，仅分析第一个时间点 T={times[0]}。"


def _channel_metadata(xml, ids):
    info = {}
    if xml is not None:
        for source in (".//Information/Image/Dimensions/Channels/Channel",
                       ".//DisplaySetting/Channels/Channel"):
            for el in xml.findall(source):
                match = re.fullmatch(r"Channel:(\d+)", el.get("Id", ""))
                if match:
                    record = info.setdefault(int(match[1]), {})
                    for key, text in (("name", el.get("Name")), ("dye", el.findtext("Fluor")),
                                      ("color", el.findtext("Color")), ("dye", el.findtext("DyeName"))):
                        if text:
                            record.setdefault(key, text)
    result = []
    for channel in ids:
        rec = info.get(channel, {})
        color = rec.get("color", "").lstrip("#")
        rgb = None
        if re.fullmatch(r"(?:[0-9A-Fa-f]{2})?[0-9A-Fa-f]{6}", color):
            rgb = tuple(bytes.fromhex(color[-6:]))  # ZEN uses ARGB
        result.append((rec.get("name") or rec.get("dye") or f"通道 {channel + 1}", rgb))
    return result


def _spacing(xml):
    um = {}
    if xml is not None:
        for el in xml.findall(".//Scaling/Items/Distance"):
            if el.get("Id") in {"X", "Y"}:
                um[el.get("Id")] = float(el.findtext("Value", "nan")) * 1e6
    if len(um) != 2 or any(not math.isfinite(v) or v <= 0 for v in um.values()):
        raise ValueError("CZI 的 X/Y 像素标定不完整，请在 ZEN 核对标定并重新导出。")
    if not math.isclose(um["X"], um["Y"], rel_tol=1e-4):
        raise ValueError("CZI 使用非正方形像素。请先在 ZEN 重采样为等距 X/Y 像素再导入。")
    return um["X"]


@dataclass(frozen=True)
class _Tile:
    entry: object
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_entry(cls, entry):
        x, y = _axis(entry, "X"), _axis(entry, "Y")
        return cls(entry, x.start, y.start, len(x), len(y))


class _Plane:
    """Index tile footprints for a single C/Z plane, ordered by acquisition M."""
    def __init__(self, tiles):
        self.tiles = sorted(tiles, key=lambda t: (t.entry.mosaic_index, t.entry.file_position))
        self.bin_size = max(64, int(np.median([max(t.width, t.height) for t in self.tiles])))
        self.bins = defaultdict(list)
        for i, t in enumerate(self.tiles):
            for gy in range(t.y // self.bin_size, (t.y + t.height - 1) // self.bin_size + 1):
                for gx in range(t.x // self.bin_size, (t.x + t.width - 1) // self.bin_size + 1):
                    self.bins[gx, gy].append(i)

    def intersect(self, x, y, width, height):
        xs = range(x // self.bin_size, (x + width - 1) // self.bin_size + 1)
        ys = range(y // self.bin_size, (y + height - 1) // self.bin_size + 1)
        if len(xs) * len(ys) >= len(self.tiles):
            return self.tiles
        return [self.tiles[i] for i in sorted({i for gy in ys for gx in xs for i in self.bins.get((gx, gy), ())})]

    def covers(self, box):
        """Is the whole rectangle covered by native tiles, including mosaic seams?"""
        right, bottom = box.x + box.width, box.y + box.height
        spans = []
        for t in self.intersect(box.x, box.y, box.width, box.height):
            x0, x1 = max(t.x, box.x), min(t.x + t.width, right)
            y0, y1 = max(t.y, box.y), min(t.y + t.height, bottom)
            if (x0, y0, x1, y1) == (box.x, box.y, right, bottom):
                return True
            if x0 < x1 and y0 < y1:
                spans.append((x0, x1, y0, y1))
        spans.sort()
        edges = {box.y}
        for _, _, top, end in spans:
            edges.add(top)
            if end < bottom:
                edges.add(end)
        for y in edges:
            covered = box.x
            for left, end, top, lower in spans:
                if top <= y < lower:
                    if left > covered:
                        return False
                    covered = max(covered, end)
                    if covered == right:
                        break
            if covered != right:
                return False
        return True

    def footprint(self):
        # Exact footprint so a partial overview never replaces base-level data.
        # Limit uncommon expensive shapes: read generated levels from source tiles.
        rects = {(t.x, t.y, t.x + t.width, t.y + t.height) for t in self.tiles}
        if len(rects) > 256:
            return rects
        bands = []
        edges = sorted({y for r in rects for y in (r[1], r[3])})
        for top, bottom in zip(edges, edges[1:]):
            spans = []
            for left, right in sorted((l, r) for l, t, r, b in rects if t <= top < b):
                if spans and left <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(spans[-1][1], right))
                else:
                    spans.append((left, right))
            if bands and bands[-1][2] == spans:
                bands[-1] = (bands[-1][0], bottom, spans)
            else:
                bands.append((top, bottom, spans))
        return bands


def _verify_redundant_overviews(base, unmarked):
    # Some CZI writers omit the pyramid flag. Only discard reduced XY data
    # when native pixels cover its entire area in the same acquisition.
    def acquisition(e):
        axes = sorted((d, a, n) for d, a, n in zip(e.dims, e.start, e.shape)
                      if d not in ("X", "Y") and (a != 0 or n != 1))
        return e.scene_index, e.pixel_type, tuple(axes)

    grouped = defaultdict(list)
    for e in base:
        grouped[acquisition(e)].append(_Tile.from_entry(e))
    planes = {key: _Plane(tiles) for key, tiles in grouped.items()}
    for e in unmarked:
        plane = planes.get(acquisition(e))
        xy_reduced = {"X", "Y"} <= set(e.dims) and all(
            0 < stored < size if d in ("X", "Y") else stored == size
            for d, stored, size in zip(e.dims, e.stored_shape, e.shape))
        if not xy_reduced or plane is None or not plane.covers(_Tile.from_entry(e)):
            raise ValueError("CZI 含非标准重采样图块，无法确认其范围内的全部原始像素，请从 ZEN 导出 OME-TIFF。")


@dataclass(frozen=True)
class _Level:
    scale: int
    shape: tuple[int, int]
    planes: dict


class CziReader:
    def __init__(self, path, scene_id, z_mode):
        self.file = CziFile(path, memmap=True)
        self.cache = TileCache()
        try:
            scenes, _, ids = _catalogue(self.file)
            if scene_id is None:
                if len(scenes) != 1:
                    raise ValueError("此 CZI 含多个扫描区，请从项目中按扫描区打开。")
                scene_id = next(iter(scenes))
            if scene_id not in scenes:
                raise ValueError(f"CZI 没有场景 S={scene_id}。")
            self.scene_id = scene_id
            self.channel_ids = ids
            self.channels = _channel_metadata(self.file.xml_element, ids)
            self.n_channels = len(ids)
            self.pixel_size_um = _spacing(self.file.xml_element)
            self.z_mode = z_mode
            entries = scenes[scene_id]
            self.times = _time_points(entries)
            self.notes = [_time_note(scene_id, self.times)] if len(self.times) > 1 else []
            first = [e for e in entries if self.times[0] in _axis(e, "T")]
            full = [e for e in entries if e.pyramid_type == 0]
            expected_z = {z for e in full for z in _axis(e, "Z")}
            if len(self.times) > 1 and any(e.mosaic_index >= 0 for e in full):
                # M identifiers may be renumbered between time points. Count
                # distinct acquired tiles per C/Z plane independently of M order.
                counts = defaultdict(set)
                for e in full:
                    for c in _axis(e, "C"):
                        for z in _axis(e, "Z"):
                            for t in _axis(e, "T"):
                                counts[t, c, z].add(e.mosaic_index)
                if any((self.times[0], c, z) in counts and len(tiles) > len(counts[self.times[0], c, z])
                       for (t, c, z), tiles in counts.items()):
                    raise ValueError(f"CZI 的首个时间点 T={self.times[0]} 缺少拼接图块，无法完整分析。")
            self._index(first, expected_z)
        except Exception:
            self.close()
            raise

    def _index(self, entries, expected_z):
        base = [e for e in entries if e.pyramid_type == 0 and not e.is_pyramid]
        if any(e.pixel_type.name not in {"GRAY8", "GRAY16"} for e in base):
            raise ValueError("目前支持 CZI 的 8/16 位单色荧光通道。RGB、浮点及其他像素类型请另存为多通道 OME-TIFF。")
        if any("X" not in e.dims or "Y" not in e.dims for e in base):
            raise ValueError("CZI 不是二维 XY 图像。")
        # A singleton acquisition axis is harmless. Distinct angles, phases,
        # or detector states must never be merged into one brain section.
        extra = {d for e in base for d in e.dims} - set("XYZCST")
        for dim in sorted(extra):
            values = {_axis(e, dim).start for e in base}
            if len(values) != 1 or any(len(_axis(e, dim)) != 1 for e in base):
                raise ValueError(f"CZI 含多值 {dim} 轴，请先在 ZEN 选择一个时间点或采集状态再导出。")
        self.dtype = np.result_type(*{e.pixel_type.dtype for e in base})
        full = self._planes(base)
        self.z_ids = sorted({z for c, z in full})
        self.n_z = len(self.z_ids)
        if (set(full) != {(c, z) for c in self.channel_ids for z in expected_z}
                or self.z_ids != list(range(self.z_ids[0], self.z_ids[-1] + 1))):
            raise ValueError("此 CZI 场景缺少通道或 Z 层，无法完整读取。")
        tiles = [_Tile.from_entry(e) for e in base]
        self.x = min(t.x for t in tiles)
        self.y = min(t.y for t in tiles)
        width = max(t.x + t.width for t in tiles) - self.x
        height = max(t.y + t.height for t in tiles) - self.y
        native = defaultdict(list)
        for e in entries:
            if e.pyramid_type and e.pixel_type.name in {"GRAY8", "GRAY16"}:
                ix, iy = e.dims.index("X"), e.dims.index("Y")
                ratio = round(e.shape[ix] / e.stored_shape[ix])
                if ratio > 1 and all(abs(e.shape[i] - ratio * e.stored_shape[i]) < ratio for i in (ix, iy)):
                    native[ratio].append(e)
        scales = {1, *native}
        scale = 1
        while max(width, height) > 1024 * scale:
            scale *= 2
            scales.add(scale)
        self.levels = []
        usable = full
        for scale in sorted(scales):
            if scale in native:
                candidate = self._planes(native[scale])
                if set(candidate) == set(full) and all(candidate[k].footprint() == full[k].footprint() for k in full):
                    usable = candidate
            shape = ((height + scale - 1) // scale, (width + scale - 1) // scale)
            self.levels.append(_Level(scale, shape, usable))

    @staticmethod
    def _planes(entries):
        grouped = defaultdict(list)
        for e in entries:
            if int(e.compression) not in {0, 1, 2, 4, 5, 6, 7}:
                raise ValueError(f"暂不支持 CZI 的压缩类型 {e.compression}。请在 ZEN 导出为无压缩或 Zstd CZI。")
            tile = _Tile.from_entry(e)
            for c in _axis(e, "C"):
                for z in _axis(e, "Z"):
                    grouped[c, z].append(tile)
        return {key: _Plane(tiles) for key, tiles in grouped.items()}

    def _read_plane(self, level, x, y, w, h, channel, z):
        step = level.scale
        plane = level.planes[channel, z]
        gx, gy = self.x + x * step, self.y + y * step
        out = np.zeros((h, w), self.dtype)
        for t in plane.intersect(gx, gy, w * step, h * step):
            x0 = max(0, (t.x - gx + step - 1) // step)
            x1 = min(w, (t.x + t.width - gx + step - 1) // step)
            y0 = max(0, (t.y - gy + step - 1) // step)
            y1 = min(h, (t.y + t.height - gy + step - 1) // step)
            if x1 <= x0 or y1 <= y0:
                continue
            e = t.entry
            img = self.cache.get(e.file_position) if e.compression else None
            if img is None:
                block = e.read_segment_data(self.file)
                size = e.stored_size * e.pixel_type.dtype.itemsize
                if e.compression and block.data_size == size:
                    # The upstream reader interprets equal-size payloads as raw
                    # despite their codec declaration. The encoding is ambiguous.
                    raise ValueError("CZI 图块压缩标记与数据长度有歧义，请在 ZEN 导出为无压缩 CZI 或 OME-TIFF。")
                if not e.compression and block.data_size != size:
                    raise ValueError("CZI 原始图块的字节数与声明尺寸不一致。")
                img = block.data(copy=False)
                if img.dtype != e.pixel_type.dtype or img.shape != e.stored_shape:
                    raise ValueError("CZI 图块的解码像素类型或尺寸与目录不一致。")
                if e.compression:
                    self.cache.put(e.file_position, img)
            indices = {"C": channel, "Z": z, "T": self.times[0]}
            index = tuple(slice(None) if d in {"X", "Y"} else indices.get(d, e.start[i]) - e.start[i]
                          for i, d in enumerate(e.dims))
            src = img[index]
            if e.dims.index("X") < e.dims.index("Y"):
                src = src.T
            if step == 1 and src.shape == (t.height, t.width):
                ty, tx = gy + y0 - t.y, gx + x0 - t.x
                out[y0:y1, x0:x1] = src[ty:ty+y1-y0, tx:tx+x1-x0]
            else:
                yy = (gy + np.arange(y0, y1) * step - t.y) * src.shape[0] // t.height
                xx = (gx + np.arange(x0, x1) * step - t.x) * src.shape[1] // t.width
                out[y0:y1, x0:x1] = src[np.ix_(yy, xx)]
        return out

    def read(self, level, x, y, w, h, channels):
        from .slide_io import project_z
        lv = self.levels[level]
        if not w or not h or not channels:
            return np.empty((len(channels), h, w), dtype=self.dtype)
        z_ids = self.z_ids
        if self.z_mode.startswith("z") and self.z_mode[1:].isdigit():
            index = int(self.z_mode[1:]) - 1
            if not 0 <= index < self.n_z:
                raise ValueError(f"当前 CZI 场景只有 {self.n_z} 个 Z 层。")
            z_ids = [z_ids[index]]
        # The local sharpness and median filters need neighbouring pixels to
        # give the same selection at ROI boundaries as a full-scene EDF.
        rx, ry, rw, rh = x, y, w, h
        if self.z_mode == "edf" and self.n_z > 1:
            rx, ry = max(0, x - 12), max(0, y - 12)
            rw = min(lv.shape[1], x + w + 12) - rx
            rh = min(lv.shape[0], y + h + 12) - ry
        out = []
        for c in channels:
            channel = self.channel_ids[c]
            planes = (self._read_plane(lv, rx, ry, rw, rh, channel, z) for z in z_ids)
            if len(z_ids) == 1:
                a = next(planes)
            elif self.z_mode == "max":
                a = next(planes)
                for p in planes:
                    np.maximum(a, p, out=a)
            else:
                a = project_z(np.stack(list(planes)), self.z_mode)
            out.append(a[y-ry:y-ry+h, x-rx:x-rx+w])
        return np.stack(out)

    def close(self):
        self.cache.clear()
        self.file.close()
