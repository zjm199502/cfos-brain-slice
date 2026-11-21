"""读取多通道荧光全片图像（qptiff / ome.tif / tif / 江丰 .kfbf / 奥林巴斯 .vsi），支持金字塔分辨率按需读取。"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import zarr

DEFAULT_COLORS = {
    "DAPI": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "RED": (255, 0, 0),
    "FAR RED": (255, 0, 255),
}


@dataclass
class ChannelInfo:
    index: int
    name: str
    color: tuple[int, int, int]
    file_color: tuple[int, int, int] | None = None   # 文件里存的显示颜色（显微镜软件里的颜色；没有时为 None）（v0.8.3）


@dataclass
class Slide:
    path: Path
    channels: list[ChannelInfo]
    pixel_size_um: float  # 第 0 层每像素微米数
    level_shapes: list[tuple[int, int]]  # 每层 (H, W)
    _tif: tifffile.TiffFile = field(repr=False, default=None)
    _levels: list = field(repr=False, default_factory=list)
    axes: str = "CYX"                 # 原始文件的轴顺序（可能含 Z）
    n_z: int = 1                      # z-stack 层数，1 表示单层扫描 / 已做景深合成
    z_mode: str = "max"               # z-stack 的合成方式：max=最大投影，edf=景深合成（取最清晰的层），或 "z3" 表示只用第 3 层
    _kf: object = field(repr=False, default=None)   # 江丰 .kfbf（v0.8.2）或奥林巴斯 .vsi（v0.8.4）文件，不是 TIFF 时用它读
    notes: list = field(default_factory=list)       # 打开时给用户的提示（如 .vsi 里有多幅图、用了哪一幅）
    _fast: dict = field(repr=False, default_factory=dict)   # 每层的多线程图块读取器（v0.8.3，见 tileread.py）；None = 用 zarr
    _cache: object = field(repr=False, default=None)        # 已解码图块缓存（v0.8.3）
    _fh: object = field(repr=False, default=None)
    _fh_lock: object = field(repr=False, default=None)
    preview_norm_cache: dict = field(repr=False, default_factory=dict)
    scene_id: int | None = None
    vsi_channels: tuple[str, ...] = ()  # ordered instrument channel identities for this VSI

    @property
    def n_levels(self) -> int:
        return len(self.level_shapes)

    def downsample(self, level: int) -> float:
        return self.level_shapes[0][1] / self.level_shapes[level][1]

    def vsi_source_signature(self):
        """区分同路径下的扫描区、通道索引和外置 ETS 像素版本。"""
        ets = Path(self._kf.index.path).stat()
        return [self.scene_id, list(self._kf._channel_indices), ets.st_size, ets.st_mtime_ns]

    def level_for_pixel_size(self, target_um: float) -> int:
        """返回像素尺寸不超过 target_um 的最粗一层。"""
        best = 0
        for lv in range(self.n_levels):
            if self.pixel_size_um * self.downsample(lv) <= target_um * 1.01:
                best = lv
        return best

    def _arr(self, level: int):
        while len(self._levels) <= level:
            self._levels.append(None)
        if self._levels[level] is None and self._kf is not None:
            from .kfbf import _KfbfLevelArray
            self._levels[level] = _KfbfLevelArray(self._kf, level)
        if self._levels[level] is None:
            raw = zarr.open(self._tif.series[0].aszarr(level=level), mode="r")
            self._levels[level] = raw if self.axes in ("CYX", "YX") else ZProjected(raw, self.axes, self.z_mode)
        return self._levels[level]

    def _fast_level(self, level: int):
        """图块存储的单层多通道 TIFF（qptiff / ome.tif 常见）：多线程解码、带缓存的读取器（v0.8.3）。
        其他情况（z-stack、条带存储、RGB 等）返回 None，仍按原来的方式读。"""
        if level in self._fast:
            return self._fast[level]
        r = None
        if (self._tif is not None and self.axes in ("CYX", "YX")
                and os.environ.get("CFOS_FAST_READ", "1") != "0"):
            try:
                from . import tileread
                pages = list(self._tif.series[0].levels[level].pages)
                if pages and tileread.TiledTiffLevel.supported(pages):
                    if self._cache is None:
                        import threading
                        self._cache = tileread.TileCache()
                        self._fh = open(self.path, "rb")
                        self._fh_lock = threading.Lock()
                    r = tileread.TiledTiffLevel(self.path, pages, level, self._cache, self._fh_lock, self._fh)
                    if (r.H, r.W) != tuple(self.level_shapes[level]) and level < len(self.level_shapes):
                        r = None
            except Exception:  # noqa: BLE001  任何意外都退回原来的读法
                r = None
        self._fast[level] = r
        return r

    def read(self, level: int, bbox0: tuple[int, int, int, int] | None = None,
             channels: list[int] | None = None) -> np.ndarray:
        """读取一块区域，返回 (C, H, W)。bbox0 是第 0 层坐标 (x, y, w, h)。"""
        fast = self._fast_level(level)
        a = None if fast is not None else self._arr(level)
        ds = self.downsample(level)
        H, W = self.level_shapes[level]
        if bbox0 is None:
            ys, xs = slice(0, H), slice(0, W)
        else:
            x, y, w, h = bbox0
            x0, y0 = max(0, int(x / ds)), max(0, int(y / ds))
            x1, y1 = min(W, int(np.ceil((x + w) / ds))), min(H, int(np.ceil((y + h) / ds)))
            ys, xs = slice(y0, y1), slice(x0, x1)
        chs = channels if channels is not None else list(range(len(self.channels)))
        if fast is not None:
            if self.axes == "YX":
                chs = [0]
            return fast.read(list(chs), ys.start, ys.stop, xs.start, xs.stop)
        if a.ndim == 2:
            return np.asarray(a[ys, xs])[None]
        out = np.stack([np.asarray(a[c, ys, xs]) for c in chs])
        return out

    def close(self):
        if self._cache is not None:
            self._cache.clear()
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        self._fast.clear()
        if self._tif is not None:
            self._tif.close()
        if self._kf is not None:
            self._kf.close()


def _sharpness(stack: np.ndarray) -> np.ndarray:
    """每一层每个像素附近的清晰程度（拉普拉斯能量，局部平滑）。stack: (Z, H, W)。"""
    from scipy import ndimage as ndi
    out = np.empty(stack.shape, np.float32)
    for k, im in enumerate(stack.astype(np.float32)):
        lap = ndi.laplace(ndi.gaussian_filter(im, 1.0))
        out[k] = ndi.uniform_filter(lap * lap, 9)
    return out


def project_z(stack: np.ndarray, mode: str) -> np.ndarray:
    """把 (Z, ...) 合成为一层。"""
    if mode.startswith("z") and mode[1:].isdigit():
        return stack[min(int(mode[1:]) - 1, stack.shape[0] - 1)]
    if mode == "edf" and stack.ndim == 3 and min(stack.shape[1:]) >= 8:
        idx = np.argmax(_sharpness(stack), axis=0)
        from scipy import ndimage as ndi
        idx = np.rint(ndi.median_filter(idx, 5)).astype(int)
        return np.take_along_axis(stack, idx[None], 0)[0]
    if mode == "mean":
        return stack.mean(axis=0).astype(stack.dtype)
    return stack.max(axis=0)


class ZProjected:
    """把带 Z 轴的 zarr 数组包装成 (C, H, W) 或 (H, W)，读取时即时合成各层（最大投影 / 景深合成 / 指定层）。"""

    def __init__(self, arr, axes: str, mode: str = "max"):
        self.arr, self.axes, self.mode = arr, axes, mode
        self.zi = axes.index("Z")
        keep = [i for i, a in enumerate(axes) if a != "Z"]
        self.shape = tuple(arr.shape[i] for i in keep)
        self.dtype = arr.dtype
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = list(key) + [slice(None)] * (self.ndim - len(key))
        full = list(key)
        full.insert(self.zi, slice(None))
        a = np.asarray(self.arr[tuple(full)])
        # Z 轴在结果中的位置 = Z 之前非整数下标的个数
        pos = sum(1 for k in full[: self.zi] if not isinstance(k, (int, np.integer)))
        a = np.moveaxis(a, pos, 0)
        if self.mode == "edf" and a.ndim == 4:  # (Z, C, H, W)：每个通道用同一张清晰度图
            idx = np.argmax(_sharpness(a.max(axis=1)), axis=0)
            return np.take_along_axis(a, idx[None, None].repeat(a.shape[1], 1), 0)[0]
        return project_z(a, self.mode)

    def __array__(self, dtype=None, copy=None):
        a = self[tuple(slice(None) for _ in range(self.ndim))]
        return a.astype(dtype) if dtype is not None else a

    def __len__(self):
        return self.shape[0]


def _ome_color(v) -> tuple[int, int, int] | None:
    """OME 的 Color 属性：有符号 32 位整数 RGBA。"""
    try:
        x = int(v) & 0xFFFFFFFF
        return ((x >> 24) & 255, (x >> 16) & 255, (x >> 8) & 255)
    except (TypeError, ValueError):
        return None


def _parse_qptiff_page(desc: str):
    try:
        txt = desc.encode("utf-8", "ignore").decode("utf-8")
        txt = re.sub(r"<\?xml[^>]*\?>", "", txt)
        root = ET.fromstring(txt)
        name = root.findtext("Name")
        color = root.findtext("Color")
        itype = root.findtext("ImageType")
        rgb = tuple(int(v) for v in color.split(",")) if color else None
        return itype, name, rgb
    except Exception:
        return None, None, None


# ---------------------------------------------------------------- 通道染色判断（v0.8.3）
# 先看通道名里的染料或波长，文件里存的显示颜色只作最后参考（有的显微镜把 Far Red 存成黄色）。
ROLE_COLORS = {"DAPI": (0, 0, 255), "Green": (0, 255, 0), "Red": (255, 0, 0), "Far red": (255, 0, 255)}
_ROLE_WORDS = [   # 按顺序匹配（长词在前，避免“FAR RED”被当成“RED”）
    ("DAPI", ("DAPI", "HOECHST", "BV421", "AF405", "ALEXA 405", "ALEXA FLUOR 405", "BFP", "蓝")),
    ("Far red", ("FAR RED", "FAR-RED", "FARRED", "FAR_RED", "CY5", "CY 5", "APC", "ALEXA 647", "DY647",
                 "远红", "CY7", "IRDYE")),
    ("Green", ("FITC", "EGFP", "GFP", "YFP", "CY2", "OPAL 520", "ALEXA 488", "GREEN", "绿")),
    ("Red", ("TRITC", "TXRED", "TEXAS RED", "TEXASRED", "CY3", "CY 3", "MCHERRY", "TDTOMATO", "RFP", "PE",
             "RHODAMINE", "RED", "红")),
]


def _wavelength_role(name: str) -> str | None:
    """通道名里的三位数波长（染料名如 AF488、Alexa 594，或滤光片 / 发射波长如 520、620、670）。"""
    for m in re.finditer(r"(?<!\d)(\d{3})(?!\d)", name):
        wl = int(m.group(1))
        if 380 <= wl < 480:
            return "DAPI"
        if 480 <= wl < 560:
            return "Green"
        if 560 <= wl < 640:
            return "Red"
        if 640 <= wl <= 800:
            return "Far red"
    return None


def _color_role(rgb) -> str | None:
    """最后参考：文件里存的显示颜色。只认纯蓝 / 纯绿 / 纯红 / 品红，其他颜色（如黄色）不猜。"""
    if rgb is None:
        return None
    rgb = tuple(int(v) for v in rgb)
    for role, col in ROLE_COLORS.items():
        if rgb == col:
            return role
    return None


def guess_role(name: str, file_color=None) -> str | None:
    """判断通道是 DAPI / Green / Red / Far red：先看名字里的染料词，再看波长，最后才看文件里的显示颜色。
    都判断不出时返回 None（保留扫描仪写的名字，在“② 脑片拆分”里手动选）。"""
    n = " ".join(str(name or "").upper().replace("_", " ").split())
    for role, words in _ROLE_WORDS:
        for w in words:
            if w in ("PE", "APC"):   # 短词只匹配整个词
                if re.search(rf"(?<![A-Z]){w}(?![A-Z])", n):
                    return role
            elif w in n:
                return role
    return _wavelength_role(n) or _color_role(file_color)


def _guess_color(name: str, idx: int):
    n = name.upper()
    if "DAPI" in n or "HOECHST" in n:
        return DEFAULT_COLORS["DAPI"]
    if any(k in n for k in ("FITC", "GFP", "488", "520", "GREEN", "CY2")):
        return DEFAULT_COLORS["GREEN"]
    if any(k in n for k in ("647", "650", "660", "670", "680", "690", "700", "CY5", "FAR")):
        return DEFAULT_COLORS["FAR RED"]
    if any(k in n for k in ("TRITC", "CY3", "555", "568", "594", "620", "RED")):
        return DEFAULT_COLORS["RED"]
    return [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)][idx % 4]


SUPPORTED_AXES = ("CYX", "YX", "ZCYX", "CZYX", "ZYX")
Z_MODE = "max"   # 当前项目的 z-stack 合成方式（打开项目时由 Project 设置）
Z_MODE_NAMES = {"max": "最大投影（推荐，所有层中最亮的值）", "edf": "景深合成（每处取最清晰的一层）",
                "mean": "平均投影"}


def open_slide(path: str | Path, z_mode: str | None = None, *, scene_id: int | None = None,
               vsi_channels: list[str] | None = None) -> Slide:
    """打开图片。z-stack（OME-TIFF 等含 Z 轴的文件）读取时按 z_mode 合成为一层。"""
    path = Path(path)
    z_mode = z_mode or Z_MODE
    if not path.exists():
        raise FileNotFoundError(f"找不到图片文件：{path}（可能被移动、改名，或移动硬盘没有插上）")
    if scene_id is not None and path.suffix.lower() != ".vsi":
        raise ValueError("带扫描区编号的项目图片需要对应的 .vsi 原文件")
    if path.suffix.lower() == ".kfbf":
        return _open_kfbf(path, z_mode)
    if path.suffix.lower() == ".vsi":
        return _open_vsi(path, z_mode, scene_id, vsi_channels)
    if path.suffix.lower() == ".ets":
        raise ValueError("请选择 .vsi 文件，不要直接选 .ets。（.ets 是 .vsi 配套的像素文件，放在“_名字_”文件夹里。）")
    tif = tifffile.TiffFile(path)
    s = tif.series[0]
    axes = s.axes.replace("Q", "Z").replace("I", "Z")
    shape = s.shape
    if axes == "YXS" or axes.endswith("S"):
        tif.close()
        raise ValueError("这是一张 RGB 彩色图，不是多通道荧光图，请导出为多通道 TIFF / qptiff。")
    if axes not in SUPPORTED_AXES:
        tif.close()
        raise ValueError(f"暂不支持的图像轴顺序：{axes}（{path.name}）。支持：多通道单层、多通道 z-stack。")
    n_ch = shape[axes.index("C")] if "C" in axes else 1
    n_z = shape[axes.index("Z")] if "Z" in axes else 1
    level_shapes = [tuple(lv.shape[-2:]) for lv in s.levels]

    # 通道信息
    channels: list[ChannelInfo] = []
    names = []
    for p in tif.pages[: n_ch * 2]:
        itype, name, rgb = _parse_qptiff_page(getattr(p, "description", "") or "")
        if itype == "FullResolution" and name:
            names.append((name, rgb))
    if not names and tif.is_ome:
        try:
            pixels = ET.fromstring(tif.ome_metadata or "").find("./{*}Image/{*}Pixels")
        except ET.ParseError:
            pixels = None
        if pixels is not None:
            names = [(ch.get("Name", ""), _ome_color(ch.get("Color")))
                     for ch in pixels.findall("{*}Channel")]
    for i in range(n_ch):
        if i < len(names):
            nm, rgb = names[i]
        else:
            nm, rgb = "", None
        nm = nm or f"通道{i + 1}"
        if rgb is not None and tuple(rgb) == (255, 255, 255) and tif.is_ome:
            rgb = None        # OME 默认的白色（-1）不是显微镜设置的颜色
        channels.append(ChannelInfo(i, nm, rgb or _guess_color(nm, i), file_color=tuple(rgb) if rgb else None))

    # 像素尺寸
    px = None
    p0 = tif.pages[0]
    try:
        xr = p0.tags["XResolution"].value
        unit = p0.tags["ResolutionUnit"].value
        res = xr[0] / xr[1]
        if res > 0:
            if int(unit) == 3:  # 厘米
                px = 1e4 / res
            elif int(unit) == 2:  # 英寸
                px = 25400 / res
    except Exception:
        pass
    if tif.is_ome:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata or "")
        if m:
            px = float(m.group(1))
    if not px or px > 50:
        px = 0.25  # 无法读取时使用默认值，界面中可以修改

    slide = Slide(path=path, channels=channels, pixel_size_um=float(px),
                  level_shapes=level_shapes, _tif=tif, axes=axes, n_z=n_z, z_mode=z_mode)
    # 部分扫描仪写出的最粗几层是空白图（全为同一数值），去掉这些层，避免显示成黑屏
    while slide.n_levels > 1 and max(slide.level_shapes[-1]) <= 4096:
        a = slide.read(slide.n_levels - 1)
        if int(a.max()) - int(a.min()) > 2:
            break
        slide.level_shapes.pop()
    return slide


def _open_kfbf(path: Path, z_mode: str) -> Slide:
    """江丰生物（KFBIO）扫描仪的荧光原始文件 .kfbf（v0.8.2 起直接读取，不用先转格式）。
    每个通道是 8 位灰度 JPEG 图块，自带金字塔（20x、10x、5x …）；目前见过的样本都是单层扫描。"""
    from .kfbf import KfbfFile
    try:
        kf = KfbfFile(path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"无法读取江丰 .kfbf 文件（{path.name}）：{e}。"
                         "如果是很旧或很新的扫描软件导出的，请用江丰的软件另存为 TIFF 再导入。") from e
    channels = [ChannelInfo(i, nm, _guess_color(nm, i)) for i, nm in enumerate(kf.channel_names)]
    slide = Slide(path=path, channels=channels, pixel_size_um=float(kf.pixel_size_um),
                  level_shapes=[tuple(lv.shape) for lv in kf.levels], axes="CYX", n_z=1, z_mode=z_mode, _kf=kf)
    # 只存了组织图块的最粗几层可能全是 0，和 TIFF 一样去掉，避免显示成黑屏
    while slide.n_levels > 1 and max(slide.level_shapes[-1]) <= 4096:
        a = slide.read(slide.n_levels - 1)
        if int(a.max()) - int(a.min()) > 2:
            break
        slide.level_shapes.pop()
    return slide


def _open_vsi(path: Path, z_mode: str, scene_id: int | None = None, vsi_channels=None) -> Slide:
    """奥林巴斯 / Evident（cellSens、VS120、VS200）的 .vsi（v0.8.4 起直接读取，不需要 Java）。
    像素在同名的“_名字_”文件夹的 .ets 里；z-stack 读取时按 z_mode 合成为一层。"""
    from .vsi import VsiFile, VsiReadError
    try:
        vf = VsiFile(path, z_mode, scene_id=scene_id, channel_order=vsi_channels)
    except VsiReadError as e:
        raise ValueError(f"无法读取奥林巴斯 .vsi 文件（{path.name}）：{e}") from e
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"无法读取奥林巴斯 .vsi 文件（{path.name}）：{e}。"
                         "可以在 cellSens / OlyVIA 里另存为 OME-TIFF 再导入。") from e
    channels = []
    for i, nm in enumerate(vf.channel_names):
        em = vf.emission_nm[i]
        if em and guess_role(nm) is None:      # 名字里看不出染料（如“通道 1”）时，把发射波长写进名字，便于自动识别
            nm = f"{nm} {int(round(em))}nm"
        channels.append(ChannelInfo(i, nm, ROLE_COLORS.get(guess_role(nm)) or _guess_color(nm, i)))
    slide = Slide(path=path, channels=channels, pixel_size_um=float(vf.pixel_size_um or 0.25),
                  level_shapes=[tuple(lv.shape) for lv in vf.levels], axes="CYX", n_z=vf.n_z, z_mode=z_mode,
                  _kf=vf, notes=list(vf.notes), scene_id=vf.scene_id, vsi_channels=tuple(vf.channel_ids))
    try:
        while slide.n_levels > 1 and max(slide.level_shapes[-1]) <= 4096:
            a = slide.read(slide.n_levels - 1)
            if int(a.max()) - int(a.min()) > 2:
                break
            slide.level_shapes.pop()
    except Exception as exc:
        slide.close()
        raise ValueError(f"无法读取 {path.name} 扫描区 stack{vf.scene_id} 的缩略图：{exc}") from exc
    if not vf.pixel_size_um:
        slide.notes.append("文件里没有读到像素尺寸，暂按 0.25 µm/像素，请在“① 项目与图片”里核对。")
    return slide
