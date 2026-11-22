"""项目文件：一个文件夹保存所有图片的设置和结果，可随时关闭、重新打开。"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import pandas as pd

from .coloc import ColocParams
from .detection import ChannelParams, DetectionSettings
from .sections import Section

PROJECT_FILE = "cfos_project.json"
STANDARD_CHANNELS = ["DAPI", "Green", "Red", "Far red"]
# 脑片拆分步骤里每个通道的下拉框（v0.8.2）：显示文字 → 保存的通道名
DYE_LABELS = {"DAPI": "DAPI（AF405）", "Green": "Green（AF488）", "Red": "Red（AF594）", "Far red": "Far Red（AF647）"}
UNUSED_LABEL = "不使用"


def unused_name(c: int) -> str:
    return f"未用通道{c + 1}"


@dataclass
class DisplayChannel:
    visible: bool = True
    low: float = 0.0
    high: float = 255.0
    gamma: float = 1.0
    color: list | None = None    # 显示颜色 [R, G, B]（v0.8.3）；None = 按通道染色的默认颜色（v0.8.2 以前的项目）


def _display_dict(v: DisplayChannel) -> dict:
    """保存进项目文件的显示设置（不含颜色，颜色另存在 display_colors）。"""
    d = asdict(v)
    d.pop("color", None)
    return d


# “③ 亮度 / 对比度”里的常用显示颜色（v0.8.3）
COLOR_PRESETS = {"蓝": (0, 0, 255), "绿": (0, 255, 0), "红": (255, 0, 0), "品红": (255, 0, 255),
                 "黄": (255, 255, 0), "青": (0, 255, 255), "白": (255, 255, 255)}


def _check_czi_channels(expected: list[str] | tuple[str, ...], actual: list[str]):
    def identity(name):
        name = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
        if re.fullmatch(r"(?:(?:channel|ch|c|通道)\s*[:#_-]?\s*)?\d+", name):
            return ""
        return name

    if len(expected) == len(actual):
        for before, after in zip(map(identity, expected), map(identity, actual)):
            if before and after and before != after:
                break
        else:
            return
    raise ValueError(f"CZI 仪器通道的名称或顺序不匹配。项目记录为 {' / '.join(expected)}，"
                     f"当前文件为 {' / '.join(actual)}。请在 ZEN 统一通道顺序和名称，或使用独立项目。")


@dataclass
class ImageEntry:
    path: str
    sections: list[Section] = field(default_factory=list)
    status: str = "未处理"
    exclusions: list = field(default_factory=list)  # 排除区域：多边形列表，每个为 [(x, y), ...]（第 0 层像素）
    rois: list = field(default_factory=list)        # 自定义分析区域：[{"name": 名称, "polygon": [(x, y), ...]}]
    figure_rois: list = field(default_factory=list)  # 截图框（⑨ 截图制图）：[{"name", "x", "y", "w", "h"}]（第 0 层像素）
    error: str = ""                                  # 上次处理出错的原因（批量处理时记录）

    rel: str | None = None                           # 相对项目文件夹的路径（项目整体移动后自动找回）
    size: int | None = None                          # 文件大小（重新定位时确认是同一个文件）

    storage_key: str = ""
    review_settings: dict = field(default_factory=dict)
    field_reviews: list = field(default_factory=list)
    scene_id: int | None = None        # VSI 的 stack 编号或 CZI 的 S 维编号
    scene_name: str = ""
    vsi_channels: list[str] = field(default_factory=list)  # ordered scanner labels, independent of user-assigned dyes
    czi_channels: list[str] = field(default_factory=list)
    czi_time_index: int | None = None
    czi_time_count: int | None = None
    display_overrides: dict = field(default_factory=dict)
    section_display: dict = field(default_factory=dict)  # section name → channel → changed display fields

    def __post_init__(self):
        if self.storage_key and (self.storage_key in (".", "..") or
                                 any(c in self.storage_key for c in "/\\:\x00")):
            raise ValueError("项目图片标识无效")

    def to_dict(self):
        return {"path": self.path, "scene_id": self.scene_id, "scene_name": self.scene_name,
                **({"vsi_channels": self.vsi_channels} if self.vsi_channels else {}),
                **({"czi_channels": self.czi_channels} if self.czi_channels else {}),
                **({"display_overrides": self.display_overrides} if self.display_overrides else {}),
                **({"section_display": self.section_display} if self.section_display else {}),
                **({"czi_time_index": self.czi_time_index, "czi_time_count": self.czi_time_count}
                   if self.czi_time_index is not None else {}),
                "review_settings": self.review_settings, "field_reviews": self.field_reviews, "storage_key": self.key, "rel": self.rel, "size": self.size, "status": self.status, "error": self.error,
                "sections": [s.to_dict() for s in self.sections],
                "exclusions": [[list(map(float, p)) for p in poly] for poly in self.exclusions],
                "rois": [{"name": r["name"], "polygon": [list(map(float, p)) for p in r["polygon"]]} for r in self.rois],
                "figure_rois": [{"name": r["name"], "x": float(r["x"]), "y": float(r["y"]),
                                 "w": float(r["w"]), "h": float(r["h"])} for r in self.figure_rois]}

    @staticmethod
    def from_dict(d):
        return ImageEntry(path=d["path"], scene_id=d.get("scene_id"), scene_name=d.get("scene_name", ""),
                          vsi_channels=list(d.get("vsi_channels") or []),
                          czi_channels=list(d.get("czi_channels") or []),
                          display_overrides=d.get("display_overrides") or {},
                          section_display=d.get("section_display") or {},
                          czi_time_index=d.get("czi_time_index"), czi_time_count=d.get("czi_time_count"),
                          review_settings=d.get("review_settings", {}),
                          field_reviews=d.get("field_reviews", []), storage_key=d.get("storage_key", ""), rel=d.get("rel"), size=d.get("size"),
                          status=d.get("status", "未处理"), error=d.get("error", ""),
                          sections=[Section.from_dict(s) for s in d.get("sections", [])],
                          exclusions=[[tuple(p) for p in poly] for poly in d.get("exclusions", [])],
                          rois=[{"name": r["name"], "polygon": [tuple(p) for p in r["polygon"]]} for r in d.get("rois", [])],
                          figure_rois=[dict(r) for r in d.get("figure_rois", [])])

    @property
    def key(self) -> str:
        return self.storage_key or Path(self.path).stem

    @property
    def display_name(self) -> str:
        from .relocate import file_name
        name = file_name(self.path)
        if self.scene_id is None:
            return name
        return f"{name} · {self.scene_name or self.scene_code}"

    @property
    def scene_code(self) -> str:
        return f"S{self.scene_id}" if Path(self.path).suffix.lower() == ".czi" else f"stack{self.scene_id}"

    @property
    def scene_description(self) -> str:
        code = self.scene_code
        if not self.scene_name:
            return code
        if self.scene_name == code or self.scene_name.endswith(f"({code})"):
            return self.scene_name
        return f"{self.scene_name} ({code})"

    @property
    def source_id(self):
        return self.path if self.scene_id is None else (self.path, self.scene_id)

    def open_slide(self, z_mode: str | None = None):
        from .slide_io import open_slide
        if self.vsi_channels:
            return open_slide(self.path, z_mode=z_mode, scene_id=self.scene_id, vsi_channels=self.vsi_channels)
        slide = open_slide(self.path, z_mode=z_mode, scene_id=self.scene_id)
        if self.czi_channels:
            try:
                _check_czi_channels(self.czi_channels, [c.name for c in slide.channels])
            except ValueError:
                slide.close()
                raise
        return slide


def processed_section_names(image: ImageEntry, cells: pd.DataFrame | None) -> set[str]:
    """旧版整图结果没有覆盖记录，当时一次处理全部已启用脑片。"""
    if cells is None:
        return set()
    enabled = {s.name for s in image.sections if s.enabled}
    return enabled.intersection(cells.attrs.get("processed_sections", enabled))


def section_result_states(image: ImageEntry, cells: pd.DataFrame | None) -> dict[str, str]:
    """根据分析覆盖范围和筛选后的细胞表列出每张启用脑片的状态。"""
    processed = processed_section_names(image, cells)
    positive = set(cells["section"].unique()) if cells is not None and len(cells) else set()
    states = {}
    for section in image.sections:
        if section.enabled:
            state = "未分析"
            if section.name in processed:
                state = "已分析（有细胞）" if section.name in positive else "已分析（零细胞）"
            states[section.name] = state
    return states


DEFAULT_EXPORT_HEMISPHERE = "仅左右合并"
EXPORT_HEMISPHERES = {
    DEFAULT_EXPORT_HEMISPHERE: ("左右合计",),
    "仅左半脑": ("左",),
    "仅右半脑": ("右",),
    "左右分别导出": ("左", "右"),
    "左右分别导出并附合计": None,
}


@dataclass
class Project:
    folder: Path
    images: list[ImageEntry] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)  # 实验中各通道的名称（DAPI / Green / ...）
    display: dict[int, DisplayChannel] = field(default_factory=dict)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    coloc: ColocParams = field(default_factory=ColocParams)
    thickness: str = "thin"            # 切片厚度预设：thin（≤20 µm）/ thick（30–50 µm）
    z_mode: str = "max"                # z-stack 合成方式，见 slide_io.Z_MODE_NAMES
    export_skip_layers: bool = False   # 导出按脑区表时不列出皮层各层（layer 1、2/3…）
    export_max_depth: int = 0          # 导出按脑区表时最多列到第几级（0 = 不限）；只在“按图谱层级”时用
    export_region_set: str = ""        # 导出哪些脑区：见 core/region_sets.py（空 = 旧项目，按上面两项）
    export_hemisphere: str = DEFAULT_EXPORT_HEMISPHERE
    export_intensity: bool = False     # 导出时统计脑区平均荧光强度（要读原图，较慢；v0.8.2 起默认不算）
    samples: dict = field(default_factory=dict)   # 实验分组：{"图片/脑片": {"animal": 动物编号, "group": 组别}}
    stats: dict = field(default_factory=dict)     # 组间统计参数（stats.StatsParams 的字典）
    network: dict = field(default_factory=dict)   # 功能网络分析参数（network.NetParams 的字典）
    relocated: int = 0                             # 打开项目时自动找回的图片数（不保存）
    grading: dict = field(default_factory=dict)   # 强度分级：{通道名: [弱/中分界, 中/强分界]}（倍背景）
    figure: dict = field(default_factory=dict)    # 截图制图设置（figure.FigureSettings 的字典）
    accuracy: list = field(default_factory=list)  # 识别准确率检查记录（accuracy.check 的结果）
    behavior: dict = field(default_factory=dict)  # 每只动物的行为分数：{动物编号: 数值}（v0.8.1）
    behavior_params: dict = field(default_factory=dict)  # 脑区与行为相关的参数（behavior.BehaviorParams 的字典）
    unused_channels: list = field(default_factory=list)  # 不使用的通道（不识别、不显示、不参与拆分和串色校正，导出表里不列）（v0.8.2）
    section_channels: list = field(default_factory=list)  # 找组织轮廓、拆分脑片用哪些通道（空 = 所有在用的通道）（v0.8.2）
    unmix: dict = field(default_factory=dict)  # 识别前去除通道串色的设置（unmix.UnmixSettings 的字典）（v0.8.2）

    reserved_keys: list = field(default_factory=list)
    export_region_ids: list[int] = field(default_factory=list)  # Excel 自选的 Allen 结构 ID

    def __post_init__(self):
        from . import slide_io
        slide_io.Z_MODE = self.z_mode

    def import_image(self, path):
        """兼容单条返回值，同时加入该文件的所有可读扫描区。"""
        entries, notes = self.import_images(path)
        if len(entries) > 1 and Path(path).suffix.lower() == ".vsi":
            from .vsi import list_vsi_scenes
            by_id = {entry.scene_id: entry for entry in entries}
            for scene in list_vsi_scenes(path):
                if scene.scene_id in by_id:
                    return by_id[scene.scene_id], notes
        return entries[0], notes

    def import_images(self, path):
        """同一原文件可有多个扫描区，各自保存坐标、通道设置来源和分析结果。"""
        from collections import Counter
        from contextlib import closing
        from uuid import uuid4
        path = Path(path).resolve()
        existing = {im.scene_id: im for im in self.images if Path(im.path).resolve() == path}
        notes = []
        legacy = None
        channel_order = []
        anchor = None
        suffix = path.suffix.lower()
        prefix = "S" if suffix == ".czi" else "stack"
        if suffix in (".vsi", ".czi"):
            if suffix == ".vsi":
                from .vsi import list_vsi_scenes as list_scenes
            else:
                from .czi import list_czi_scenes as list_scenes
            scenes = list_scenes(path, notes=notes)
            name_counts = Counter(s.name for s in scenes)
            choices = []
            for s in sorted(scenes, key=lambda s: s.scene_id):
                code = f"{prefix}{s.scene_id}"
                name = s.name or code
                if s.name and name_counts[s.name] > 1:
                    name = f"{s.name} ({code})"
                choices.append((s.scene_id, name))
        else:
            choices = [(None, "")]
        if suffix == ".vsi":
            legacy = existing.pop(None, None)
            if legacy is not None:
                with closing(legacy.open_slide(self.z_mode)) as old_slide:
                    default_id = old_slide.scene_id
                    channel_order = list(old_slide.vsi_channels)
                existing[default_id] = legacy
                anchor = legacy
            elif existing:
                anchor = next(iter(existing.values()))
                channel_order = list(anchor.vsi_channels)
                if not channel_order:
                    with closing(anchor.open_slide(self.z_mode)) as original:
                        channel_order = list(original.vsi_channels)

        used = {key.casefold() for key in self.reserved_keys}
        used.update(im.key.casefold() for im in self.images)
        entries, added = [], []
        channels = None
        channel_count = len(self.channel_names)
        czi_orders = dict.fromkeys(tuple(im.czi_channels) for im in self.images if im.czi_channels) if suffix == ".czi" else {}
        for scene_id, name in choices:
            if scene_id in existing:
                entries.append(existing[scene_id])
                continue
            base = path.stem
            if len(choices) > 1:
                label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name or "").strip(" .")[:64]
                base += f"~{label or f'{prefix}{scene_id}'}"
            key = base
            while key.casefold() in used:
                key = f"{base}~{uuid4().hex[:12]}"
            entry = ImageEntry(str(path), storage_key=key, scene_id=scene_id, scene_name=name or "",
                               vsi_channels=list(channel_order))
            try:
                slide = entry.open_slide(self.z_mode)
            except (ValueError, OSError) as exc:
                if len(choices) == 1:
                    raise
                notes.append(f"跳过 {entry.display_name}：{exc}")
                continue
            try:
                if channel_count and len(slide.channels) != channel_count:
                    msg = f"{entry.display_name} 的通道数（{len(slide.channels)}）与项目中其他图片不同"
                    if len(choices) == 1:
                        raise ValueError(msg)
                    notes.append(f"跳过：{msg}。")
                    continue
                channel_count = len(slide.channels)
                if suffix == ".vsi" and not channel_order:
                    channel_order = list(slide.vsi_channels)
                    entry.vsi_channels = list(channel_order)
                if suffix == ".czi":
                    labels = [ch.name for ch in slide.channels]
                    for expected in czi_orders:
                        _check_czi_channels(expected, labels)
                    czi_orders[tuple(labels)] = None
                    entry.czi_channels = labels
                    entry.czi_time_index = slide.czi_time_index
                    entry.czi_time_count = slide.czi_time_count
                if channels is None:
                    channels = slide.channels
                notes.extend(slide.notes)
            finally:
                slide.close()
            entries.append(entry)
            added.append(entry)
            used.add(key.casefold())
        if not entries:
            raise ValueError("；".join(notes) or "没有可导入的扫描区")
        if channels is not None:
            self.ensure_channels(channels)
        if legacy is not None:
            legacy.scene_id, legacy.scene_name = default_id, dict(choices)[default_id]
        if anchor is not None and not anchor.vsi_channels:
            anchor.vsi_channels = list(channel_order)
        self.images.extend(added)
        self.reserved_keys.extend(im.key for im in added)
        return entries, list(dict.fromkeys(notes))

    # ---------- 读写 ----------
    @property
    def results_dir(self) -> Path:
        d = self.folder / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        from .relocate import rel_path, file_size
        for im in self.images:
            if Path(im.path).exists():  # 只在能找到原图时更新，避免把找不到的记录覆盖掉
                im.rel = rel_path(im.path, self.folder)
                im.size = file_size(im.path) or im.size
        d = {
            "version": 1, "reserved_keys": self.reserved_keys,
            "images": [im.to_dict() for im in self.images],
            "channel_names": self.channel_names,
            "display": {str(k): _display_dict(v) for k, v in self.display.items()},
            # 显示颜色单独存（v0.8.3）：放进 display 里的话，“恢复到安装时的版本”后旧版本软件会打不开项目
            "display_colors": {str(k): list(v.color) for k, v in self.display.items() if v.color},
            "detection": {
                "pixel_um": self.detection.pixel_um,
                "tile": self.detection.tile,
                "overlap": self.detection.overlap,
                "channels": {str(k): v.to_dict() for k, v in self.detection.channels.items()},
            },
            "coloc": self.coloc.to_dict(),
            "thickness": self.thickness, "z_mode": self.z_mode,
            "export_skip_layers": self.export_skip_layers, "export_max_depth": self.export_max_depth,
            "export_region_set": self.export_region_set,
            "export_region_ids": self.export_region_ids,
            "export_hemisphere": self.export_hemisphere,
            "export_intensity": self.export_intensity,
            "samples": self.samples, "stats": self.stats, "grading": self.grading, "network": self.network,
            "figure": self.figure,
            "accuracy": self.accuracy,
            "behavior": self.behavior, "behavior_params": self.behavior_params,
            "unused_channels": self.unused_channels, "section_channels": self.section_channels,
            "unmix": self.unmix,
        }
        from . import slide_io
        slide_io.Z_MODE = self.z_mode
        tmp = self.folder / (PROJECT_FILE + ".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        target = self.folder / PROJECT_FILE
        if target.exists():  # 保留上一个版本，项目文件损坏时可以恢复
            try:
                import shutil
                shutil.copy2(target, self.folder / (PROJECT_FILE + ".bak"))
            except OSError:
                pass
        tmp.replace(target)

    @staticmethod
    def load(folder: str | Path) -> "Project":
        folder = Path(folder)
        try:
            d = json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            bak = folder / (PROJECT_FILE + ".bak")
            if not bak.exists():
                raise
            d = json.loads(bak.read_text(encoding="utf-8"))  # 主文件损坏（如写入时断电），用备份
        p = Project(folder=folder)
        p.images = [ImageEntry.from_dict(x) for x in d.get("images", [])]
        from collections import Counter
        from uuid import uuid5, NAMESPACE_URL
        p.reserved_keys = list(dict.fromkeys(d.get("reserved_keys", []) + [im.key for im in p.images]))
        counts = Counter(im.key.casefold() for im in p.images)
        for im in p.images:
            if counts[im.key.casefold()] > 1:
                im.storage_key = f"{Path(im.path).stem}~{uuid5(NAMESPACE_URL, im.path).hex[:12]}"
                p.reserved_keys.append(im.storage_key)
                im.status = "旧项目有同名图片，请重新识别并核对分组"
                im.error = "旧结果归属不明，已保留原文件。"
            else:
                im.storage_key = im.key
        p.channel_names = d.get("channel_names", [])
        p.display = {int(k): DisplayChannel(**{kk: vv for kk, vv in v.items() if kk in DisplayChannel.__dataclass_fields__})
                     for k, v in d.get("display", {}).items()}
        for k, col in (d.get("display_colors") or {}).items():
            p.display.setdefault(int(k), DisplayChannel()).color = [int(x) for x in col][:3]
        det = d.get("detection", {})
        p.detection = DetectionSettings(
            pixel_um=det.get("pixel_um", 0.5), tile=det.get("tile", 2048), overlap=det.get("overlap", 96),
            channels={int(k): ChannelParams.from_dict(v) for k, v in det.get("channels", {}).items()})
        p.coloc = ColocParams(**{k: v for k, v in d.get("coloc", {}).items() if k in ColocParams.__dataclass_fields__})
        p.thickness = d.get("thickness", "thin")
        p.z_mode = d.get("z_mode", "max")
        p.export_skip_layers = bool(d.get("export_skip_layers", False))
        p.export_max_depth = int(d.get("export_max_depth", 0))
        p.export_region_set = str(d.get("export_region_set", "") or "")
        region_ids = d.get("export_region_ids", [])
        if isinstance(region_ids, list):
            p.export_region_ids = list(dict.fromkeys(i for i in region_ids if type(i) is int and i > 0))
        mode = d.get("export_hemisphere")
        if isinstance(mode, str) and mode in EXPORT_HEMISPHERES:
            p.export_hemisphere = mode
        p.export_intensity = bool(d.get("export_intensity", False))
        p.samples = d.get("samples", {})
        p.stats = d.get("stats", {})
        p.grading = d.get("grading", {})
        p.network = d.get("network", {})
        p.figure = d.get("figure", {})
        p.accuracy = d.get("accuracy", [])
        p.behavior = d.get("behavior", {})
        p.behavior_params = d.get("behavior_params", {})
        p.unused_channels = [int(c) for c in d.get("unused_channels", [])]
        p.section_channels = [int(c) for c in d.get("section_channels", [])]
        # 旧项目（v0.8.2 之前）没有这一项：保持“不做串色校正”，原来的识别结果不变；新项目默认开启（见 ensure_channels）
        p.unmix = d.get("unmix", {"enabled": False})
        p.__post_init__()
        from .relocate import auto_fix
        p.relocated = auto_fix(p)  # 项目和图片一起整体搬走时，自动按相对路径找回
        if p.relocated:
            p.save()
        return p

    # ---------- 实验分组 ----------
    def sample_rows(self) -> list[dict]:
        """所有启用的脑片，带上动物编号和组别（没填的为空）。"""
        rows = []
        for im in self.images:
            for s in im.sections:
                if not s.enabled:
                    continue
                k = f"{im.key}/{s.name}"
                d = self.samples.get(k, {})
                a = d.get("animal", "")
                b = self.behavior.get(a) if a else None
                rows.append({"key": k, "图片": im.key, "脑片": s.name, "动物": a,
                             "组别": d.get("group", ""), "行为分数": "" if b is None else f"{b:g}"})
        return rows

    def set_sample(self, key: str, animal: str | None = None, group: str | None = None):
        d = self.samples.setdefault(key, {"animal": "", "group": ""})
        if animal is not None:
            d["animal"] = animal.strip()
        if group is not None:
            d["group"] = group.strip()

    def set_behavior(self, animal: str, text: str) -> bool:
        """填写一只动物的行为分数（空 = 删除）。不是数字时返回 False。"""
        animal = (animal or "").strip()
        if not animal:
            return False
        t = str(text).strip().replace("％", "").replace("%", "")
        if not t:
            self.behavior.pop(animal, None)
            return True
        try:
            v = float(t)
        except ValueError:
            return False
        if v != v:   # NaN
            self.behavior.pop(animal, None)
            return True
        self.behavior[animal] = v
        return True

    def guess_animals(self) -> int:
        """按图片名猜动物编号：没填的脑片，动物编号 = 图片名（一张玻片 = 一只动物的常见情况）。返回填写数。"""
        n = 0
        for r in self.sample_rows():
            if not r["动物"]:
                self.set_sample(r["key"], animal=r["图片"]); n += 1
        return n

    # ---------- 参数模板 ----------
    def template_dict(self) -> dict:
        """不含图片的参数（识别、共标、显示、厚度、z-stack），可以保存成模板给别的项目用。"""
        return {"kind": "cfos参数模板", "version": 1, "channel_names": self.channel_names,
                "detection": {"pixel_um": self.detection.pixel_um, "tile": self.detection.tile,
                              "overlap": self.detection.overlap,
                              "channels": {str(k): v.to_dict() for k, v in self.detection.channels.items()}},
                "coloc": self.coloc.to_dict(), "thickness": self.thickness, "z_mode": self.z_mode,
                "grading": self.grading, "unused_channels": self.unused_channels,
                "section_channels": self.section_channels,
                "unmix": {k: v for k, v in self.unmix.items() if k in ("enabled", "scale")},
                "display": {str(k): _display_dict(v) for k, v in self.display.items()},
                "display_colors": {str(k): list(v.color) for k, v in self.display.items() if v.color}}

    def save_template(self, path):
        Path(path).write_text(json.dumps(self.template_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_template(self, path) -> list[str]:
        """导入模板，返回提示信息（例如通道数不一致）。"""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("kind") != "cfos参数模板":
            raise ValueError("这不是本软件保存的参数模板文件。")
        notes = []
        names = d.get("channel_names", [])
        if self.channel_names and names and len(names) != len(self.channel_names):
            notes.append(f"模板有 {len(names)} 个通道，当前项目有 {len(self.channel_names)} 个，按通道名称对应。")
        by_name = {n: i for i, n in enumerate(names)}
        det = d.get("detection", {})
        self.detection.pixel_um = det.get("pixel_um", self.detection.pixel_um)
        chs = det.get("channels", {})
        targets = self.channel_names or names
        for i, nm in enumerate(targets):
            j = by_name.get(nm, i if i < len(names) else None)
            if j is None or str(j) not in chs:
                notes.append(f"模板中没有“{nm}”通道的参数，保持原样。")
                continue
            self.detection.channels[i] = ChannelParams.from_dict(chs[str(j)])
            if str(j) in d.get("display", {}):
                self.display[i] = DisplayChannel(**{k: v for k, v in d["display"][str(j)].items()
                                                    if k in DisplayChannel.__dataclass_fields__})
                col = (d.get("display_colors") or {}).get(str(j))
                if col:
                    self.display[i].color = [int(x) for x in col][:3]
        if not self.channel_names:
            self.channel_names = list(names)
        self.coloc = ColocParams(**{k: v for k, v in d.get("coloc", {}).items() if k in ColocParams.__dataclass_fields__})
        self.thickness = d.get("thickness", self.thickness)
        self.grading = d.get("grading", self.grading)
        self.z_mode = d.get("z_mode", self.z_mode)
        if len(names) == len(self.channel_names):
            self.unused_channels = list(d.get("unused_channels", self.unused_channels))
            self.section_channels = list(d.get("section_channels", self.section_channels))
        if "unmix" in d:   # 只带开关和微调比例；串色比例每个项目按自己的图片重新估算
            self.unmix = {**self.unmix, **d["unmix"]}
        self.save()
        return notes

    # ---------- 显示颜色（v0.8.3） ----------
    def channel_display(self, c: int, image=None, section=None) -> DisplayChannel:
        """Resolve one channel from project defaults, image and section overrides."""
        values = {}
        if image is not None:
            values.update(image.display_overrides.get(str(c), {}))
            if section is not None:
                values.update(image.section_display.get(section.name, {}).get(str(c), {}))
        base = self.display.get(c)
        return replace(base if base is not None else DisplayChannel(high=0), **values)

    def update_display(self, selection, c: int, **changes):
        """Apply only the edited fields. Whole-image edits supersede those fields on its sections."""
        if selection is None:
            self.display[c] = replace(self.display.get(c, DisplayChannel()), **changes)
            return
        key = str(c)
        for image in self.images:
            if image.key not in selection:
                continue
            names = selection[image.key]
            if names is None:
                image.display_overrides.setdefault(key, {}).update(changes)
                for name, overrides in list(image.section_display.items()):
                    channel = overrides.get(key, {})
                    for attr in changes:
                        channel.pop(attr, None)
                    if not channel:
                        overrides.pop(key, None)
                    if not overrides:
                        del image.section_display[name]
            else:
                for section in image.sections:
                    if section.name in names:
                        overrides = image.section_display.setdefault(section.name, {})
                        overrides.setdefault(key, {}).update(changes)

    def custom_color(self, c: int, image=None, section=None) -> tuple[int, int, int] | None:
        """用户选的（或沿用显微镜的）显示颜色；没有设置时返回 None，由调用处按通道名用默认颜色。"""
        d = self.channel_display(c, image, section)
        if not d.color or len(d.color) < 3:
            return None
        return tuple(int(max(0, min(255, v))) for v in d.color[:3])

    def set_color(self, c: int, rgb) -> None:
        self.display.setdefault(c, DisplayChannel()).color = None if rgb is None else [int(v) for v in rgb[:3]]
        self.save()

    # ---------- 通道（v0.8.2） ----------
    def used_channels(self) -> list[int]:
        return [c for c in range(len(self.channel_names)) if c not in self.unused_channels]

    def tissue_channels(self) -> list[int] | None:
        """找组织轮廓、拆分脑片用的通道；None = 所有通道（与 v0.8.1 以前相同）。"""
        chs = [c for c in self.section_channels if c in self.used_channels()]
        if chs:
            return chs
        return self.used_channels() if self.unused_channels else None

    def set_channel_role(self, c: int, label: str) -> str:
        """脑片拆分步骤里改某个通道的下拉框。label：DYE_LABELS 的值、标准名称或 UNUSED_LABEL。
        返回空字符串表示成功，否则返回不能这样改的原因。"""
        if label == UNUSED_LABEL:
            if c not in self.unused_channels:
                if len(self.used_channels()) <= 1:
                    return "至少要保留一个通道。"
                self.unused_channels.append(c)
            self.channel_names[c] = unused_name(c)   # 换成不会和在用通道重名的名字（导出时整列去掉）
            p = self.detection.channels.get(c)
            if p is not None:
                p.detect = False
            self.display.setdefault(c, DisplayChannel()).visible = False
            self.save()
            return ""
        name = {v: k for k, v in DYE_LABELS.items()}.get(label, label).strip()
        if not name:
            return "通道名称不能为空。"
        for i, n in enumerate(self.channel_names):
            if i != c and i not in self.unused_channels and n == name:
                return f"“{name}”已经用在第 {i + 1} 个通道上了，两个通道不能同名。"
        was_unused = c in self.unused_channels
        if was_unused:
            self.unused_channels.remove(c)
            p = self.detection.channels.get(c)
            if p is not None:
                p.detect = True
            self.display.setdefault(c, DisplayChannel()).visible = True
        self.channel_names[c] = name
        self.save()
        return ""

    def apply_thickness_preset(self, thickness: str):
        """按切片厚度设置推荐参数。厚切片：识别前减背景（去掉焦平面外的雾）并去卷积（细胞边界更清楚），
        去掉和周围差别很小的模糊细胞，
        共标距离放宽到 8 µm（厚切片中同一细胞在不同通道的中心可能错开得更多）。"""
        self.thickness = thickness
        thick = thickness == "thick"
        for i, nm in enumerate(self.channel_names):
            p = self.detection.channels.get(i)
            if p is None:
                continue
            p.bg_radius_um = 25.0 if thick else 0.0
            p.deconv = thick          # 厚切片默认识别前去卷积（亮度测量仍用原图）
            if nm == "DAPI":
                p.gamma, p.min_contrast = 0.5, 0.0
            else:
                p.min_contrast = 1.5 if thick else 0.0
        self.coloc.max_dist_um = 8.0 if thick else 6.0
        self.save()

    @staticmethod
    def is_project(folder: str | Path) -> bool:
        return (Path(folder) / PROJECT_FILE).exists()

    # ---------- 通道 ----------
    def ensure_channels(self, slide_channels):
        """第一次加入图片时，按图片里的通道初始化名称和参数。"""
        if not self.channel_names:
            self.unmix = {"enabled": True}      # 新项目默认识别前去除串色（v0.8.2）
            # v0.8.3：先看通道名里的染料 / 波长，文件里存的显示颜色只作最后参考（见 slide_io.guess_role）
            from .slide_io import guess_role
            names = []
            for ch in slide_channels:
                role = guess_role(ch.name, getattr(ch, "file_color", None))
                names.append(role if role and role not in names else ch.name)
            self.channel_names = names
            # 显示颜色默认沿用显微镜软件里的颜色（文件里存的，如 Far Red 存成黄色就显示黄色）
            for i, ch in enumerate(slide_channels):
                fc = getattr(ch, "file_color", None)
                if fc is not None:
                    self.display.setdefault(i, DisplayChannel()).color = [int(v) for v in fc]
        for i, nm in enumerate(self.channel_names):
            self.display.setdefault(i, DisplayChannel())
            if i not in self.detection.channels:
                if nm == "DAPI":
                    self.detection.channels[i] = ChannelParams(min_area_um2=10, max_area_um2=200, gamma=0.5)
                else:
                    self.detection.channels[i] = ChannelParams(min_area_um2=30, max_area_um2=600)

    # ---------- 结果 ----------
    # v0.8.1 起结果文件改用紧凑格式（见 core/contours.py），换了文件名：
    # 旧版本软件（比如“恢复到安装时的版本”之后）看不懂新格式，换名可以让它当作“还没识别”，而不是出错。
    def cells_path(self, image: ImageEntry) -> Path:
        return self.results_dir / f"{image.key}_cells.v2.pkl"

    def _old_cells_path(self, image: ImageEntry) -> Path:
        return self.results_dir / f"{image.key}_cells.pkl"

    def has_cells(self, image: ImageEntry) -> bool:
        return self.cells_path(image).exists() or self._old_cells_path(image).exists()

    @staticmethod
    def _dump(p: Path, obj):
        import pickle
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def save_cells(self, image: ImageEntry, df: pd.DataFrame):
        from .contours import pack
        from .workflow_status import save_cell_summary
        path = self.cells_path(image)
        self._dump(path, pack(df))
        save_cell_summary(path, df)
        old = self._old_cells_path(image)
        if old.exists():
            old.unlink()

    # 每张脑片的原始识别结果（未按阈值筛选），用于中断后继续、改阈值后不重新识别
    def part_path(self, image: ImageEntry, section_name: str) -> Path:
        d = self.results_dir / "parts" / image.key
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{section_name}.v2.pkl"

    def _old_part_path(self, image: ImageEntry, section_name: str) -> Path:
        return self.results_dir / "parts" / image.key / f"{section_name}.pkl"

    def save_part(self, image: ImageEntry, section, signature: str, df: pd.DataFrame):
        from .contours import pack
        self._dump(self.part_path(image, section.name),
                   {"signature": signature, "bbox0": tuple(section.bbox0), **pack(df)})
        old = self._old_part_path(image, section.name)
        if old.exists():
            old.unlink()

    def load_part(self, image: ImageEntry, section, signature: str | None = None):
        import pickle
        from .contours import normalize, unpack
        p = self.part_path(image, section.name)
        if not p.exists():
            p = self._old_part_path(image, section.name)
            if not p.exists():
                return None
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if tuple(d.get("bbox0", ())) != tuple(section.bbox0):
                return None
            if signature is not None and d.get("signature") != signature:
                return None
            return unpack(d) if d.get("format") else normalize(d["cells"])
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError,
                AttributeError, KeyError, IndexError, ImportError):
            return None

    def load_cells(self, image: ImageEntry) -> pd.DataFrame | None:
        import pickle
        from .contours import normalize, unpack
        p = self.cells_path(image)
        if p.exists():
            with open(p, "rb") as f:
                return unpack(pickle.load(f))
        p = self._old_cells_path(image)
        return normalize(pd.read_pickle(p)) if p.exists() else None
