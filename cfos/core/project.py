"""项目文件：一个文件夹保存所有图片的设置和结果，可随时关闭、重新打开。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from .coloc import ColocParams
from .detection import ChannelParams, DetectionSettings
from .sections import Section

PROJECT_FILE = "cfos_project.json"
STANDARD_CHANNELS = ["DAPI", "Green", "Red", "Far red"]


@dataclass
class DisplayChannel:
    visible: bool = True
    low: float = 0.0
    high: float = 255.0
    gamma: float = 1.0


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

    def to_dict(self):
        return {"path": self.path, "rel": self.rel, "size": self.size, "status": self.status, "error": self.error,
                "sections": [s.to_dict() for s in self.sections],
                "exclusions": [[list(map(float, p)) for p in poly] for poly in self.exclusions],
                "rois": [{"name": r["name"], "polygon": [list(map(float, p)) for p in r["polygon"]]} for r in self.rois],
                "figure_rois": [{"name": r["name"], "x": float(r["x"]), "y": float(r["y"]),
                                 "w": float(r["w"]), "h": float(r["h"])} for r in self.figure_rois]}

    @staticmethod
    def from_dict(d):
        return ImageEntry(path=d["path"], rel=d.get("rel"), size=d.get("size"),
                          status=d.get("status", "未处理"), error=d.get("error", ""),
                          sections=[Section.from_dict(s) for s in d.get("sections", [])],
                          exclusions=[[tuple(p) for p in poly] for poly in d.get("exclusions", [])],
                          rois=[{"name": r["name"], "polygon": [tuple(p) for p in r["polygon"]]} for r in d.get("rois", [])],
                          figure_rois=[dict(r) for r in d.get("figure_rois", [])])

    @property
    def key(self) -> str:
        return Path(self.path).stem


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
    export_max_depth: int = 0          # 导出按脑区表时最多列到第几级（0 = 不限）
    samples: dict = field(default_factory=dict)   # 实验分组：{"图片/脑片": {"animal": 动物编号, "group": 组别}}
    stats: dict = field(default_factory=dict)     # 组间统计参数（stats.StatsParams 的字典）
    network: dict = field(default_factory=dict)   # 功能网络分析参数（network.NetParams 的字典）
    relocated: int = 0                             # 打开项目时自动找回的图片数（不保存）
    grading: dict = field(default_factory=dict)   # 强度分级：{通道名: [弱/中分界, 中/强分界]}（倍背景）
    figure: dict = field(default_factory=dict)    # 截图制图设置（figure.FigureSettings 的字典）
    accuracy: list = field(default_factory=list)  # 识别准确率检查记录（accuracy.check 的结果）
    behavior: dict = field(default_factory=dict)  # 每只动物的行为分数：{动物编号: 数值}（v0.8.1）
    behavior_params: dict = field(default_factory=dict)  # 脑区与行为相关的参数（behavior.BehaviorParams 的字典）

    def __post_init__(self):
        from . import slide_io, detection
        slide_io.Z_MODE = self.z_mode
        detection.MODEL_ROOT = str(Path(self.folder) / "models")

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
            "version": 1,
            "images": [im.to_dict() for im in self.images],
            "channel_names": self.channel_names,
            "display": {str(k): asdict(v) for k, v in self.display.items()},
            "detection": {
                "pixel_um": self.detection.pixel_um,
                "tile": self.detection.tile,
                "overlap": self.detection.overlap,
                "channels": {str(k): v.to_dict() for k, v in self.detection.channels.items()},
            },
            "coloc": self.coloc.to_dict(),
            "thickness": self.thickness, "z_mode": self.z_mode,
            "export_skip_layers": self.export_skip_layers, "export_max_depth": self.export_max_depth,
            "samples": self.samples, "stats": self.stats, "grading": self.grading, "network": self.network,
            "figure": self.figure,
            "accuracy": self.accuracy,
            "behavior": self.behavior, "behavior_params": self.behavior_params,
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
        p.channel_names = d.get("channel_names", [])
        p.display = {int(k): DisplayChannel(**v) for k, v in d.get("display", {}).items()}
        det = d.get("detection", {})
        p.detection = DetectionSettings(
            pixel_um=det.get("pixel_um", 0.5), tile=det.get("tile", 2048), overlap=det.get("overlap", 96),
            channels={int(k): ChannelParams(**{kk: vv for kk, vv in v.items() if kk in ChannelParams.__dataclass_fields__}) for k, v in det.get("channels", {}).items()})
        p.coloc = ColocParams(**{k: v for k, v in d.get("coloc", {}).items() if k in ColocParams.__dataclass_fields__})
        p.thickness = d.get("thickness", "thin")
        p.z_mode = d.get("z_mode", "max")
        p.export_skip_layers = bool(d.get("export_skip_layers", False))
        p.export_max_depth = int(d.get("export_max_depth", 0))
        p.samples = d.get("samples", {})
        p.stats = d.get("stats", {})
        p.grading = d.get("grading", {})
        p.network = d.get("network", {})
        p.figure = d.get("figure", {})
        p.accuracy = d.get("accuracy", [])
        p.behavior = d.get("behavior", {})
        p.behavior_params = d.get("behavior_params", {})
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
                "grading": self.grading,
                "display": {str(k): asdict(v) for k, v in self.display.items()}}

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
            self.detection.channels[i] = ChannelParams(**{k: v for k, v in chs[str(j)].items() if k in ChannelParams.__dataclass_fields__})
            if str(j) in d.get("display", {}):
                self.display[i] = DisplayChannel(**d["display"][str(j)])
        if not self.channel_names:
            self.channel_names = list(names)
        self.coloc = ColocParams(**{k: v for k, v in d.get("coloc", {}).items() if k in ColocParams.__dataclass_fields__})
        self.thickness = d.get("thickness", self.thickness)
        self.grading = d.get("grading", self.grading)
        self.z_mode = d.get("z_mode", self.z_mode)
        self.save()
        return notes

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
                p.gamma, p.prob_thresh, p.min_contrast = 0.5, 0.3, 0.0
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
            names = []
            for ch in slide_channels:
                n = ch.name.upper()
                if "DAPI" in n:
                    names.append("DAPI")
                elif ch.color == (0, 255, 0):
                    names.append("Green")
                elif ch.color == (255, 0, 255):
                    names.append("Far red")
                elif ch.color == (255, 0, 0):
                    names.append("Red")
                else:
                    names.append(ch.name)
            self.channel_names = names
        for i, nm in enumerate(self.channel_names):
            self.display.setdefault(i, DisplayChannel())
            if i not in self.detection.channels:
                if nm == "DAPI":
                    self.detection.channels[i] = ChannelParams(min_area_um2=10, max_area_um2=200, gamma=0.5, prob_thresh=0.3)
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
        self._dump(self.cells_path(image), pack(df))
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
        except Exception:  # noqa: BLE001  文件损坏时当作没有
            return None
        if tuple(d.get("bbox0", ())) != tuple(section.bbox0):
            return None
        if signature is not None and d.get("signature") != signature:
            return None
        return unpack(d) if d.get("format") else normalize(d["cells"])

    def load_cells(self, image: ImageEntry) -> pd.DataFrame | None:
        import pickle
        from .contours import normalize, unpack
        p = self.cells_path(image)
        if p.exists():
            with open(p, "rb") as f:
                return unpack(pickle.load(f))
        p = self._old_cells_path(image)
        return normalize(pd.read_pickle(p)) if p.exists() else None
