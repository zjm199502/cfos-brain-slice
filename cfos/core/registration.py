"""Registration geometry, tissue refinement and atlas sampling for CCFv3 and STAM."""
from __future__ import annotations

import os
import threading
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage as ndi

from .atlas import Plane, sample_plane, load_atlas, edit_coords, native_mm
from .atlas_choice import CCF, STAM
from .sections import Section, tissue_mask
from .slide_io import Slide

THUMB_UM = 8.0  # 配准使用的分辨率（µm/像素）


@dataclass
class Registration:
    orientation: str = "r0"                # 'r0' 或 'r180'：缩略图相对原图的旋转
    mirror: bool = False                   # 左右翻转（只影响左右半脑的判定）
    plane: list = field(default_factory=list)   # 定位得到的 9 个数（联合校正后为校正后的值）
    plane_raw: list = field(default_factory=list)  # 联合校正前的原始定位结果（用于撤销）
    ap_shift: float = 0.0                  # 前后平移（体素，25 µm）
    ap_auto: float = 0.0                   # 自动校正的前后平移（体素，v0.8.5）；与 ap_shift 相加，滑块只显示手动部分
    tilt_dv: float = 0.0                   # 背腹方向倾斜（度）
    tilt_ml: float = 0.0                   # 左右方向倾斜（度）
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0                  # 平面内旋转（度）
    shift_x: float = 0.0                   # 平面内平移（图像宽度的比例）
    shift_y: float = 0.0
    control_points: list = field(default_factory=list)  # [[src_s, src_t, dst_s, dst_t], ...] 归一化坐标
    landmarks: list = field(default_factory=list)  # [[s, t, X, Y, Z], ...]：脑片归一化坐标处应显示的图谱体素坐标（深部贴合的约束）
    nonlinear: bool = False                # 是否使用 elastix 非线性贴合
    nl_method: str = ""                    # 非线性贴合实际用的方法（"轮廓 + DAPI 纹理" / "只用轮廓"；v0.7 前为空）
    warnings: list = field(default_factory=list)  # 自动配准后需要人工重点检查的原因（v0.8.5）；手动调整后清空
    status: str = "未配准"

    # None is the legacy CCFv3 filename convention. New results reference immutable fields.
    warp_file: str | None = None

    fit_atlas: str = CCF
    method: str = "DeepSlice"     # Historical untagged CCF results.
    bbox0: list = field(default_factory=list)
    shape_hw: list = field(default_factory=list)
    downsample: float = 1.
    initialization: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        record = asdict(self)
        if self.fit_atlas == STAM:
            plane = record.pop("plane")
            record["plane_native_mm"] = native_mm(np.reshape(plane, (3, 3))).ravel().tolist() if plane else []
        return record

    @classmethod
    def from_dict(cls, record):
        if not record:
            return cls()
        if record.get("fit_atlas") == STAM and not record.get("plane_native_mm"):
            return cls(fit_atlas=STAM, method="giRAff")
        values = {k: v for k, v in record.items() if k in cls.__dataclass_fields__}
        if record.get("plane_native_mm"):
            values["plane"] = edit_coords(np.reshape(record["plane_native_mm"], (3, 3))).ravel().tolist()
            values.update(fit_atlas=STAM, method="giRAff")
            values.setdefault("status", "STAM · giRAff")
            if values.get("nonlinear"):
                values.setdefault("nl_method", "轮廓 + DAPI 纹理")
        return cls(**values)

    @property
    def done(self):
        return bool(self.plane)


# ------------------------------------------------------------------ 缩略图和方向
def section_thumbnail(slide: Slide, section: Section, nucleus_ch: int | None):
    """返回 (缩略图 uint8, 组织掩膜, 缩略图像素对应的第 0 层像素数)。"""
    lv = slide.level_for_pixel_size(THUMB_UM)
    ds = slide.downsample(lv)
    a = slide.read(lv, section.bbox0).astype(np.float32)
    chans = [nucleus_ch] if nucleus_ch is not None else list(range(a.shape[0]))
    ims = []
    for c in chans:
        x = a[c]
        hi = np.percentile(x, 99.5)
        ims.append(np.clip(x / max(hi, 1e-6), 0, 1))
    img = (np.max(ims, axis=0) * 255).astype(np.uint8)
    mask = tissue_mask(a, slide.pixel_size_um * ds)
    return img, mask, ds


def orient(img: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "r180":
        return np.ascontiguousarray(img[::-1, ::-1])
    return img



def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else 0.0


def auto_register(slide: Slide, sections: list[Section], nucleus_ch, workdir, progress=None,
                  return_thumbs: bool = False, *, atlas=CCF):
    """Fit each section with giRAff and optionally return the thumbnails for older callers."""
    from .autoreg import fit_section
    results, thumbnails = {}, {}
    for section in sections:
        thumb = section_thumbnail(slide, section, nucleus_ch)
        results[section.name] = fit_section(slide, section, nucleus_ch, workdir, False, atlas, progress, thumb=thumb)
        if return_thumbs:
            thumbnails[section.name] = thumb[:2]
    return (results, thumbnails) if return_thumbs else results


def _rot(axis, deg):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def adjusted_plane(reg: Registration) -> Plane:
    pl = Plane.from_list(reg.plane)
    o, u, v = pl.o.copy(), pl.u.copy(), pl.v.copy()
    c = o + 0.5 * u + 0.5 * v
    n = np.cross(u, v)
    n /= np.linalg.norm(n) + 1e-12
    # 前后平移：沿平面法向（让 y 增大的方向为“向前”）
    if n[1] < 0:
        n = -n
    c = c + n * (reg.ap_shift + reg.ap_auto)
    # 倾斜：绕 u 轴（背腹倾斜）、绕 v 轴（左右倾斜）
    R = _rot(v, reg.tilt_ml) @ _rot(u, reg.tilt_dv)
    u, v = R @ u, R @ v
    # 平面内：缩放、旋转、平移（围绕中心）
    th = np.deg2rad(reg.rotation)
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1 / reg.scale_x, 1 / reg.scale_y])
    u2 = A[0, 0] * u + A[1, 0] * v
    v2 = A[0, 1] * u + A[1, 1] * v
    c = c - reg.shift_x * u - reg.shift_y * v
    o2 = c - 0.5 * u2 - 0.5 * v2
    return Plane(o2, u2, v2, reg.fit_atlas)


def tps_warp(points: list, W: int, H: int):
    """修正点 → 每个像素的位移（像素单位）。points: [[src_s, src_t, dst_s, dst_t(, radius)]]，归一化坐标
    （radius 以图宽为单位，可省略）。含义：图谱中原本在 src 处的内容被拖到了 dst，因此在 dst 处采样时要取 src 的图谱内容。
    不带 radius 的点：薄板样条（全局平滑，四周固定）。带 radius 的点：只影响 dst 周围 radius 以内，
    位移按 (1 - (d/r)^2)^2 平滑衰减到 0（边界弹性跟随，远处完全不动）。"""
    if not points:
        return None
    P = [list(map(float, q)) for q in points]
    glob = [q for q in P if len(q) < 5 or q[4] <= 0]
    local = [q for q in P if len(q) >= 5 and q[4] > 0]
    step = 4
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    g = np.zeros(gy.shape + (2,), np.float64)
    if glob:
        from scipy.interpolate import RBFInterpolator
        Q = np.asarray(glob, float)
        dst = Q[:, 2:4] * [W, H]
        disp = (Q[:, 0:2] - Q[:, 2:4]) * [W, H]
        # 四周固定点，让修正只影响局部
        border = []
        for f in np.linspace(0, 1, 6):
            border += [[f * W, 0], [f * W, H], [0, f * H], [W, f * H]]
        border = np.unique(np.round(np.array(border), 3), axis=0)
        dmin = np.min(np.linalg.norm(border[:, None, :] - dst[None, :, :], axis=2), axis=1)
        border = border[dmin > 1.0]
        X = np.vstack([dst, border])
        Y = np.vstack([disp, np.zeros_like(border)])
        rbf = RBFInterpolator(X, Y, kernel="thin_plate_spline", smoothing=1e-6)
        g += rbf(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gy.shape + (2,))
    for q in local:
        sx, sy, dx, dy = q[0] * W, q[1] * H, q[2] * W, q[3] * H
        r = max(q[4] * W, 2.0)
        d2 = ((gx - dx) ** 2 + (gy - dy) ** 2) / (r * r)
        wgt = np.clip(1.0 - d2, 0.0, None) ** 2
        g[..., 0] += wgt * (sx - dx)
        g[..., 1] += wgt * (sy - dy)
    full = [ndi.zoom(g[..., k], (H / g.shape[0], W / g.shape[1]), order=1)[:H, :W] for k in range(2)]
    return full[0], full[1]


_TPS_CACHE: dict = {}


def tps_warp_cached(points: list, W: int, H: int):
    """tps_warp 的结果按修正点缓存一份：拖滑块时修正点不变，不用每次重算（v0.8.5）。返回的数组不要原地修改。"""
    if not points:
        return None
    key = (tuple(tuple(float(v) for v in q) for q in points), W, H)
    if _TPS_CACHE.get("key") != key:
        _TPS_CACHE["key"], _TPS_CACHE["val"] = key, tps_warp(points, W, H)
    return _TPS_CACHE["val"]


def warp_path(results_dir, image_key, section_name, reg):
    folder = Path(results_dir) / ("stam_giraff" if reg.fit_atlas == STAM else "registration")
    name = reg.warp_file
    if name is None:
        path = folder / f"{image_key}_{section_name}_warp.npy"
    elif not name:
        path = folder / "no-warp"
    elif len(name) == 68 and name.endswith(".npy") and all(c in "0123456789abcdef" for c in name[:-4]):
        path = folder / name
    else:
        raise ValueError("配准位移场文件名无效。")
    return path


def save_warp(results_dir, warp, atlas=CCF):
    folder = Path(results_dir) / ("stam_giraff" if atlas == STAM else "registration")
    folder.mkdir(exist_ok=True, parents=True)
    data = np.ascontiguousarray(warp, dtype=np.float32)
    # Immutable warp names keep existing projects and backups valid if a job is interrupted.
    digest = hashlib.sha256()
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.data)
    name = digest.hexdigest() + ".npy"
    if not (folder / name).exists():
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
        try:
            with tmp.open("wb") as f:
                np.save(f, data, allow_pickle=False)
            os.replace(tmp, folder / name)
        finally:
            tmp.unlink(missing_ok=True)
    return name


def _outline_warp(reg, image, mask, pixel_um):
    from . import stam_refinement
    h, w = image.shape
    plane = adjusted_plane(reg)
    ref = sample_plane(plane, w, h, ("reference",))["reference"]
    if reg.fit_atlas == STAM:
        tissue = stam_refinement.tissue_mask(ref, pixel_um)
    else:
        tissue = sample_plane(plane, w, h)["annotation"] > 0
    return stam_refinement.refine(orient(image, reg.orientation), orient(mask, reg.orientation),
                                  ref, tissue, pixel_um)


DEEP_ATLASES = (CCF,)   # 深部纹理贴合目前只在 CCFv3 上验证过


def _load_warp(reg, results_dir, h, w):
    """The saved nonlinear field, or None when there is none or it no longer fits."""
    if not reg.nonlinear or not reg.warp_file:
        return None
    try:
        warp = np.load(warp_path(results_dir, "", "", reg), allow_pickle=False)
    except (OSError, ValueError):
        return None
    return warp if warp.shape == (2, h, w) else None


def refit_outline(reg, image, mask, results_dir, pixel_um, search_plane=False, ap_bounds=None):
    """外轮廓贴合，随后（CCFv3）用 DAPI 纹理贴合深部结构。
    search_plane=True 时先微调切面（前后位置、背腹/左右倾斜）；只在自动配准时使用，不覆盖手动调整过的切面。
    ap_bounds=(lo, hi)：前后微调允许的范围（体素，相对当前 ap_auto），由同一动物的切片顺序给出（series.position_series）。"""
    h, w = image.shape
    landmarks = list(reg.landmarks)
    if reg.fit_atlas in DEEP_ATLASES and reg.control_points:
        # 手动修正（拖动、规范点寻边）先换算成图谱坐标下的标志点，再在新的贴合里作为约束。
        old = _load_warp(reg, results_dir, h, w)
        landmarks += edit_landmarks(reg, w, h, old)
    warp, reg.metrics = _outline_warp(reg, image, mask, pixel_um)
    if reg.fit_atlas in DEEP_ATLASES:
        from . import deep_refine
        oi, om = orient(image, reg.orientation), orient(mask, reg.orientation)
        if search_plane and not reg.control_points and not landmarks:
            changed, plane_report = deep_refine.search_plane(reg, oi, om, warp, pixel_um, ap_bounds=ap_bounds)
            if changed:
                for key, value in changed.items():
                    setattr(reg, key, value)
                warp, reg.metrics = _outline_warp(reg, image, mask, pixel_um)
            reg.metrics = {**reg.metrics, "plane_search": plane_report}
        pairs = []
        if landmarks:
            px, py = plane_pixels(reg, w, h, np.asarray(landmarks, float)[:, 2:5])
            pairs = [((L[0] * w - .5, L[1] * h - .5), (ax - .5, ay - .5)) for L, ax, ay in zip(landmarks, px, py)]
        probe = Registration(**{**reg.__dict__, "control_points": []})
        deep, deep_report = deep_refine.refine(probe, oi, om, warp, pixel_um, landmarks=pairs)
        reg.metrics = {**reg.metrics, "deep": deep_report}
        if deep is not None:
            warp = deep
            if landmarks:
                reg.landmarks, reg.control_points = landmarks, []
    reg.nonlinear = warp is not None
    reg.warp_file = save_warp(results_dir, warp, reg.fit_atlas) if reg.nonlinear else ""
    deep_ok = reg.metrics.get("deep", {}).get("accepted", False)
    reg.nl_method = ("轮廓 + DAPI 纹理（含深部结构）" if deep_ok else "轮廓 + DAPI 纹理") if reg.nonlinear else ""
    reg.warnings = [] if reg.nonlinear else [reg.metrics["reason"]]


NL_GRIDS = (40.0, 60.0)   # B 样条控制点间距（缩略图像素），依次尝试：先 40，出现折叠再放宽到 60。
NL_GRID = NL_GRIDS[0]     # v0.8.5 前是 20：示例图的第 2 张脑片在 20 下发生折叠、整个纹理贴合被放弃；
                          # 40 在两张脑片上都不折叠，独立检验（NeuN 灰质/白质亮度比）也不比 20 差
NL_TEXTURE_W = 0.5        # DAPI 纹理在贴合中的权重（其余为组织轮廓）

_ITK_LOCK = threading.Lock()
_ITK = {}


def _itk():
    """加载 elastix。第一次要约 10 秒（ITK 按需加载大量模块），所以进入配准页面时会在后台线程里先加载
    （preload_elastix）；这里加锁，保证后台和正式使用同时发生时只加载一次。"""
    with _ITK_LOCK:
        if "itk" not in _ITK:
            import itk
            itk.ElastixRegistrationMethod[itk.Image[itk.F, 2], itk.Image[itk.F, 2]]
            try:    # 只用允许的核数（软件最多用一半的核）
                from .cpu import budget
                itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(budget())
            except Exception:  # noqa: BLE001
                pass
            _ITK["itk"] = itk
    return _ITK["itk"]


def preload_elastix():
    """在后台线程里提前加载 elastix；失败不报错（真正用到时再报）。"""
    def run():
        try:
            _itk()
        except Exception:  # noqa: BLE001
            pass
    if "itk" not in _ITK:
        threading.Thread(target=run, name="preload-elastix", daemon=True).start()


def _elastix(fixed, moving, fixed2=None, moving2=None, grid=NL_GRID, w2=NL_TEXTURE_W, bspline=True, iters=300):
    """bspline=False：只做仿射（用于前后位置搜索，快）。"""
    itk = _itk()
    m = itk.ElastixRegistrationMethod[itk.Image[itk.F, 2], itk.Image[itk.F, 2]].New()
    m.SetFixedImage(itk.image_view_from_array(np.ascontiguousarray(fixed)))
    m.SetMovingImage(itk.image_view_from_array(np.ascontiguousarray(moving)))
    po = itk.ParameterObject.New()
    aff = po.GetDefaultParameterMap("affine", 3)
    aff["MaximumNumberOfIterations"] = [str(iters)]
    bs = po.GetDefaultParameterMap("bspline", 3, 40.0)
    bs["MaximumNumberOfIterations"] = [str(iters)]
    bs["FinalGridSpacingInPhysicalUnits"] = [str(grid)]
    if fixed2 is not None:
        m.AddFixedImage(itk.image_view_from_array(np.ascontiguousarray(fixed2)))
        m.AddMovingImage(itk.image_view_from_array(np.ascontiguousarray(moving2)))
        for pm, w in ((aff, 0.0), (bs, w2)):    # 仿射只用轮廓；B 样条同时用轮廓和纹理
            pm["Registration"] = ["MultiMetricMultiResolutionRegistration"]
            pm["Metric"] = ["AdvancedMattesMutualInformation", "AdvancedMattesMutualInformation"]
            pm["Metric0Weight"] = [str(1.0 - w if w > 0 else 1.0)]
            pm["Metric1Weight"] = [str(w)]
            for k, v in (("FixedImagePyramid", "FixedSmoothingImagePyramid"),
                         ("MovingImagePyramid", "MovingSmoothingImagePyramid"),
                         ("Interpolator", "BSplineInterpolator"), ("ImageSampler", "RandomCoordinate")):
                pm[k] = [v, v]
    po.AddParameterMap(aff)
    if bspline:
        po.AddParameterMap(bs)
    m.SetParameterObject(po); m.SetLogToConsole(False)
    m.UpdateLargestPossibleRegion()
    return _deformation_field(itk, moving, m.GetTransformParameterObject())


def _deformation_field(itk, moving, parameters):
    """transformix 会把 deformationField.nii 写进输出目录；不指定时写进当前目录。
    从 Finder 启动的 App 当前目录是只读的“/”，此前每次非线性贴合都会因此失败，所以显式给一个临时目录。"""
    image = itk.image_view_from_array(np.ascontiguousarray(moving, dtype=np.float32))
    with tempfile.TemporaryDirectory(prefix="fos-transformix-") as tmp:
        t = itk.TransformixFilter[type(image)].New()
        t.SetMovingImage(image)
        t.SetTransformParameterObject(parameters)
        t.SetComputeDeformationField(True)
        t.SetLogToConsole(False)
        t.SetOutputDirectory(tmp)
        t.UpdateLargestPossibleRegion()
        f = np.array(itk.array_from_image(t.GetOutputDeformationField()), copy=True)  # (H, W, 2)，像素，(x, y)
    return f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)


def min_jacobian(w) -> float:
    """位移场的最小雅可比行列式；≤ 0 表示局部发生了折叠（不合理的变形）。"""
    gx = np.gradient(w[0]); gy = np.gradient(w[1])
    return float(((1 + gx[1]) * (1 + gy[0]) - gx[0] * gy[1]).min())


TEXTURE, CONTOUR = "轮廓 + DAPI 纹理", "只用轮廓"


def nonlinear_fit(reg: Registration, mask_oriented: np.ndarray, img_oriented: np.ndarray | None = None):
    """见 _nonlinear_fit；实际所用的方法记在 reg.nl_method（写进 Excel 的分析参数）。"""
    w, method = _nonlinear_fit(reg, mask_oriented, img_oriented)
    reg.nl_method = method
    return w


def _nonlinear_fit(reg: Registration, mask_oriented: np.ndarray, img_oriented: np.ndarray | None = None):
    """用 elastix 把图谱切片贴到组织上，返回 (dx, dy) 像素位移场。
    有 DAPI 缩略图时，同时用组织轮廓和 DAPI 纹理（图谱平均模板 ↔ DAPI，互信息）：胼胝体、前连合、脑室、海马等
    内部结构也会对齐，而不只是外圈。
    变形出现折叠时先把网格放宽再试（NL_GRIDS：40 → 60），都不行、或轮廓重合度明显下降时，才退回只用轮廓。"""
    from .log import log
    H, W = mask_oriented.shape
    pl = adjusted_plane(reg)
    smp = sample_plane(pl, W, H, ("annotation", "reference"))
    lab, ref = smp["annotation"], smp["reference"].astype(np.float32)
    fixed = ndi.gaussian_filter(mask_oriented.astype(np.float32), 2)
    moving = ndi.gaussian_filter((lab > 0).astype(np.float32), 2)
    w0 = d0 = None
    if img_oriented is not None and ref.max() > 0:
        f2 = ndi.gaussian_filter(img_oriented.astype(np.float32), 1) / 255.0
        m2 = ndi.gaussian_filter(ref, 1) / float(ref.max())
        for grid in NL_GRIDS:
            try:
                w = _elastix(fixed, moving, f2, m2, grid=grid)
            except Exception:  # noqa: BLE001
                import traceback
                log.warning("DAPI 纹理贴合失败，改用只用轮廓\n%s", traceback.format_exc())
                break
            mj = min_jacobian(w)
            if mj <= 0.05:
                log.info("DAPI 纹理贴合出现折叠（网格 %.0f，最小雅可比 %.2f），放宽网格再试", grid, mj)
                continue
            d2 = dice(sample_plane(pl, W, H, warp=w)["annotation"] > 0, mask_oriented)
            if d2 >= 0.95:          # 外轮廓贴得好，直接用（通常情况，不用再算一遍）
                return w, TEXTURE
            if w0 is None:
                w0 = _elastix(fixed, moving, grid=60.0)
                d0 = dice(sample_plane(pl, W, H, warp=w0)["annotation"] > 0, mask_oriented)
            if d2 >= d0 - 0.02:
                return w, TEXTURE
            log.info("DAPI 纹理贴合后轮廓重合度下降（网格 %.0f），放宽网格再试", grid)
    if w0 is None:
        w0 = _elastix(fixed, moving, grid=60.0)
    return w0, CONTOUR


# ------------------------------------------------------------------ 自动校正前后位置（v0.8.5）
def similarity(ref: np.ndarray, img: np.ndarray, mask: np.ndarray) -> float:
    """图谱平均模板与 DAPI 缩略图在组织范围内的互信息（越大越像）。"""
    a = ndi.gaussian_filter(ref.astype(np.float32), 2)
    b = ndi.gaussian_filter(img.astype(np.float32), 2)
    m = mask.astype(bool)
    if m.sum() < 100:
        return 0.0
    h = np.histogram2d(a[m], b[m], bins=48)[0]
    p = h / max(h.sum(), 1.0)
    q = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())


def atlas_points(reg, W, H, warp, x, y, edits=True):
    """Atlas voxel coordinates (3, ...) shown at oriented section corner coordinates x, y."""
    from .edge_drag import is_fine, transport
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    points = reg.control_points if edits else []
    for point in reversed(points):
        if is_fine(point):
            x, y = transport(x, y, point, W, H, inverse=True)
    legacy = [p for p in points if not is_fine(p)]
    manual = tps_warp_cached(legacy, W, H)
    for field in (manual, warp if reg.nonlinear else None):
        if field is not None:
            coords = [y - .5, x - .5]
            x = x + ndi.map_coordinates(field[0], coords, order=1, mode="nearest")
            y = y + ndi.map_coordinates(field[1], coords, order=1, mode="nearest")
    plane = adjusted_plane(reg)
    s, t = x / W, y / H
    return np.stack([plane.o[k] + s * plane.u[k] + t * plane.v[k] for k in range(3)])


def plane_pixels(reg, W, H, xyz):
    """Inverse of the affine plane: atlas voxel coordinates -> plane corner coordinates (x, y)."""
    plane = adjusted_plane(reg)
    A = np.stack([plane.u, plane.v], 1)
    st, *_ = np.linalg.lstsq(A, (np.reshape(xyz, (-1, 3)) - plane.o).T, rcond=None)
    return st[0] * W, st[1] * H


def edit_landmarks(reg, W, H, warp, spacing=8, limit=400):
    """Turn manual edits into landmarks [[section_s, section_t, X, Y, Z], ...] where they move the atlas.

    Points are sampled on a grid; only those the edits displace by more than one pixel are kept."""
    if not reg.control_points:
        return []
    ys, xs = np.mgrid[spacing / 2:H:spacing, spacing / 2:W:spacing]
    x, y = xs.ravel(), ys.ravel()
    edited = atlas_points(reg, W, H, warp, x, y)
    base = atlas_points(reg, W, H, warp, x, y, edits=False)
    bx, by = plane_pixels(reg, W, H, base.T)
    ex, ey = plane_pixels(reg, W, H, edited.T)
    moved = np.hypot(ex - bx, ey - by) > 1.
    idx = np.nonzero(moved)[0]
    if len(idx) > limit:
        idx = idx[np.linspace(0, len(idx) - 1, limit).round().astype(int)]
    return [[float(x[i] / W), float(y[i] / H), *map(float, edited[:, i])] for i in idx]


def registered_samples(reg, W, H, warp, x, y, which=("annotation", "hemispheres")):
    """Sample the original atlas at arbitrary oriented section corner coordinates."""
    from .atlas import SHAPE_ASR
    xyz = atlas_points(reg, W, H, warp, x, y)
    atlas = load_atlas(reg.fit_atlas)
    if reg.fit_atlas == STAM:
        coords = np.stack((-xyz[1], -xyz[2], xyz[0])) * .025
        result = {k: ndi.map_coordinates(getattr(atlas, k), coords / (.032 if k == "reference" else .040),
                                         order=int(k == "reference"), mode="constant") for k in which}
    else:
        coords = [size - 1 - xyz[k] for size, k in zip(SHAPE_ASR, (1, 2, 0))]
        result = {k: ndi.map_coordinates(getattr(atlas, k), coords, order=int(k == "reference"), mode="constant")
                  for k in which}
    if "hemispheres" in result and reg.mirror:
        h = result["hemispheres"]
        result["hemispheres"] = np.where(h == 1, 2, np.where(h == 2, 1, h)).astype(h.dtype)
    return result


def sample_grid(reg, W, H, warp, xs, ys, which):
    """Bound temporary coordinate memory independently of export and preview size."""
    if reg.nonlinear and np.shape(warp) != (2, H, W):
        raise ValueError("配准位移场缺失或尺寸与脑片不符，请重新配准。")
    atlas = load_atlas(reg.fit_atlas)
    out = {k: np.empty((len(ys), len(xs)), getattr(atlas, k).dtype) for k in which}
    rows = max(1, 192000 // max(1, len(xs)))
    for start in range(0, len(ys), rows):
        sl = slice(start, start + rows)
        values = registered_samples(reg, W, H, warp, xs[None, :], ys[sl, None], which)
        for k in which:
            out[k][sl] = values[k]
    return out


def label_maps(reg: Registration, W: int, H: int, warp=None):
    """在“方向已校正”的缩略图坐标下，生成脑区标签图和半脑图。"""
    if reg.nonlinear and np.shape(warp) != (2, H, W):
        raise ValueError("配准位移场缺失或尺寸与脑片不符，请重新配准。")
    from .edge_drag import is_fine
    if any(is_fine(p) for p in reg.control_points):
        r = sample_grid(reg, W, H, warp, np.arange(W) + .5, np.arange(H) + .5,
                        ("annotation", "hemispheres"))
        return r["annotation"], r["hemispheres"]
    pl = adjusted_plane(reg)
    total = None
    t = tps_warp_cached(reg.control_points, W, H)
    if t is not None:
        total = (t[0], t[1])
    if warp is not None and reg.nonlinear:
        if total is None:
            total = (warp[0], warp[1])
        else:
            gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
            px, py = gx + total[0], gy + total[1]
            ex = ndi.map_coordinates(warp[0], [py, px], order=1, mode="nearest")
            ey = ndi.map_coordinates(warp[1], [py, px], order=1, mode="nearest")
            total = (total[0] + ex, total[1] + ey)
    r = sample_plane(pl, W, H, ("annotation", "hemispheres"), warp=total)
    hemi = r["hemispheres"]
    if reg.mirror:
        hemi = np.where(hemi == 1, 2, np.where(hemi == 2, 1, hemi)).astype(hemi.dtype)
    return r["annotation"], hemi


def assign_cells(x0: np.ndarray, y0: np.ndarray, section: Section, reg: Registration, ds: float,
                 shape_hw, warp=None):
    """细胞坐标（第 0 层像素）→ (脑区 index, 半脑 1/2)。"""
    H, W = shape_hw
    lab, hemi = label_maps(reg, W, H, warp)
    return cell_labels(x0, y0, section, reg, ds, lab, hemi, warp)


def cell_labels(x0, y0, section, reg, ds, lab, hemi, warp=None, excluded=None):
    from .edge_drag import is_fine, transport
    H, W = lab.shape
    bx, by = section.bbox0[0], section.bbox0[1]
    px = (np.asarray(x0) - bx) / ds
    py = (np.asarray(y0) - by) / ds
    if reg.orientation == "r180":
        px, py = W - 1 - px, H - 1 - py
    ix = np.clip(np.round(px).astype(int), 0, W - 1)
    iy = np.clip(np.round(py).astype(int), 0, H - 1)
    regions, sides = lab[iy, ix], hemi[iy, ix]
    edits = [point for point in reg.control_points if is_fine(point)]
    if edits:
        # Legacy cell indexing used rounded thumbnail pixels. Only recompute cells
        # within new stroke supports or whose old pixel centre was affected.
        x, y = (np.asarray(x0)-bx)/ds, (np.asarray(y0)-by)/ds
        if reg.orientation == "r180":
            x, y = W-x, H-y
        moved = np.zeros(x.shape, bool)
        for p in edits:
            for cx, cy in ((x,y), (ix+.5,iy+.5)):
                sx, sy = transport(cx, cy, p, W, H, inverse=True)
                moved |= (sx != cx) | (sy != cy)
        if moved.any():
            values = registered_samples(reg,W,H,warp,x[moved],y[moved])
            regions[moved], sides[moved] = values["annotation"], values["hemispheres"]
    if excluded is not None:
        regions[excluded[iy, ix]] = 0
    return regions, sides


HEMI_NAMES = {0: "", 1: "左", 2: "右"}


def region_table():
    return load_atlas().structures
