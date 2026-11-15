"""脑片与 Allen CCFv3 图谱的配准。

流程：
1. 每张脑片生成一张约 8 µm/像素 的缩略图（DAPI），交给 DeepSlice 估计它在图谱中的切片平面。
   切片在玻片上可能是倒放的，所以同时试 0° 和 180° 两个方向，选与图谱轮廓重合最好的方向。
2. 自动校正前后位置（v0.8.5）：前后挪动切面，取图谱模板与 DAPI 最像的位置（refine_ap）。
3. 可选：用 elastix 做非线性贴合（组织轮廓 + DAPI 纹理），修正切片变形。
4. 用户可以手动调整：前后位置、两个倾斜角、缩放、旋转、平移，以及在图上拖动“修正点”做局部修正。
5. 最终得到一张与缩略图对齐的脑区标签图，用来给每个细胞分配脑区和左右半脑。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage as ndi

from .atlas import Plane, sample_plane, load_atlas
from .resources import resources_dir
from .sections import Section, tissue_mask
from .slide_io import Slide

THUMB_UM = 8.0  # 配准使用的分辨率（µm/像素）


@dataclass
class Registration:
    orientation: str = "r0"                # 'r0' 或 'r180'：缩略图相对原图的旋转
    mirror: bool = False                   # 左右翻转（只影响左右半脑的判定）
    plane: list = field(default_factory=list)   # DeepSlice 给出的 9 个数（联合校正后为校正后的值）
    plane_raw: list = field(default_factory=list)  # 联合校正前 DeepSlice 的原始结果（用于撤销）
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
    nonlinear: bool = False                # 是否使用 elastix 非线性贴合
    nl_method: str = ""                    # 非线性贴合实际用的方法（"轮廓 + DAPI 纹理" / "只用轮廓"；v0.7 前为空）
    warnings: list = field(default_factory=list)  # 自动配准后需要人工重点检查的原因（v0.8.5）；手动调整后清空
    status: str = "未配准"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        if not d:
            return Registration()
        return Registration(**{k: v for k, v in d.items() if k in Registration.__dataclass_fields__})

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


# ------------------------------------------------------------------ DeepSlice
_DS = None


def _deepslice():
    global _DS
    if _DS is None:
        from DeepSlice import DSModel
        from DeepSlice.metadata import metadata_loader
        from DeepSlice.neural_network import neural_network

        d = resources_dir() / "deepslice"
        orig = metadata_loader.get_data_path

        def local_path(url_path_dict, path):
            p = d / os.path.basename(url_path_dict["path"])
            return str(p) if p.exists() else orig(url_path_dict, path)

        metadata_loader.get_data_path = local_path
        import DeepSlice.main as dsmain
        dsmain.metadata_loader.get_data_path = local_path
        m = DSModel.__new__(DSModel)
        m.species = "mouse"
        m.config, m.metadata_path = metadata_loader.load_config()
        def _patched(model, weights, species="mouse"):
            _load_deepslice_weights(model, weights)
            return model
        neural_network.load_xception_weights = _patched
        # DeepSlice 的权重文件覆盖了 Xception 的全部卷积层，只有最顶上的 ImageNet 分类层（predictions，8 MB）
        # 不在里面，所以 v0.8 起只带这一层（xception_top.npz），不再带 92 MB 的完整 Xception 预训练文件；
        # 两种方式的输出完全相同（实测差值 0）。旧安装（v0.7，补丁升级）仍有完整文件时照旧用它。
        full = d / "xception_weights_tf_dim_ordering_tf_kernels.h5"
        top = d / "xception_top.npz"
        if top.exists():
            model = neural_network.initialise_network(None, None, "mouse")
            z = np.load(top)
            model.layers[0].get_layer("predictions").set_weights([z["kernel"], z["bias"]])
        else:
            model = neural_network.initialise_network(str(full), None, "mouse")
        _load_deepslice_weights(model, str(d / "Allen_Mixed_Best.h5"))
        m.model = model
        _DS = m
    return _DS


def _load_deepslice_weights(model, path):
    """按层名加载 DeepSlice 权重。Keras 会给未命名的层自动编号（conv2d_2、batch_normalization_5 …），
    同一进程里先建过其他模型（如 StarDist）时编号会错位，所以这里按出现顺序把它们换成标准名字再对应。"""
    import re
    import h5py
    with h5py.File(path, "r") as f:
        for k, nm in ((1, "dense"), (2, "dense_1"), (3, "dense_2")):
            g = f[nm][nm]
            model.layers[k].set_weights([np.array(g["kernel:0"]), np.array(g["bias:0"])])
        x = f["xception"]
        counters = {}
        for layer in model.layers[0].layers:
            if not layer.weights:
                continue
            name = layer.name
            mt = re.match(r"^(conv2d|batch_normalization)(_\d+)?$", name)
            if mt:
                base = mt.group(1)
                i = counters.get(base, 0)
                counters[base] = i + 1
                name = base if i == 0 else f"{base}_{i}"
            if name not in x:
                continue
            g = x[name]
            wn = []
            for w in layer.weights:
                short = w.name.split("/")[-1]
                short = short if short.endswith(":0") else f"{short}:0"
                wn.append(short)
            layer.set_weights([np.array(g[k]) for k in wn])


def deepslice_predict(images: list[np.ndarray], workdir) -> list[Plane]:
    """对若干张缩略图预测切片平面。

    v0.8.1 起默认用 ONNX 版本（core/deepslice_onnx.py，不需要 TensorFlow；与 TensorFlow 结果最大差
    0.0001 体素）。找不到 ONNX 文件或设置了 CFOS_NET=tf 时用原来的 DeepSlice（需要 TensorFlow）。"""
    from . import accel, deepslice_onnx as DO
    d = resources_dir() / "deepslice"
    if accel.net_backend() == "onnx" and DO.available(d):
        return [Plane.from_list(v) for v in DO.predict(images, d)]
    from .components import require_tensorflow
    require_tensorflow("脑图谱自动定位（DeepSlice）")
    return _deepslice_predict_tf(images, workdir)


def _deepslice_predict_tf(images: list[np.ndarray], workdir) -> list[Plane]:
    from PIL import Image
    import tempfile
    m = _deepslice()
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        names = []
        for i, im in enumerate(images):
            n = os.path.join(td, f"img_{i:04d}.png")
            Image.fromarray(im).save(n)
            names.append(n)
        m.predict(image_list=names, ensemble=True, section_numbers=False)
        p = m.predictions.copy()
    p["base"] = p["Filenames"].map(lambda f: os.path.basename(str(f)))
    planes = []
    for n in names:
        row = p[p["base"] == os.path.basename(n)].iloc[0]
        planes.append(Plane.from_list(row[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]].to_numpy(float)))
    return planes


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else 0.0


def auto_register(slide: Slide, sections: list[Section], nucleus_ch, workdir, progress=None,
                  return_thumbs: bool = False):
    """返回 {section.name: Registration}；return_thumbs=True 时再返回 {section.name: (缩略图, 掩膜)}，
    后面的前后位置校正和非线性贴合直接用，不用再读一遍图。"""
    thumbs = {}
    imgs = []
    keys = []
    for s in sections:
        img, mask, ds = section_thumbnail(slide, s, nucleus_ch)
        thumbs[s.name] = (img, mask)
        for o in ("r0", "r180"):
            imgs.append(orient(img, o))
            keys.append((s.name, o))
    if progress:
        progress(1, 3, "正在用 DeepSlice 估计切片位置…")
    planes = deepslice_predict(imgs, workdir)
    best = {}
    for (name, o), pl in zip(keys, planes):
        img, mask = thumbs[name]
        H, W = mask.shape
        lab = sample_plane(pl, W, H)["annotation"]
        d = dice(lab > 0, orient(mask, o))
        if name not in best or d > best[name][0]:
            best[name] = (d, o, pl)
    out = {}
    for s in sections:
        d, o, pl = best[s.name]
        r = Registration(orientation=o, plane=pl.to_list(), status=f"自动配准（轮廓重合度 {d:.2f}）")
        out[s.name] = r
    if return_thumbs:
        return out, thumbs
    return out


# ------------------------------------------------------------------ 手动调整
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
    return Plane(o2, u2, v2)


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


def warp_path(results_dir, image_key, section_name):
    d = results_dir / "registration"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_key}_{section_name}_warp.npy"


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
    field_ = itk.transformix_deformation_field(itk.image_view_from_array(np.ascontiguousarray(moving)),
                                               m.GetTransformParameterObject())
    f = itk.array_from_image(field_)  # (H, W, 2)，单位：像素，(x, y)
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
AP_RANGE_UM = 600.0     # 最多前后各挪这么多
AP_COARSE_UM = 200.0    # 粗找的步长
AP_FINE_UM = 100.0      # 在最好的位置两侧再细找一步
AP_MIN_GAIN = 1.05      # 相似度至少比原位置高 5% 才挪动（否则认为 DeepSlice 的位置已经够好）
AP_BIG_UM = 500.0       # 挪动量达到这个值时提示人工检查


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


def refine_ap(reg: Registration, mask_oriented: np.ndarray, img_oriented: np.ndarray, cancelled=None) -> dict:
    """DeepSlice 给的前后位置（Bregma）常有几百微米的误差。这里把切面前后挪动试几个位置，
    每个位置先做一次快速的仿射对齐（只用轮廓），再比较图谱模板与 DAPI 的相似度，取最像的位置，写进 reg.ap_auto。

    示例图（两张相邻脑片）实测：DeepSlice 给 +0.77 和 +0.11 mm，本方法给 +0.37 和 +0.41 mm；
    不参与配准的 NeuN 通道“灰质/白质亮度比”从 2.0 左右提高到 2.6 左右。
    返回 {"shift_um": 挪动量, "scores": {挪动量: 相似度}, "edge": 最好的位置是否在搜索范围边上}。"""
    H, W = mask_oriented.shape
    fixed = ndi.gaussian_filter(mask_oriented.astype(np.float32), 2)
    f2 = ndi.gaussian_filter(img_oriented.astype(np.float32), 1) / 255.0
    r2 = Registration.from_dict(reg.to_dict())
    scores: dict[float, float] = {}

    def score(um: float) -> float:
        if um not in scores:
            if cancelled is not None and cancelled():
                raise InterruptedError
            r2.ap_auto = um / 25.0
            pl = adjusted_plane(r2)
            smp = sample_plane(pl, W, H, ("annotation", "reference"))
            lab, ref = smp["annotation"], smp["reference"].astype(np.float32)
            if ref.max() <= 0 or (lab > 0).mean() < 0.02:    # 挪出了脑子
                scores[um] = 0.0
            else:
                moving = ndi.gaussian_filter((lab > 0).astype(np.float32), 2)
                w = _elastix(fixed, moving, f2, ndi.gaussian_filter(ref, 1) / float(ref.max()),
                             bspline=False, iters=100)
                scores[um] = similarity(sample_plane(pl, W, H, ("reference",), warp=w)["reference"],
                                        img_oriented, mask_oriented)
        return scores[um]

    n = int(round(AP_RANGE_UM / AP_COARSE_UM))
    for k in range(-n, n + 1):
        score(k * AP_COARSE_UM)
    best = max(scores, key=scores.get)
    for um in (best - AP_FINE_UM, best + AP_FINE_UM):
        if abs(um) <= AP_RANGE_UM:
            score(um)
    best = max(scores, key=scores.get)
    if best != 0.0 and scores[best] < scores[0.0] * AP_MIN_GAIN:
        best = 0.0
    reg.ap_auto = best / 25.0
    return {"shift_um": best, "scores": dict(sorted(scores.items())), "edge": abs(best) >= AP_RANGE_UM}


def label_maps(reg: Registration, W: int, H: int, warp=None):
    """在“方向已校正”的缩略图坐标下，生成脑区标签图和半脑图。"""
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
    bx, by = section.bbox0[0], section.bbox0[1]
    px = (np.asarray(x0) - bx) / ds
    py = (np.asarray(y0) - by) / ds
    if reg.orientation == "r180":
        px, py = W - 1 - px, H - 1 - py
    ix = np.clip(np.round(px).astype(int), 0, W - 1)
    iy = np.clip(np.round(py).astype(int), 0, H - 1)
    return lab[iy, ix], hemi[iy, ix]


HEMI_NAMES = {0: "", 1: "左", 2: "右"}


def region_table():
    return load_atlas().structures
