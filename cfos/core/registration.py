"""脑片与 Allen CCFv3 图谱的配准。

流程：
1. 每张脑片生成一张约 8 µm/像素 的缩略图（DAPI），交给 DeepSlice 估计它在图谱中的切片平面。
   切片在玻片上可能是倒放的，所以同时试 0° 和 180° 两个方向，选与图谱轮廓重合最好的方向。
2. 可选：用 elastix 做非线性贴合（以组织轮廓为准），修正切片变形。
3. 用户可以手动调整：前后位置、两个倾斜角、缩放、旋转、平移，以及在图上拖动“修正点”做局部修正。
4. 最终得到一张与缩略图对齐的脑区标签图，用来给每个细胞分配脑区和左右半脑。
"""
from __future__ import annotations

import os
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
    tilt_dv: float = 0.0                   # 背腹方向倾斜（度）
    tilt_ml: float = 0.0                   # 左右方向倾斜（度）
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0                  # 平面内旋转（度）
    shift_x: float = 0.0                   # 平面内平移（图像宽度的比例）
    shift_y: float = 0.0
    control_points: list = field(default_factory=list)  # [[src_s, src_t, dst_s, dst_t], ...] 归一化坐标
    nonlinear: bool = False                # 是否使用 elastix 非线性贴合
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
        model = neural_network.initialise_network(
            str(d / "xception_weights_tf_dim_ordering_tf_kernels.h5"), None, "mouse")
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
    """对若干张缩略图预测切片平面。"""
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


def auto_register(slide: Slide, sections: list[Section], nucleus_ch, workdir, progress=None) -> dict:
    """返回 {section.name: Registration}。"""
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
    c = c + n * reg.ap_shift
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


def warp_path(results_dir, image_key, section_name):
    d = results_dir / "registration"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{image_key}_{section_name}_warp.npy"


def nonlinear_fit(reg: Registration, mask_oriented: np.ndarray):
    """用 elastix 把图谱切片的脑轮廓贴到组织轮廓上，返回 (dx, dy) 像素位移场。"""
    import itk
    H, W = mask_oriented.shape
    pl = adjusted_plane(reg)
    lab = sample_plane(pl, W, H)["annotation"]
    fixed = ndi.gaussian_filter(mask_oriented.astype(np.float32), 2)
    moving = ndi.gaussian_filter((lab > 0).astype(np.float32), 2)
    fi = itk.image_view_from_array(fixed)
    mi = itk.image_view_from_array(moving)
    po = itk.ParameterObject.New()
    aff = po.GetDefaultParameterMap("affine", 3)
    aff["MaximumNumberOfIterations"] = ["300"]
    bs = po.GetDefaultParameterMap("bspline", 3, 40.0)
    bs["MaximumNumberOfIterations"] = ["300"]
    bs["FinalGridSpacingInPhysicalUnits"] = ["60"]
    po.AddParameterMap(aff)
    po.AddParameterMap(bs)
    _, tp = itk.elastix_registration_method(fi, mi, parameter_object=po, log_to_console=False)
    field_ = itk.transformix_deformation_field(mi, tp)
    f = itk.array_from_image(field_)  # (H, W, 2)，单位：像素，(x, y)
    return f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)


def label_maps(reg: Registration, W: int, H: int, warp=None):
    """在“方向已校正”的缩略图坐标下，生成脑区标签图和半脑图。"""
    pl = adjusted_plane(reg)
    total = None
    t = tps_warp(reg.control_points, W, H)
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
