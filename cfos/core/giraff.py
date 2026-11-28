"""Single-section giRAff-style NiftyReg localization in the selected native atlas reference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import sys
import struct
import subprocess
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import normalized_mutual_information

from .atlas import load_atlas
from .atlas_choice import STAM

Z_STRIDE = 1
# Coarse-to-fine plane search: every COARSE_STEP-th coronal plane in both orientations first, then every plane
# within REFINE_RADIUS of the COARSE_TOP best candidates (in their own orientation).  On the example series this
# evaluates ~200 of 1052 plane/orientation pairs (5x faster) and found the same best plane as the full search for
# every grid phase.  COARSE_STEP = 1 searches every plane.
COARSE_STEP = 8
COARSE_TOP = 8
REFINE_RADIUS = 8
NIFTYREG = Path(__file__).resolve().with_name("giraff_runtime") / ("windows/reg_aladin.exe" if os.name == "nt" else "bin/reg_aladin")


def _write_plane(path, pixels, voxel_mm):
    """Write the uint8 image as an uncompressed NIfTI-1 with a positive xy sform."""
    height, width = pixels.shape
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 2, width, height, 1, 1, 1, 1, 1)
    struct.pack_into("<2h", header, 70, 2, 8)  # DT_UINT8, bits per voxel
    struct.pack_into("<8f", header, 76, 1, voxel_mm, voxel_mm, 1, 1, 1, 1, 1)
    struct.pack_into("<2f", header, 108, 352, 1)  # image data offset and intensity scale
    header[123] = 2  # millimetres
    struct.pack_into("<h", header, 254, 2)  # sform: aligned anatomical space
    struct.pack_into("<12f", header, 280, voxel_mm, 0, 0, 0, 0, voxel_mm, 0, 0, 0, 0, 1, 0)
    header[344:348] = b"n+1\0"
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(pixels.tobytes(order="C"))


def _center_mm(mask, voxel_mm):
    y, x = ndi.center_of_mass(mask)
    return np.array([x, y]) * voxel_mm


def _resample(moving, affine, shape, voxel_mm):
    # scipy arrays are indexed by y,x; NiftyReg matrices use physical x,y.
    block = affine[:2, :2][::-1, ::-1]
    offset = affine[:2, 3][::-1] / voxel_mm
    return ndi.affine_transform(moving, block, offset, output_shape=shape,
                                order=1, mode="grid-constant", cval=0)


def _geometry(affine, z, shape, orientation, scores, voxel_mm):
    h, w = shape
    linear = affine[:2, :2] * voxel_mm
    # NIfTI image voxel indices are centre-based. FosFolio's plane origin is the
    # outer edge at (-0.5, -0.5), and sampling uses normalized pixel centres.
    origin = affine[:2, 3] - .5 * linear.sum(axis=1)
    o = [*origin, z * voxel_mm]
    u = [*(w * linear[:, 0]), 0.]
    v = [*(h * linear[:, 1]), 0.]
    return {
        "plane_image_mm": np.asarray([o, u, v]).ravel().tolist(),
        "orientation": orientation, "atlas_z_index": z,
        "search_step_um": round(1000 * voxel_mm * Z_STRIDE),
        "score": float(np.mean(scores)),
        "rigid_nmi": scores[0], "affine_nmi": scores[1],
    }


def refine_planes(candidates, n, step, top, radius):
    """Second stage of the plane search: [(z, orientations)] within radius of the top coarse candidates."""
    best = sorted(candidates, key=lambda c: c["score"], reverse=True)[:top]
    seen = {(c["atlas_z_index"], c["orientation"]) for c in candidates}
    wanted = {}
    for c in best:
        z0 = c["atlas_z_index"]
        for z in range(max(0, z0 - radius), min(n, z0 + radius + 1)):
            if z % step and (z, c["orientation"]) not in seen:
                wanted.setdefault(z, set()).add(c["orientation"])
    return sorted((z, tuple(sorted(o))) for z, o in wanted.items())


def fit(image_u8, mask, pixel_um, progress=None, cancelled=None, *, atlas=STAM):
    """Search coronal planes in both orientations and return geometry, oriented DAPI, tissue."""
    image = np.asarray(image_u8)
    tissue = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or image.shape != tissue.shape or image.dtype != np.uint8:
        raise ValueError("giRAff 需要二维 uint8 结构通道及同尺寸的组织掩膜。")
    if not np.isfinite(pixel_um) or pixel_um <= 0 or not tissue.any() or image.max() == image.min():
        raise ValueError("无法配准：请检查像素标定、组织掩膜和结构通道的对比度。")
    if os.name != "nt" and NIFTYREG.is_file() and not os.access(NIFTYREG, os.X_OK):
        NIFTYREG.chmod(NIFTYREG.stat().st_mode | 0o100)
    if not NIFTYREG.is_file() or not os.access(NIFTYREG, os.X_OK):
        raise FileNotFoundError("找不到可执行的 NiftyReg 组件：" + str(NIFTYREG))
    if cancelled and cancelled():
        raise InterruptedError

    env = os.environ.copy()
    if os.name == "nt":
        prefix = Path(sys.executable).resolve().parent
        runtime_dirs = [prefix, prefix/"Library"/"bin", prefix/"Lib"/"site-packages"/"torch"/"lib"]
        env["PATH"] = os.pathsep.join([*(str(p) for p in runtime_dirs), env.get("PATH", "")])
    ref = load_atlas(atlas).reference
    voxel_mm = .032 if atlas == STAM else .025
    size = tuple(max(1, round(n * pixel_um / (1000 * voxel_mm))) for n in image.shape[::-1])
    if min(size) < 16 or max(size) > 32767:
        raise ValueError("按像素标定换算后的脑片尺寸无法用于 NiftyReg，请检查标定和裁剪范围。")
    fixed_r0 = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
    mask_r0 = np.asarray(Image.fromarray(tissue).resize(size, Image.Resampling.NEAREST))
    if not mask_r0.any():
        raise ValueError("下采样后的脑片没有可用组织。")

    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError

    with tempfile.TemporaryDirectory(prefix="fosfolio-giraff-") as tmp:
        root = Path(tmp)
        inputs = {}
        for orientation, fixed, oriented_mask in [
            ("r0", fixed_r0, mask_r0), ("r180", fixed_r0[::-1, ::-1], mask_r0[::-1, ::-1])
        ]:
            path = root / f"fixed-{orientation}.nii"
            _write_plane(path, fixed, voxel_mm)
            inputs[orientation] = (fixed, path, _center_mm(oriented_mask, voxel_mm))

        def evaluate(z, orientations=None):
            check_cancelled()
            moving = ref[z]
            support = ndi.binary_fill_holes(moving > 8)
            if support.sum() < 200:
                return []
            d = root / str(z)
            d.mkdir(exist_ok=True)
            target = d / "atlas.nii"
            _write_plane(target, moving, voxel_mm)
            center = _center_mm(support, voxel_mm)
            results = []
            for orientation, (fixed, fixed_file, fixed_center) in inputs.items():
                if orientations is not None and orientation not in orientations:
                    continue
                init = d / f"init-{orientation}.txt"
                affine = np.eye(4)
                affine[:2, 3] = center - fixed_center
                np.savetxt(init, affine)
                scores = []
                for mode, name in [("-rigOnly", "rigid"), ("-affDirect", "affine")]:
                    check_cancelled()
                    result = d / (name + ".txt")
                    command = [str(NIFTYREG), "-ref", str(fixed_file), "-flo", str(target),
                               "-omp", "1", mode, "-inaff", str(init), "-aff", str(result),
                               "-res", str(d / "res.nii")]
                    proc = subprocess.run(command, cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if proc.returncode:
                        raise RuntimeError(f"NiftyReg 配准失败（{proc.returncode}）："
                                           + proc.stderr.decode(errors="replace")[-350:])
                    affine = np.loadtxt(result)
                    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                            or np.linalg.det(affine[:2, :2]) <= 0):
                        break
                    warped = _resample(moving, affine, fixed.shape, voxel_mm)
                    nmi = float(normalized_mutual_information(fixed, warped, bins=64))
                    if not np.isfinite(nmi):
                        break
                    scores.append(nmi)
                    init = result
                if len(scores) == 2:
                    results.append(_geometry(affine, z, fixed.shape, orientation, scores, voxel_mm))
            return results

        best = None
        workers = min(4, os.cpu_count() or 1)
        candidates = []

        def run(stage, total, offset):
            nonlocal best
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(evaluate, z, o) for z, o in stage]
                try:
                    for done, job in enumerate(as_completed(jobs), start=1):
                        for candidate in job.result():
                            candidates.append(candidate)
                            # Include deterministic tie breakers since candidates arrive asynchronously.
                            rank = (candidate["score"], -candidate["atlas_z_index"], candidate["orientation"] == "r0")
                            if best is None or rank > best[0]:
                                best = (rank, candidate)
                        if progress:
                            progress(offset + done, total, f"giRAff 已搜索 {offset + done}/{total} 个 {atlas} 冠状面")
                finally:
                    for job in jobs:
                        job.cancel()

        n = ref.shape[0]
        step = max(1, int(COARSE_STEP))
        coarse = [(z, None) for z in range(0, n, step)]
        if step == 1:
            run(coarse, len(coarse), 0)
        else:
            # The refine stage size is not known yet; report it as at most this many planes.
            estimate = len(coarse) + COARSE_TOP * (2 * REFINE_RADIUS)
            run(coarse, estimate, 0)
            fine = refine_planes(candidates, n, step, COARSE_TOP, REFINE_RADIUS)
            run(fine, len(coarse) + len(fine), len(coarse))
    if best is None:
        raise ValueError(f"未找到有效的 {atlas} 冠状面匹配。")
    geometry = best[1]
    if geometry["orientation"] == "r180":
        return geometry, np.ascontiguousarray(image[::-1, ::-1]), np.ascontiguousarray(tissue[::-1, ::-1])
    return geometry, image, tissue
