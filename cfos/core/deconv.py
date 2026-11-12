"""去卷积和减背景（只用 CPU）。

说明：40x 扫描的像素约 0.25 µm，物镜本身的模糊（点扩散函数）比一个像素还小，
真正让细胞“发糊”的主要是厚切片里焦平面以外的散射光。因此这里的去卷积使用一个
可调的“模糊程度”（高斯近似的点扩散函数，单位 µm），配合 Richardson-Lucy 迭代；
另外提供“减背景”，去掉大范围的雾状背景。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


def gaussian_psf(sigma_px: float) -> np.ndarray:
    r = max(1, int(np.ceil(sigma_px * 4)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma_px * sigma_px))
    return (k / k.sum()).astype(np.float32)


def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 10) -> np.ndarray:
    img = img.astype(np.float32)
    est = np.full_like(img, max(float(img.mean()), 1e-3))
    psf_m = psf[::-1, ::-1]
    eps = 1e-6
    for _ in range(int(iterations)):
        conv = fftconvolve(est, psf, mode="same")
        ratio = img / np.maximum(conv, eps)
        est *= fftconvolve(ratio, psf_m, mode="same")
        np.maximum(est, 0, out=est)
    return est


def subtract_background(img: np.ndarray, radius_px: float) -> np.ndarray:
    """近似“滚球法”减背景：先在缩小的图上做最小值滤波 + 平滑得到背景，再相减。"""
    img = img.astype(np.float32)
    if img.size == 0:
        return img
    f = max(1, int(radius_px // 8))
    small = img[::f, ::f]
    r = max(1, int(radius_px / f))
    bg = ndi.minimum_filter(small, size=2 * r + 1)
    bg = ndi.uniform_filter(bg, size=2 * r + 1)
    bg = ndi.zoom(bg, (img.shape[0] / bg.shape[0], img.shape[1] / bg.shape[1]), order=1)
    bg = bg[: img.shape[0], : img.shape[1]]
    if bg.shape != img.shape:
        bg = np.pad(bg, ((0, img.shape[0] - bg.shape[0]), (0, img.shape[1] - bg.shape[1])), mode="edge")
    return np.maximum(img - bg, 0)


def enhance(img: np.ndarray, pixel_um: float, deconv: bool, blur_um: float, iterations: int,
            bg_radius_um: float) -> np.ndarray:
    """按参数依次做：减背景 → 去卷积。返回 float32，强度量纲与原图相同。"""
    out = img.astype(np.float32)
    if bg_radius_um and bg_radius_um > 0:
        out = subtract_background(out, bg_radius_um / pixel_um)
    if deconv and blur_um > 0 and iterations > 0:
        sigma = blur_um / pixel_um
        if sigma >= 0.3:
            pad = int(np.ceil(sigma * 4))
            p = np.pad(out, pad, mode="reflect")
            out = richardson_lucy(p, gaussian_psf(sigma), iterations)[pad:-pad, pad:-pad]
    return out
