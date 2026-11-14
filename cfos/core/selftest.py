"""安装后自检：用软件自带的示例脑片把关键步骤各走一遍（读图 → 细胞识别 → 图谱配准 → 导出），
确认安装包里的组件和资源都齐全、能正常工作。约 1 分钟（主要是加载深度学习组件）。

结果写到日志文件夹的“自检结果.txt”，诊断包里也会带上。
命令行：python main.py --selftest   （或安装后的程序加 --selftest 参数）"""
from __future__ import annotations

import datetime as _dt
import importlib
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# 软件运行需要的组件（导入名, 显示名）
MODULES = [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("skimage", "scikit-image"),
           ("tifffile", "tifffile"), ("PIL", "Pillow"), ("openpyxl", "openpyxl（Excel）"),
           ("matplotlib", "matplotlib"), ("onnxruntime", "ONNX Runtime"), ("stardist", "StarDist"),
           ("csbdeep", "CSBDeep"), ("itk", "ITK / elastix"), ("DeepSlice", "DeepSlice"),
           ("zarr", "zarr"), ("networkx", "networkx（功能网络）"),
           ("psutil", "psutil"), ("napari", "napari（看图界面）"), ("qtpy", "Qt 界面")]


class Step:
    def __init__(self, name):
        self.name, self.ok, self.detail, self.sec = name, False, "", 0.0


def _run(steps, name, fn, report, i, n):
    st = Step(name)
    if report:
        report(i, n, f"自检：{name}…")
    t = time.time()
    try:
        st.detail = fn() or ""
        st.ok = True
    except Exception as e:  # noqa: BLE001
        from .log import log, friendly
        log.error("自检失败：%s\n%s", name, traceback.format_exc())
        st.detail = friendly(e) if str(e) else type(e).__name__
    st.sec = time.time() - t
    steps.append(st)
    return st.ok


def run(report=None) -> tuple[bool, str, list]:
    """返回 (全部通过, 结果文字, 各步骤)。report(i, n, 文字) 用于显示进度。"""
    from .resources import resources_dir, stardist_model_dir
    steps: list[Step] = []
    ctx: dict = {}
    n = 6

    def modules():
        missing = []
        for mod, nm in MODULES:
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                missing.append(f"{nm}（{type(e).__name__}: {str(e)[:80]}）")
        if missing:
            raise RuntimeError("缺少组件：" + "；".join(missing))
        return f"{len(MODULES)} 个组件都能加载"

    def resources():
        from . import accel
        r = resources_dir()
        need = ["atlas/allen_ccfv3_25um.npz", "atlas/structures.csv", "deepslice/Allen_Mixed_Best.h5",
                "demo/示例脑片.ome.tif"]
        miss = [p for p in need if not (r / p).exists()]
        if not (r / "deepslice/xception_top.npz").exists() and \
                not (r / "deepslice/xception_weights_tf_dim_ordering_tf_kernels.h5").exists():
            miss.append("deepslice/xception_top.npz")
        d = stardist_model_dir()
        if d is None:
            miss.append("models/2D_versatile_fluo")
        if miss:
            raise RuntimeError("缺少资源文件：" + "、".join(miss))
        onnx_ok = accel._sig_ok(accel.onnx_path(d), accel._signature(d))
        return "图谱、DeepSlice、细胞识别模型、示例图片都在" + ("" if onnx_ok else "（识别模型的 ONNX 文件需要第一次使用时转换）")

    def read():
        from .slide_io import open_slide
        from .sections import find_sections
        sl = open_slide(resources_dir() / "demo" / "示例脑片.ome.tif")
        secs = find_sections(sl)
        if not secs:
            raise RuntimeError("示例图片里没找到脑片")
        ctx["slide"], ctx["section"] = sl, secs[0]
        return f"{len(sl.channels)} 个通道，{sl.pixel_size_um:.2f} µm/像素，找到 {len(secs)} 张脑片"

    def detect():
        from . import accel
        from .detection import DetectionSettings, ChannelParams, detect_section
        from .sections import Section
        sl, s = ctx["slide"], ctx["section"]
        # 取脑片中心 400 µm 见方的一小块
        side = int(round(400 / sl.pixel_size_um))
        x0, y0, w, h = s.bbox0
        bx, by = x0 + w // 2 - side // 2, y0 + h // 2 - side // 2
        sub = Section(id=0, bbox0=(bx, by, side, side),
                      contour0=[(bx, by), (bx + side, by), (bx + side, by + side), (bx, by + side)], name="自检")
        st = DetectionSettings(channels={c: ChannelParams() for c in range(len(sl.channels))})
        df = detect_section(sl, sub, st)
        counts = [int((df["channel"] == c).sum()) for c in range(len(sl.channels))]
        if sum(counts) == 0:
            raise RuntimeError("示例脑片上没有识别到细胞")
        dev = accel._status.get("device") or "TensorFlow"
        names = [c.name for c in sl.channels]
        return "、".join(f"{a} {b} 个" for a, b in zip(names, counts)) + f"（400 µm 见方；神经网络用 {dev}）"

    def register():
        from . import registration as R
        from .coloc import nucleus_channel
        sl, s = ctx["slide"], ctx["section"]
        nuc = nucleus_channel([c.name for c in sl.channels])
        with tempfile.TemporaryDirectory() as td:
            reg = R.auto_register(sl, [s], nuc, td)[s.name]
        img, mask, ds = R.section_thumbnail(sl, s, nuc)
        w = R.nonlinear_fit(reg, R.orient(mask, reg.orientation), R.orient(img, reg.orientation))
        H, W = mask.shape
        lab = R.sample_plane(R.adjusted_plane(reg), W, H, warp=w)["annotation"]
        d = R.dice(lab > 0, R.orient(mask, reg.orientation))
        if d < 0.8:
            raise RuntimeError(f"配准结果异常（轮廓重合度 {d:.2f}）")
        return f"DeepSlice + 非线性贴合（{reg.nl_method}），轮廓重合度 {d:.2f}"

    def export():
        import pandas as pd
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "自检.xlsx"
            pd.DataFrame({"脑区": ["CA1", "DG"], "细胞数": [1, 2]}).to_excel(p, index=False)
            pd.read_excel(p)
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(Path(td) / "自检.png", dpi=(300, 300))
            import tifffile
            tifffile.imwrite(Path(td) / "自检.tif", np.zeros((32, 32), np.uint16))
        return "Excel、PNG、TIFF 都能写出"

    plan = [("加载组件", modules), ("资源文件", resources), ("读取示例图片", read),
            ("细胞识别", detect), ("图谱配准", register), ("导出文件", export)]
    for i, (nm, fn) in enumerate(plan):
        if nm in ("细胞识别", "图谱配准") and "slide" not in ctx:
            st = Step(nm); st.detail = "跳过（示例图片没能读取）"; steps.append(st)
            continue
        _run(steps, nm, fn, report, i, n)
    if "slide" in ctx:
        try:
            ctx["slide"].close()
        except Exception:  # noqa: BLE001
            pass
    ok = all(s.ok for s in steps)
    text = _format(steps, ok)
    try:
        from .log import log_dir, log
        (log_dir() / "自检结果.txt").write_text(text, encoding="utf-8")
        log.info("自检%s\n%s", "通过" if ok else "未通过", text)
    except Exception:  # noqa: BLE001
        pass
    return ok, text, steps


def _format(steps, ok) -> str:
    from .. import __version__, APP_NAME
    from .cpu import describe
    lines = [f"{APP_NAME} v{__version__} 自检（{_dt.datetime.now():%Y-%m-%d %H:%M}）",
             "结果：全部正常" if ok else "结果：有步骤出错（请把诊断包发给开发者）", describe(), ""]
    for s in steps:
        lines.append(f"{'✔' if s.ok else '✘'} {s.name}（{s.sec:.0f} 秒）：{s.detail}")
    return "\n".join(lines)
