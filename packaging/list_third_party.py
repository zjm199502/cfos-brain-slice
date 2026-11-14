"""生成 THIRD_PARTY.md（第三方组件与致谢）：软件用到的所有公开软件包、模型、数据和代码的名称、版本和来源。
按开发环境（与安装包相同的版本）自动列出所有 Python 包；模型和数据手工列在 KEY 里。
用法：python packaging/list_third_party.py   （打包前运行，更新仓库里的 THIRD_PARTY.md）"""
import importlib.metadata as md
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
EXCL = {re.sub(r"[-_.]+", "-", l.strip().lower()) for l in
        (APP / "packaging" / "不打包的组件.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
DEV_ONLY = {"pip", "setuptools", "wheel", "pyflakes", "sympy", "mpmath", "typing", "tensorflow-cpu"}
DEV_PREFIX = ("nvidia-", "triton")


def _v(n):
    try:
        return md.version(n)
    except md.PackageNotFoundError:
        return "—"


# 模型、数据、关键算法（手工维护）
KEY = [
    ("StarDist", _v("stardist"), "细胞识别算法", "BSD-3-Clause", "https://github.com/stardist/stardist",
     "Schmidt et al., MICCAI 2018; Weigert et al., WACV 2020"),
    ("StarDist 预训练模型 2D_versatile_fluo", "0.1", "细胞核识别模型（随安装包附带，并转换为 ONNX 格式）",
     "BSD-3-Clause", "https://github.com/stardist/stardist-models", "同上"),
    ("DeepSlice", _v("DeepSlice"), "切片在图谱中的自动定位",
     "GPL-3.0", "https://github.com/PolarBean/DeepSlice", "Carey et al., Nature Communications 2023"),
    ("DeepSlice 小鼠模型权重（Allen_Mixed_Best、Synthetic_data_final）", "—",
     "DeepSlice 的预训练模型（两个模型取平均；v0.8.1 起转换为 ONNX 格式随安装包附带）",
     "GPL-3.0", "https://github.com/PolarBean/DeepSliceWeightsMQ", "同上"),
    ("Xception ImageNet 预训练权重", "v0.4", "DeepSlice 网络的顶层（只带其中 ImageNet 分类层的权重）", "MIT",
     "https://github.com/fchollet/deep-learning-models/releases/tag/v0.4", "Chollet, CVPR 2017"),
    ("Allen Mouse Brain Common Coordinate Framework v3（CCFv3）", "2017，25 µm",
     "脑图谱（注释、平均模板、脑区结构表），经 BrainGlobe（allen_mouse_25um）获取", "Allen Institute Terms of Use",
     "https://atlas.brain-map.org/ ； https://brainglobe.info/", "Wang et al., Cell 2020"),
    ("elastix（itk-elastix）", _v("itk-elastix"), "非线性配准", "Apache-2.0", "https://github.com/InsightSoftwareConsortium/ITKElastix",
     "Klein et al., IEEE TMI 2010; Shamonin et al., Front. Neuroinform. 2014"),
    ("QuickNII 坐标约定", "—", "图谱切面参数（与 DeepSlice 输出一致），只参考了公开的格式说明，没有使用其代码", "—",
     "https://www.nitrc.org/projects/quicknii", "Puchades et al., PLoS ONE 2019"),
    ("TensorFlow / Keras", f"{_v('tensorflow')} / {_v('keras')}", "神经网络运行（StarDist、DeepSlice）", "Apache-2.0",
     "https://www.tensorflow.org/", ""),
    ("ONNX Runtime", _v("onnxruntime"), "细胞识别神经网络的 CPU 计算", "MIT", "https://onnxruntime.ai/", ""),
    ("napari", _v("napari"), "看图界面", "BSD-3-Clause", "https://napari.org/", "napari contributors, Zenodo"),
    ("Qt for Python（PySide6）", _v("PySide6"), "图形界面", "LGPL-3.0（未修改，动态链接）", "https://www.qt.io/qt-for-python", ""),
    ("Python（python-build-standalone）", "3.13", "安装包中的 Python 运行环境", "PSF License-2.0 等",
     "https://github.com/astral-sh/python-build-standalone", ""),
    ("NSIS", "3", "Windows 安装程序", "zlib/libpng", "https://nsis.sourceforge.io/", ""),
    ("Noto Sans CJK / 系统字体", "—", "截图和图表中的中文（使用系统已有字体，不随软件分发）", "SIL OFL 1.1（Noto）",
     "https://github.com/notofonts/noto-cjk", ""),
]


def _license(d):
    m = d.metadata
    lic = (m.get("License-Expression") or "").strip()
    if not lic:
        raw = (m.get("License") or "").strip()
        if raw and len(raw) < 60 and "\n" not in raw:
            lic = raw
    if not lic:
        cls = [c.split("::")[-1].strip() for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
        lic = "; ".join(cls)
    if not lic:
        raw = (m.get("License") or "").strip().splitlines()
        lic = raw[0][:60] if raw else "见项目主页"
    return lic.replace("|", "/")


def _url(d):
    m = d.metadata
    urls = m.get_all("Project-URL") or []
    for pref in ("Source", "Repository", "Homepage", "Home", "source", "repository", "homepage", "Code", "GitHub"):
        for u in urls:
            k, _, v = u.partition(",")
            if k.strip().lower() == pref.lower() and v.strip():
                return v.strip()
    if m.get("Home-page"):
        return m["Home-page"]
    if urls:
        return urls[0].partition(",")[2].strip()
    return f"https://pypi.org/project/{m['Name']}/"


def main():
    import sys
    sys.path.insert(0, str(APP / "packaging"))
    from packaging_comp import component_set
    comp = component_set()
    rows, crow = [], []
    seen = set()
    for d in md.distributions():
        n = d.metadata["Name"]
        key = re.sub(r"[-_.]+", "-", n.lower())
        if key in seen or key in EXCL or key in DEV_ONLY or key.startswith(DEV_PREFIX):
            continue
        seen.add(key)
        (crow if key in comp else rows).append((n, d.version, _license(d), _url(d)))
    rows.sort(key=lambda r: r[0].lower())
    crow.sort(key=lambda r: r[0].lower())
    L = ["# 第三方组件与致谢 / Third-party components and acknowledgements", "",
         "下面列出 FosMap 脑片活动图谱用到的所有公开软件包、模型、数据和代码，以及它们的版本和来源；"
         "版本即安装包中实际使用的版本。", "",
         "## 模型、数据和主要算法", "",
         "| 名称 | 版本 | 用途 | 来源 | 引用 |", "|---|---|---|---|---|"]
    for r in KEY:
        L.append("| " + " | ".join(r[:3] + r[4:]) + " |")
    L += ["", f"## 安装包中的全部 Python 软件包（{len(rows)} 个）", "",
          "由 `packaging/list_third_party.py` 按安装包的版本自动生成。", "",
          "| 软件包 | 版本 | 来源 |", "|---|---|---|"]
    for n, v, lic, u in rows:
        L.append(f"| {n} | {v} | {u} |")
    if crow:
        L += ["", f"## 微调组件中的 Python 软件包（{len(crow)} 个）", "",
              "只有在软件里训练（微调）识别模型时才会下载安装（见 `packaging/make_components.py`），"
              "细胞识别和图谱配准不需要它们。", "",
              "| 软件包 | 版本 | 来源 |", "|---|---|---|"]
        for n, v, lic, u in crow:
            L.append(f"| {n} | {v} | {u} |")
    L += ["", "## 本软件自己的代码", "",
          "`cfos/` 文件夹中的全部代码、`packaging/` 中的打包脚本和 `resources/teaching/` 中的教学示意图由本项目编写。"
          "`resources/demo/` 中的示例脑片由本项目提供，仅供学习本软件使用。", "",
          "联系：zjm199502@163.com，https://github.com/zjm199502/cfos-brain-slice", ""]
    (APP / "THIRD_PARTY.md").write_text("\n".join(L), encoding="utf-8")
    print("已写出 THIRD_PARTY.md：", len(KEY), "项模型/数据/算法，", len(rows), "个 Python 包 +", len(crow), "个微调组件包")


if __name__ == "__main__":
    main()
