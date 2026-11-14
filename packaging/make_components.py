"""制作“微调组件”包（TensorFlow、tf2onnx 等，只有训练模型时才需要；v0.8.1 计划 2.6）。

版本与开发环境（也就是测试过的版本）完全相同；只装安装包里没有的那些组件（按依赖关系算出来）。
输出到 <输出文件夹>：
- components/微调组件_<版本>_<平台>.zip.001 …（90 MB 分卷，放到 installers 分支 updates/components/）
- 微调组件_<版本>_<平台>.zip（完整的离线包，给没联网的电脑用，可以放 Release）
- components.json（清单：版本、各平台分卷的大小和 MD5；多次运行会合并各平台）

用法：python packaging/make_components.py <输出文件夹> --platform win-amd64|mac-arm64|linux-x86_64
     python packaging/make_components.py --list     只列出组件包里会有哪些软件包（打安装包时排除它们）"""
import argparse
import hashlib
import importlib.metadata as md
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from cfos.core.components import REQUIRED  # noqa: E402

PART = 90 * 1024 * 1024
ROOTS = {"win-amd64": ["tensorflow-cpu", "tf2onnx"], "mac-arm64": ["tensorflow", "tf2onnx"],
         "linux-x86_64": ["tensorflow-cpu", "tf2onnx"]}
PIP_PLAT = {"win-amd64": ["--platform", "win_amd64"],
            "mac-arm64": ["--platform", "macosx_12_0_arm64", "--platform", "macosx_11_0_arm64"],
            "linux-x86_64": ["--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_17_x86_64",
                             "--platform", "manylinux_2_27_x86_64", "--platform", "manylinux_2_28_x86_64"]}
# 训练才需要、但安装包里原来也带的（DeepSlice 只在 TensorFlow 路径下用）
EXTRA = ["DeepSlice"]
DEV_ONLY = {"pip", "setuptools", "wheel", "tensorflow", "typing"}


def norm(n):
    return re.sub(r"[-_.]+", "-", n.lower())


def _requires(name):
    try:
        reqs = md.requires(name) or []
    except md.PackageNotFoundError:
        return []
    out = []
    for r in reqs:
        if "extra ==" in r:
            continue
        out.append(norm(re.split(r"[ ;<>=!~\[(]", r, maxsplit=1)[0]))
    return out


def closure(roots):
    seen, todo = set(), [norm(r) for r in roots]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        todo += _requires(n)
    return seen


def base_roots():
    """安装包要带的（requirements.txt 去掉 TensorFlow 和 DeepSlice）。"""
    out = []
    for ln in (APP / "requirements.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        n = norm(re.split(r"[ ;<>=!~\[]", ln, maxsplit=1)[0])
        if n in ("tensorflow", "tensorflow-cpu", "deepslice", "tf2onnx"):
            continue
        out.append(n)
    return out


def component_packages(platform="win-amd64") -> list[str]:
    """组件包里的软件包：(TensorFlow + tf2onnx + DeepSlice 的全部依赖) − 安装包本来就带的。"""
    need = closure(ROOTS[platform] + EXTRA)
    base = closure(base_roots())
    return sorted(n for n in need - base if n not in DEV_ONLY or n == norm(ROOTS[platform][0]))


def _ver(n):
    for cand in (n, n.replace("-", "_")):
        try:
            return md.version(cand)
        except md.PackageNotFoundError:
            pass
    if n == "tensorflow":          # 开发环境装的是 tensorflow-cpu；Mac 用同版本的 tensorflow
        return md.version("tensorflow-cpu")
    raise SystemExit(f"开发环境里没有 {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--platform", default="win-amd64", choices=list(ROOTS))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    pk = component_packages(a.platform)
    if a.list:
        print("\n".join(pk))
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / f"_build_{a.platform}"
    wheels, site = work / "wheels", work / "site-packages"
    import shutil
    shutil.rmtree(site, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    pins = [f"{n}=={_ver(n)}" for n in pk]
    print("组件包内容：", " ".join(pins))
    plat = PIP_PLAT[a.platform] + ["--python-version", "3.13", "--implementation", "cp", "--abi", "cp313",
                                   "--only-binary=:all:"]
    pip = [sys.executable, "-m", "pip"]
    for r in pins:
        cmd = pip + ["download", "-q", "--no-deps", r, "-d", str(wheels)] + plat
        if subprocess.run(cmd).returncode != 0:      # 纯 Python 包没有平台标签
            subprocess.run(pip + ["download", "-q", "--no-deps", r, "-d", str(wheels), "--only-binary=:all:"], check=True)
    subprocess.run(pip + ["install", "-q", "--no-deps", "--no-compile", "--target", str(site)] + plat
                   + [str(w) for w in sorted(wheels.glob("*.whl"))], check=True)
    # 去掉用不到的部分（头文件、测试）
    for d in ("tensorflow/include", "tensorflow/python/kernel_tests", "bin"):
        shutil.rmtree(site / d, ignore_errors=True)
    for p in site.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    info = {"version": REQUIRED, "platform": a.platform, "packages": pins}
    name = f"微调组件_{REQUIRED}_{a.platform}.zip"
    zp = out / name
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("component.json", json.dumps(info, ensure_ascii=False, indent=1))
        for f in sorted(site.rglob("*")):
            if f.is_file():
                z.write(f, "site-packages/" + f.relative_to(site).as_posix())
    data_md5 = hashlib.md5(zp.read_bytes()).hexdigest()
    (out / "components").mkdir(exist_ok=True)
    parts = []
    with open(zp, "rb") as f:
        k = 1
        while True:
            b = f.read(PART)
            if not b:
                break
            pn = f"components/{name}.{k:03d}"
            (out / pn).write_bytes(b)
            parts.append({"file": pn, "size": len(b), "md5": hashlib.md5(b).hexdigest()})
            k += 1
    mf = out / "components.json"
    man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    if man.get("version") != REQUIRED:
        man = {"version": REQUIRED, "packs": {}}
    man["packs"][a.platform] = {"size": zp.stat().st_size, "md5": data_md5, "parts": parts, "packages": pins,
                                "offline_file": name}
    mf.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"{name}：{zp.stat().st_size / 1e6:.0f} MB，{len(parts)} 个分卷；清单 {mf}")


if __name__ == "__main__":
    main()
