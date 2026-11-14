"""v0.8.1（计划 2.6）核对：不经 TensorFlow 的 ONNX 版本与原来（TensorFlow）结果一致。
1. DeepSlice：预处理与 Keras 读图逐位相同；示例脑片（及其他测试图）切面参数的差别；
2. StarDist：predict_sparse / predict_instances 与 StarDist2D（同样用 ONNX 网络）完全相同。
用法：python tests/v081_onnx_check.py [其他测试图 png ...]"""
import sys, os, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from cfos.core import registration as R, deepslice_onnx as DO, accel
from cfos.core.resources import resources_dir, stardist_model_dir
from cfos.core.slide_io import open_slide
from cfos.core.sections import find_sections

dsd = resources_dir() / "deepslice"
# ---- 测试图：示例脑片的每张脑片（正、倒）+ 命令行给的图
sl = open_slide(resources_dir() / "demo" / "示例脑片.ome.tif")
secs = find_sections(sl)
imgs = []
for s in secs:
    img, mask, ds = R.section_thumbnail(sl, s, 0)
    imgs += [img, R.orient(img, "r180")]
for f in sys.argv[1:]:
    imgs.append(np.asarray(Image.open(f).convert("L")))
print("测试图", len(imgs), "张")

# ---- 1a. 预处理逐位相同
from DeepSlice.neural_network import neural_network as NN
with tempfile.TemporaryDirectory() as td:
    names = []
    for i, im in enumerate(imgs):
        n = os.path.join(td, f"img_{i:04d}.png"); Image.fromarray(im).save(n); names.append(n)
    gen, _, _ = NN.load_images_from_list(names)
    ref = np.concatenate([gen[i] for i in range(len(names))])
mine = np.stack([DO.preprocess(im) for im in imgs])
print("预处理最大差", float(np.abs(ref - mine).max()))
assert np.array_equal(ref, mine), "预处理与 Keras 不同"

# ---- 1b. 切面参数
t = time.time(); tf_planes = R._deepslice_predict_tf(imgs, tempfile.gettempdir()); t_tf = time.time() - t
t = time.time(); on = DO.predict(imgs, dsd); t_onnx = time.time() - t
tfv = np.array([p.to_list() if hasattr(p, "to_list") else [p.ox, p.oy, p.oz, p.ux, p.uy, p.uz, p.vx, p.vy, p.vz] for p in tf_planes])
d = np.abs(tfv - on)
print(f"DeepSlice 切面参数最大差 {d.max():.4f} 体素（AP 位置 oy 最大差 {d[:, 1].max():.4f}）；"
      f"TensorFlow {t_tf:.1f} 秒，ONNX {t_onnx:.1f} 秒（含加载）")
assert d.max() < 0.05, "DeepSlice ONNX 与 TensorFlow 差别过大"

# ---- 2. StarDist
from stardist.models import StarDist2D
from cfos.core.sdnet import OnnxStarDist, onnx_for
md = stardist_model_dir()
ref_m = StarDist2D(None, name=md.name, basedir=str(md.parent))
ref_m.keras_model = accel.OrtNet(accel.onnx_path(md), "CPUExecutionProvider")
mine_m = OnnxStarDist(md, onnx_for(md))
rng = np.random.default_rng(0)
for shape in ((256, 256), (300, 517), (1024, 1000)):
    x = np.clip(rng.normal(0.2, 0.1, shape), 0, 1).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for _ in range(40):
        cy, cx = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        x[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(16, 80)] += 0.6
    a = ref_m.predict_sparse(x, prob_thresh=0.4, show_tile_progress=False)
    b = mine_m.predict_sparse(x, prob_thresh=0.4)
    assert all(np.array_equal(u, v) for u, v in zip(a, b)), shape
    la, _ = ref_m.predict_instances(x, verbose=False)
    lb, _ = mine_m.predict_instances(x)
    assert np.array_equal(la, lb), shape
    print("StarDist", shape, "候选点", len(a[0]), "细胞", int(la.max()), "完全相同")
print("OK")
