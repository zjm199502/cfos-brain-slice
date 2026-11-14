"""打包前：把软件自带的通用 StarDist 模型转换成 ONNX（model.onnx + model.sig），随安装包一起带上。
这样用户第一次识别时不需要转换，也不需要往安装目录写文件。
用法：python packaging/make_onnx.py   （已是最新时什么也不做；失败时以非 0 退出，打包中止）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cfos.core import accel
from cfos.core.resources import stardist_model_dir

d = stardist_model_dir()
assert d is not None, "找不到 resources/models/2D_versatile_fluo"
sig = accel._signature(d)
p = accel.onnx_path(d)
if accel._sig_ok(p, sig) and p.with_suffix(".sig").exists():
    print("model.onnx 已是最新", p)
else:
    from stardist.models import StarDist2D
    m = StarDist2D(None, name=d.name, basedir=str(d.parent))
    accel.export_onnx(m.keras_model, p, sig)
    print("已生成", p)
# 核对：ONNX 和 TensorFlow 在同一块随机图像上输出一致
from stardist.models import StarDist2D
m = StarDist2D(None, name=d.name, basedir=str(d.parent))
x = np.random.default_rng(0).random((1, 256, 256, 1), dtype=np.float32)
ys = accel.OrtNet(p, "CPUExecutionProvider").predict(x)
ref = m.keras_model.predict(x, verbose=0)
dp, dd = float(np.abs(ys[0] - ref[0]).max()), float(np.abs(ys[1] - ref[1]).max())
print(f"ONNX 与 TensorFlow 对比：概率差 {dp:.5f}，距离差 {dd:.5f}")
assert dp < accel.CHECK_TOL_PROB and dd < accel.CHECK_TOL_DIST, "ONNX 结果与 TensorFlow 不一致"
assert p.with_suffix(".sig").read_text(encoding="utf-8").strip() == sig

# ---------------------------------------------------------------- DeepSlice（v0.8.1）
# DeepSlice 的两个小鼠模型（Allen_Mixed_Best、Synthetic_data_final）各转换成一个 ONNX 文件，
# 配准时用 ONNX Runtime 计算、不再加载 TensorFlow。转换后逐张核对与 TensorFlow 的输出。
from cfos.core import registration as R, deepslice_onnx as DO
from cfos.core.resources import resources_dir
dsd = resources_dir() / "deepslice"
need = [(DO.PRIMARY, "Allen_Mixed_Best.h5"), (DO.SECONDARY, "Synthetic_data_final.hdf5")]
srcsig = {o: DO.source_signature(dsd, h5) for o, h5 in need}
if all(DO.onnx_ok(dsd / o, srcsig[o]) for o, _ in need):
    print("DeepSlice ONNX 已是最新")
else:
    import tensorflow as tf
    import tf2onnx
    ds = R._deepslice()
    model = ds.model
    spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
    for o, h5 in need:
        R._load_deepslice_weights(model, str(dsd / h5))

        @tf.function(input_signature=spec)
        def f(x):
            return model(x, training=False)
        tmp = dsd / (o + ".tmp")
        tf2onnx.convert.from_function(f, input_signature=spec, opset=17, output_path=str(tmp))
        tmp.replace(dsd / o)
        (dsd / o).with_suffix(".sig").write_text(srcsig[o], encoding="utf-8")
        print("已生成", dsd / o, f"{(dsd / o).stat().st_size / 1e6:.0f} MB")
    R._load_deepslice_weights(model, str(dsd / "Allen_Mixed_Best.h5"))
