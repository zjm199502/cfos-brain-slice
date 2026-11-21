"""打包前转换并验证 DeepSlice 的两组图谱配准权重。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
