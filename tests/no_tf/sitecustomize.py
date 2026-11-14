"""测试用：模拟“安装包里没有 TensorFlow”。把本文件夹加到 PYTHONPATH 即可生效。
tensorflow / keras / tf2onnx / DeepSlice 只能从“微调组件”文件夹（路径里含“微调组件”）找到，否则当作没装。"""
import importlib.machinery
import sys

BLOCK = {"tensorflow", "keras", "tf2onnx", "DeepSlice", "tensorboard", "tf_keras"}
_orig = importlib.machinery.PathFinder.find_spec.__func__


def _find_spec(cls, name, path=None, target=None):
    if name.split(".")[0] in BLOCK and "." not in name:
        path = [p for p in (path or sys.path) if "微调组件" in p]
        if not path:
            return None
    return _orig(cls, name, path, target)


importlib.machinery.PathFinder.find_spec = classmethod(_find_spec)
