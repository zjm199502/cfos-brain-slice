"""运行界面测试脚本的外壳。细胞识别用多进程（spawn）时，子进程会重新导入主模块；
测试脚本没有 `if __name__ == "__main__"` 保护，所以由这个有保护的外壳来运行它们。
用法：xvfb-run python tests/run.py tests/batch_smoke.py <参数…>"""
import runpy
import sys

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__cfos_test__")
