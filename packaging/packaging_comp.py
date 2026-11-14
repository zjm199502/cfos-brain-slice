"""给 list_third_party.py 用：微调组件里的软件包（规范化名称）。"""
import importlib.util
from pathlib import Path


def component_set(platform="win-amd64") -> set:
    spec = importlib.util.spec_from_file_location("make_components", Path(__file__).with_name("make_components.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return set(m.component_packages(platform))
