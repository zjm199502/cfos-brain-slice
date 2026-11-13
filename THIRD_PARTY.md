# 第三方组件与致谢 / Third-party components and acknowledgements

下面列出 cFos 脑片分析用到的所有公开软件包、模型、数据和代码，以及它们的版本和来源；版本即安装包中实际使用的版本。
发表论文时，请同时引用下表中 StarDist、DeepSlice、elastix 和 Allen CCFv3 的原始文献。

## 模型、数据和主要算法

| 名称 | 版本 | 用途 | 来源 | 引用 |
|---|---|---|---|---|
| StarDist | 0.9.2 | 细胞识别算法 | https://github.com/stardist/stardist | Schmidt et al., MICCAI 2018; Weigert et al., WACV 2020 |
| StarDist 预训练模型 2D_versatile_fluo | 0.1 | 细胞核识别模型（随安装包附带，并转换为 ONNX 格式） | https://github.com/stardist/stardist-models | 同上 |
| DeepSlice | 1.2.8 | 切片在图谱中的自动定位 | https://github.com/PolarBean/DeepSlice | Carey et al., Nature Communications 2023 |
| DeepSlice 小鼠模型权重（Allen_Mixed_Best、Synthetic_data_final） | — | DeepSlice 的预训练模型（两个模型取平均） | https://github.com/PolarBean/DeepSliceWeightsMQ | 同上 |
| Xception ImageNet 预训练权重 | v0.4 | DeepSlice 网络的顶层（只带其中 ImageNet 分类层的权重） | https://github.com/fchollet/deep-learning-models/releases/tag/v0.4 | Chollet, CVPR 2017 |
| Allen Mouse Brain Common Coordinate Framework v3（CCFv3） | 2017，25 µm | 脑图谱（注释、平均模板、脑区结构表），经 BrainGlobe（allen_mouse_25um）获取 | https://atlas.brain-map.org/ ； https://brainglobe.info/ | Wang et al., Cell 2020 |
| elastix（itk-elastix） | 0.25.4 | 非线性配准 | https://github.com/InsightSoftwareConsortium/ITKElastix | Klein et al., IEEE TMI 2010; Shamonin et al., Front. Neuroinform. 2014 |
| QuickNII 坐标约定 | — | 图谱切面参数（与 DeepSlice 输出一致），只参考了公开的格式说明，没有使用其代码 | https://www.nitrc.org/projects/quicknii | Puchades et al., PLoS ONE 2019 |
| TensorFlow / Keras | 2.21.0 / 3.15.1 | 神经网络运行（StarDist、DeepSlice） | https://www.tensorflow.org/ |  |
| ONNX Runtime | 1.31.1 | 细胞识别神经网络的 CPU 计算 | https://onnxruntime.ai/ |  |
| napari | 0.10.0 | 看图界面 | https://napari.org/ | napari contributors, Zenodo |
| Qt for Python（PySide6） | 6.12.0 | 图形界面 | https://www.qt.io/qt-for-python |  |
| Python（python-build-standalone） | 3.13 | 安装包中的 Python 运行环境 | https://github.com/astral-sh/python-build-standalone |  |
| NSIS | 3 | Windows 安装程序 | https://nsis.sourceforge.io/ |  |
| Noto Sans CJK / 系统字体 | — | 截图和图表中的中文（使用系统已有字体，不随软件分发） | https://github.com/notofonts/noto-cjk |  |

## 安装包中的全部 Python 软件包（136 个）

由 `packaging/list_third_party.py` 按安装包的版本自动生成。

| 软件包 | 版本 | 来源 |
|---|---|---|
| absl-py | 2.5.1 | https://github.com/abseil/abseil-py |
| annotated-doc | 0.0.6 | https://github.com/fastapi/annotated-doc |
| annotated-types | 0.8.0 | https://github.com/annotated-types/annotated-types |
| app-model | 0.5.1 | https://github.com/pyapp-kit/app-model |
| astunparse | 1.6.3 | https://github.com/simonpercivall/astunparse |
| attrs | 26.1.0 | https://github.com/python-attrs/attrs |
| build | 1.6.1 | https://github.com/pypa/build |
| cachey | 0.2.1 | http://github.com/dask/cachey/ |
| certifi | 2026.10.23 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | https://github.com/jawah/charset_normalizer |
| click | 8.5.1 | https://github.com/pallets/click/ |
| cloudpickle | 3.1.2 | https://github.com/cloudpipe/cloudpickle |
| contourpy | 1.4.0 | https://github.com/contourpy/contourpy |
| csbdeep | 0.8.2 | https://github.com/csbdeep/csbdeep |
| cycler | 0.12.1 | https://github.com/matplotlib/cycler |
| dask | 2026.8.0 | https://github.com/dask/dask/ |
| DeepSlice | 1.2.8 | https://github.com/PolarBean/DeepSlice |
| docstring_parser | 0.18.0 | https://github.com/rr-/docstring_parser |
| donfig | 0.8.1.post1 | https://github.com/pytroll/donfig |
| et_xmlfile | 2.0.0 | https://foss.heptapod.net/openpyxl/et_xmlfile |
| flatbuffers | 25.12.19 | https://github.com/google/flatbuffers |
| flexcache | 0.3 | https://github.com/hgrecco/flexcache |
| flexparser | 0.4 | https://github.com/hgrecco/flexparser |
| fonttools | 4.68.0 | http://github.com/fonttools/fonttools |
| freetype-py | 2.5.1 | https://github.com/rougier/freetype-py |
| fsspec | 2026.9.0 | https://github.com/fsspec/filesystem_spec |
| gast | 0.7.0 | https://github.com/serge-sans-paille/gast/ |
| google-crc32c | 1.9.0 | https://github.com/googleapis/python-crc32c |
| google-pasta | 0.2.0 | https://github.com/google/pasta |
| grpcio | 1.85.1 | https://grpc.io |
| h5py | 3.14.0 | https://github.com/h5py/h5py |
| HeapDict | 1.0.1 | http://stutzbachenterprises.com/ |
| hsluv | 5.0.5 | https://www.hsluv.org |
| idna | 3.20 | https://github.com/kjd/idna |
| imagecodecs | 2026.10.10 | https://www.cgohlke.com |
| ImageIO | 2.38.1 | https://github.com/imageio/imageio |
| importlib_resources | 7.1.0 | https://github.com/python/importlib_resources |
| in-n-out | 0.2.1 | https://github.com/pyapp-kit/in-n-out |
| itk | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-core | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-elastix | 0.25.4 | https://github.com/InsightSoftwareConsortium/ITKElastix |
| itk-filtering | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-io | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-numerics | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-registration | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| itk-segmentation | 5.4.7 | https://github.com/InsightSoftwareConsortium/ITK |
| joblib | 1.6.0 | https://github.com/joblib/joblib |
| jsonschema | 4.26.0 | https://github.com/python-jsonschema/jsonschema |
| jsonschema-specifications | 2025.9.1 | https://github.com/python-jsonschema/jsonschema-specifications |
| keras | 3.15.1 | https://github.com/keras-team/keras |
| kiwisolver | 1.5.1 | https://github.com/nucleic/kiwi |
| lazy-loader | 0.6 | https://github.com/scientific-python/lazy-loader |
| libclang | 18.1.1 | https://github.com/sighingnow/libclang |
| llvmlite | 0.50.0 | https://github.com/numba/llvmlite |
| locket | 1.0.0 | http://github.com/mwilliamson/locket.py |
| lxml | 6.1.3 | https://github.com/lxml/lxml |
| magicgui | 0.10.2 | https://github.com/pyapp-kit/magicgui |
| markdown-it-py | 4.3.0 | https://github.com/executablebooks/markdown-it-py |
| matplotlib | 3.11.2 | https://matplotlib.org |
| mdurl | 0.1.2 | https://github.com/executablebooks/mdurl |
| ml_dtypes | 0.6.0 | https://github.com/jax-ml/ml_dtypes |
| msgspec | 0.22.0 | https://github.com/msgspec/msgspec |
| namex | 0.1.0 | https://pypi.org/project/namex/ |
| napari | 0.10.0 | https://napari.org |
| napari-plugin-engine | 0.2.1 | https://github.com/napari/napari-plugin-engine |
| napari-resources | 1.0.1 | https://github.com/napari/napari-resources |
| napari-svg | 0.2.1 | https://github.com/napari/napari-svg |
| narwhals | 2.28.0 | https://github.com/narwhals-dev/narwhals |
| networkx | 3.7 | https://networkx.org/ |
| npe2 | 0.10.2 | https://github.com/napari/npe2 |
| numba | 0.68.1 | https://numba.pydata.org |
| numcodecs | 0.17.1 | https://github.com/zarr-developers/numcodecs |
| numpy | 2.4.6 | https://github.com/numpy/numpy |
| onnx | 1.23.2 | https://github.com/onnx/onnx |
| onnxruntime | 1.31.1 | https://onnxruntime.ai |
| openpyxl | 3.1.5 | https://foss.heptapod.net/openpyxl/openpyxl |
| opt_einsum | 3.4.0 | https://pypi.org/project/opt_einsum/ |
| optree | 0.20.0 | https://github.com/metaopt/optree |
| packaging | 26.3 | https://github.com/pypa/packaging |
| pandas | 3.1.1 | https://github.com/pandas-dev/pandas |
| partd | 1.4.2 | http://github.com/dask/partd/ |
| pillow | 13.0.0 | https://github.com/python-pillow/Pillow |
| Pint | 0.26.2 | https://github.com/hgrecco/pint |
| platformdirs | 4.14.5 | https://github.com/tox-dev/platformdirs |
| pooch | 1.9.0 | https://www.fatiando.org/pooch |
| protobuf | 7.36.2 | https://developers.google.com/protocol-buffers/ |
| psutil | 7.2.2 | https://github.com/giampaolo/psutil |
| psygnal | 0.16.1 | https://github.com/pyapp-kit/psygnal |
| pyconify | 0.3.0 | https://github.com/pyapp-kit/pyconify |
| pydantic | 2.14.3 | https://github.com/pydantic/pydantic |
| pydantic-extra-types | 2.11.1 | https://github.com/pydantic/pydantic-extra-types |
| pydantic-settings | 2.16.2 | https://github.com/pydantic/pydantic-settings |
| pydantic_core | 2.50.4 | https://github.com/pydantic/pydantic/tree/main/pydantic-core |
| Pygments | 2.21.0 | https://github.com/pygments/pygments |
| PyOpenGL | 3.1.10 | https://github.com/mcfletch/pyopengl |
| pyparsing | 3.3.3 | https://github.com/pyparsing/pyparsing.git |
| pyproject_hooks | 1.3.3 | https://github.com/pypa/pyproject-hooks |
| PySide6 | 6.12.0 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| PySide6_Addons | 6.12.0 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| PySide6_Essentials | 6.12.0 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| PySide6_Pdf | 6.12.0.140 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| PySide6_WebEngine | 6.12.0.140 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| python-dateutil | 2.9.0.post0 | https://github.com/dateutil/dateutil |
| python-dotenv | 1.2.4 | https://github.com/theskumar/python-dotenv |
| PyYAML | 6.0.3 | https://pyyaml.org/ |
| QtPy | 2.4.3 | https://github.com/spyder-ide/qtpy |
| referencing | 0.37.0 | https://github.com/python-jsonschema/referencing |
| requests | 2.34.2 | https://github.com/psf/requests |
| rich | 15.0.0 | https://github.com/Textualize/rich |
| rich-argparse | 1.8.0 | https://github.com/hamdanal/rich-argparse |
| rpds-py | 2026.9.1 | https://github.com/crate-py/rpds |
| scikit-image | 0.26.0 | https://github.com/scikit-image/scikit-image |
| scikit-learn | 1.9.1 | https://github.com/scikit-learn/scikit-learn |
| scipy | 1.18.1 | https://github.com/scipy/scipy |
| shellingham | 1.5.4 | https://github.com/sarugaku/shellingham |
| shiboken6 | 6.12.0 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| six | 1.17.0 | https://github.com/benjaminp/six |
| stardist | 0.9.2 | https://github.com/stardist/stardist |
| superqt | 0.9.1 | https://github.com/pyapp-kit/superqt |
| tensorflow | 2.21.0 | https://www.tensorflow.org/ |
| termcolor | 3.3.0 | https://github.com/termcolor/termcolor |
| tf2onnx | 1.17.0 | https://github.com/onnx/tensorflow-onnx |
| threadpoolctl | 3.8.0 | https://github.com/joblib/threadpoolctl |
| tifffile | 2026.11.8 | https://www.cgohlke.com |
| tomli_w | 1.2.0 | https://github.com/hukkin/tomli-w |
| toolz | 1.2.0 | https://github.com/pytoolz/toolz |
| tqdm | 4.70.1 | https://github.com/tqdm/tqdm |
| typer | 0.29.1 | https://github.com/fastapi/typer |
| typing-inspection | 0.4.5 | https://github.com/pydantic/typing-inspection |
| typing_extensions | 4.16.0 | https://github.com/python/typing_extensions |
| tzdata | 2026.6 | https://github.com/python/tzdata |
| urllib3 | 1.26.6 | https://github.com/urllib3/urllib3 |
| vispy | 0.17.0 | http://vispy.org |
| wrapt | 2.5.2 | https://github.com/GrahamDumpleton/wrapt |
| zarr | 3.4.2 | https://github.com/zarr-developers/zarr-python |
| zarr-metadata | 0.6.0 | https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata |

## 本软件自己的代码

`cfos/` 文件夹中的全部代码、`packaging/` 中的打包脚本和 `resources/teaching/` 中的教学示意图由本项目编写。`resources/demo/` 中的示例脑片由张金铭实验室提供，仅供学习本软件使用。

联系：张金铭，zjm199502@163.com，https://github.com/zjm199502/cfos-brain-slice
