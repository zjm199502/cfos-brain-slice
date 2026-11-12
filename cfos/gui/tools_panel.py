"""中文图层工具面板：替代 napari 自带的英文图层控制面板，只保留新手需要的操作。"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class LayerTools(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        L = QVBoxLayout(self); L.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("当前图层：—"); self.title.setWordWrap(True)
        L.addWidget(self.title)
        b = QPushButton("显示整张图"); b.clicked.connect(lambda: self.viewer.reset_view()); L.addWidget(b)
        b = QPushButton("拖动看图（平移 / 滚轮缩放）"); b.clicked.connect(lambda: self._mode("pan_zoom")); L.addWidget(b)
        self.grid = QGridLayout(); L.addLayout(self.grid)
        L.addWidget(QLabel("不透明度"))
        self.opacity = QSlider(Qt.Horizontal); self.opacity.setRange(0, 100)
        self.opacity.valueChanged.connect(self._set_opacity); L.addWidget(self.opacity)
        self.help = QLabel(""); self.help.setWordWrap(True); self.help.setStyleSheet("color:#aab4c0;")
        L.addWidget(self.help)
        L.addStretch(1)
        viewer.layers.selection.events.active.connect(lambda e: self.refresh())
        self.refresh()

    def _layer(self):
        return self.viewer.layers.selection.active

    def _mode(self, m):
        layer = self._layer()
        if layer is not None:
            try:
                layer.mode = m
            except Exception:  # noqa: BLE001
                pass

    def _set_opacity(self, v):
        layer = self._layer()
        if layer is not None:
            layer.opacity = v / 100.0

    def _delete_selected(self):
        layer = self._layer()
        if layer is not None and hasattr(layer, "remove_selected"):
            layer.remove_selected()

    def refresh(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            wd = it.widget()
            if wd is not None:
                wd.hide(); wd.setParent(None); wd.deleteLater()
        layer = self._layer()
        if layer is None:
            self.title.setText("当前图层：—（在下方图层列表中点选一个图层）")
            self.help.setText("")
            return
        self.title.setText(f"当前图层：{layer.name}")
        self.opacity.blockSignals(True); self.opacity.setValue(int(layer.opacity * 100)); self.opacity.blockSignals(False)
        kind = type(layer).__name__
        buttons = []
        if kind == "Shapes" and layer.name in ("排除区域", "自定义区域"):
            buttons = [("画多边形", lambda: self._mode("add_polygon")),
                       ("选择 / 移动", lambda: self._mode("select")),
                       ("拖动顶点修改", lambda: self._mode("direct")),
                       ("删除选中的区域", self._delete_selected)]
            self.help.setText("画多边形：在图上逐个点击顶点，双击结束。画完到右侧点“保存区域”。")
        elif kind == "Shapes":
            buttons = [("选择 / 移动方框", lambda: self._mode("select")),
                       ("拖动顶点调整大小", lambda: self._mode("direct")),
                       ("画新的矩形", lambda: self._mode("add_rectangle")),
                       ("删除选中的方框", self._delete_selected)]
            self.help.setText("先点“选择 / 移动方框”再点击方框即可选中；按住拖动可移动。")
        elif kind == "Labels":
            def new_cell():
                layer.selected_label = int(layer.data.max()) + 1
                self._mode("paint")

            def delete_cell():
                layer.selected_label = 0
                self._mode("fill")

            def brush(d):
                layer.brush_size = max(1, layer.brush_size + d)
                self.help.setText(f"画笔大小：{layer.brush_size}")
            buttons = [("画新细胞（M 键）", new_cell), ("继续画当前细胞", lambda: self._mode("paint")),
                       ("橡皮擦", lambda: self._mode("erase")), ("删除点中的细胞", delete_cell),
                       ("画笔变大", lambda: brush(2)), ("画笔变小", lambda: brush(-2)),
                       ("拖动看图", lambda: self._mode("pan_zoom"))]
            self.help.setText("每个细胞一个颜色。画新细胞前按 M 键（或点“画新细胞”）换一个新颜色；"
                              "“删除点中的细胞”后点击某个细胞即可整个去掉。")
        elif kind == "Points":
            buttons = [("添加点（在图上点击）", lambda: self._mode("add")),
                       ("选择 / 拖动点", lambda: self._mode("select")),
                       ("删除选中的点", self._delete_selected)]
            self.help.setText("添加完点后，切换到“选择 / 拖动点”再拖动。")
        else:
            self.help.setText("图像图层的亮度和对比度请在“③ 亮度 / 对比度”中调整。")
        for i, (t, fn) in enumerate(buttons):
            b = QPushButton(t); b.clicked.connect(fn)
            self.grid.addWidget(b, i // 2, i % 2)
