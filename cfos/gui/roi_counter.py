from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from napari.layers import Shapes
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                            QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..core.roi_counts import count_rois, saved_coverage


class RoiShapes(Shapes):
    """Editable ROI polygons with a one-point outline in screen space."""

    def __init__(self, data, *, canvas_widget, zoom, **kwargs):
        self._logical_dpi = canvas_widget.logicalDpiX
        super().__init__(data, edge_width=self._logical_dpi() / (72 * zoom), **kwargs)

    @property
    def _type_string(self):
        return "shapes"

    def _update_draw(self, scale_factor, corner_pixels_displayed, shape_threshold):
        super()._update_draw(scale_factor, corner_pixels_displayed, shape_threshold)
        # Napari stores edge widths in image pixels. Pan and rotation leave the width alone.
        width = self._logical_dpi() / 72 * self._normalized_scale_factor
        if width != self.current_edge_width:
            self.edge_width = width
            self.current_edge_width = width

    def _outline_shapes(self):
        # The native selected/hovered outline is a separate screen-space stroke.
        self._highlight_width = self._logical_dpi() / 72
        return super()._outline_shapes()

    def _compute_vertices_and_box(self):
        vertices, face, edge, box, width = super()._compute_vertices_and_box()
        if self.mode == "add_polygon_lasso":
            # Dense vertex handles obscure the stroke while tracing. Direct edit retains them.
            vertices = vertices[:0] if vertices is not None else None
        return vertices, face, edge, box, width


class RoiCounter(QGroupBox):
    def __init__(self, window):
        super().__init__("ROI 细胞计数")
        self.window_ref = window
        layout = QVBoxLayout(self)
        self.source = QComboBox()
        self.source.addItems(["项目已保存的识别结果", "最近一次视野预览", "所选图块识别结果"])
        layout.addWidget(self.source)
        tip = QLabel("按住鼠标沿边缘拖动，松开闭合。按细胞中心计入各 ROI；同一细胞可计入多个重叠 ROI。合并细胞数按已有共标编号去重。点“保存区域”可随项目保留轮廓。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        buttons = QGridLayout()
        for index, (text, callback) in enumerate((
            ("自由圈选 ROI", lambda: self.edit("add_polygon_lasso")),
            ("调整轮廓", lambda: self.edit("direct")),
            ("删除选中 ROI", self.remove_selected),
            ("保存区域", window.save_polygons),
        )):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            buttons.addWidget(btn, index // 2, index % 2)
        layout.addLayout(buttons)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(105)
        layout.addWidget(self.table)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.export = QPushButton("导出当前 ROI 计数（Excel）")
        self.export.clicked.connect(self.export_counts)
        layout.addWidget(self.export)
        self.frame = pd.DataFrame()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.refresh)
        self.source.currentIndexChanged.connect(self.schedule)
        self.preview = None
        self.export.setEnabled(False)

    def schedule(self, *_):
        self.timer.start()

    def edit(self, mode):
        w = self.window_ref
        w.canvas_tools.reset()
        if w.slide is None:
            return
        if "自定义区域" not in w.viewer.layers:
            w._draw_polygons()
        if w.pick_tiles.isChecked():
            w.pick_tiles.setChecked(False)
        layer = w.viewer.layers["自定义区域"]
        layer.visible = True
        w.viewer.layers.selection.active = layer
        layer.mode = mode
        if w._layout_mode == "single":
            w.pane_button.setChecked(True)

    def remove_selected(self):
        w = self.window_ref
        if "自定义区域" in w.viewer.layers:
            w.viewer.layers["自定义区域"].remove_selected()
            self.schedule()

    def refresh(self):
        self.timer.stop()
        w = self.window_ref
        if w.stack.currentIndex() not in w.roi_slots:
            return
        rois = w._canvas_rois() if w.slide is not None and w.image is not None else []
        frame, coverage = None, {}
        source = self.source.currentIndex()
        if rois:
            if source == 0:
                frame = w.cells_for(w.image)
                coverage = saved_coverage(w.image, frame)
            elif source == 1 and self.preview is not None:
                frame, coverage = self.preview
            elif source == 2 and w._tile_result is not None:
                frame = w._tile_result[1]
                for section, x, y, r, b in w._tile_choices:
                    coverage.setdefault(section, []).append((x, y, r-x, b-y))
            enabled = {s.name for s in w.image.sections if s.enabled and w._section_in_scope(w.image, s)}
            if w.image.sections or source != 1:
                coverage = {k: v for k, v in coverage.items() if k in enabled}
        channels = [(i,n) for i,n in enumerate(w.project.channel_names) if i not in w.project.unused_channels] if w.project else []
        self.frame = count_rois(frame, rois, coverage, channels, w._canvas_exclusions())
        self.table.setRowCount(len(self.frame))
        self.table.setColumnCount(len(self.frame.columns))
        self.table.setHorizontalHeaderLabels(list(self.frame.columns))
        for i, row in enumerate(self.frame.itertuples(index=False, name=None)):
            for j, val in enumerate(row):
                text = "—" if pd.isna(val) else str(int(val)) if isinstance(val, (int,float,np.number)) else str(val)
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.export.setEnabled(bool(rois and coverage and frame is not None))
        self.status.setText("请圈选 ROI。" if not rois else "当前来源尚无可用识别结果。" if frame is None or not coverage
                            else "仅统计所选来源已识别范围内的细胞。")

    def export_counts(self):
        self.refresh()
        if not self.export.isEnabled():
            return
        w = self.window_ref
        table = self.frame.copy()
        table.insert(0, "识别来源", self.source.currentText())
        table.insert(0, "图片", w.image.key)
        filename, _ = QFileDialog.getSaveFileName(w, "导出 ROI 细胞计数", str(w.project.folder / "ROI细胞计数.xlsx"), "Excel (*.xlsx)")
        if not filename:
            return
        from ..core.export import _write_xlsx
        path = Path(filename).with_suffix(".xlsx")

        def write(task):
            with TemporaryDirectory(dir=path.parent) as folder:
                output = Path(folder) / "roi.xlsx"
                _write_xlsx({"ROI细胞计数": table}, output)
                if task.is_cancelled():
                    raise InterruptedError
                output.replace(path)

        w.run_task(write, lambda _: w.statusBar().showMessage(f"已导出 {path}", 7000),
                   label="导出 ROI 细胞计数")
