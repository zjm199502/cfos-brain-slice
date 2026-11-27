"""Interactive selection and export of individual scanner-coordinate inference tiles."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox

from ..core import detection as det
from ..core.coloc import colocalize, nucleus_channel
from ..core.field_review import effective_settings
from ..core.pipeline import drop_excluded
from ..core.project import processed_section_names
from ..core.unmix import prepare
from .cell_rings import add_cell_layer


def tile_data(frame, image, section, number, bounds, names, channels):
    l, t, r, b = bounds
    cells = frame[(frame.x >= l) & (frame.x < r) & (frame.y >= t) & (frame.y < b)].copy()
    cells.attrs.clear()
    cells["图块"] = number
    row = {"图片": image.display_name, "脑片": section, "图块": number,
           "左 X (px)": l, "上 Y (px)": t, "右 X (px)": r, "下 Y (px)": b}
    for channel in channels:
        row[f"{names[channel]} 检出数"] = int((cells.channel == channel).sum())
    return row, cells


class TileSelectionMixin:
    def _rotation_controls(self):
        row = QHBoxLayout()
        self.view_move = QPushButton("拖动脑片")
        self.view_move.setCheckable(True)
        self.view_move.setToolTip("拖框内移动脑片和标注，拖圆柄旋转视图。Esc 返回看图。")
        self.view_move.clicked.connect(lambda on: self.canvas_tools.begin("view") if on else self.canvas_tools.reset())
        row.addWidget(self.view_move)
        row.addStretch()
        row.addWidget(QLabel("预览旋转"))
        self.view_angle = QDoubleSpinBox()
        self.view_angle.setDecimals(1)
        self.view_angle.setRange(-180, 180)
        self.view_angle.setSuffix("°")
        self.view_angle.valueChanged.connect(self._rotate_preview)
        row.addWidget(self.view_angle)
        reset = QPushButton("复位")
        reset.clicked.connect(lambda: self.view_angle.setValue(0))
        row.addWidget(reset)
        self.tile_overlay.resized.connect(self._rotate_preview)
        return row

    def _rotate_preview(self, *_):
        from vispy.visuals.transforms import MatrixTransform
        view = self.qt_viewer.canvas.view
        angle = self.view_angle.value()
        transform = None
        if angle:
            cx, cy = np.asarray(view.size)/2
            transform = MatrixTransform()
            transform.translate((-cx, -cy, 0))
            transform.rotate(angle, (0, 0, 1))
            transform.translate((cx, cy, 0))
        view.camera.pre_transform = transform
        self.tile_overlay.update()
        if hasattr(self, "canvas_tools"):
            self.canvas_tools.update()

    def _load_coloc_controls(self):
        self._loading_coloc = True
        try:
            params = self.project.coloc
            self.coloc_dist.setValue(params.max_dist_um)
            self.coloc_contour.setChecked(params.use_contour)
            self.coloc_nuc_int.setChecked(params.nucleus_by_intensity)
            self.coloc_req_nuc.setChecked(params.require_nucleus)
        finally:
            self._loading_coloc = False

    def _tile_controls(self, layout):
        self.pick_tiles = QPushButton("鼠标点选图块")
        self.pick_tiles.setCheckable(True)
        self.pick_tiles.toggled.connect(self._toggle_tile_pick)
        self.tile_count = QLabel("已选 0 块")
        clear = QPushButton("清空选择")
        clear.clicked.connect(self._reset_tile_selection)
        row = QHBoxLayout()
        for widget in (self.pick_tiles, self.tile_count, clear):
            row.addWidget(widget)
        layout.addLayout(row)
        self.run_tiles = QPushButton("识别所选图块")
        self.run_tiles.clicked.connect(self._run_selected_tiles)
        self.export_tiles = QPushButton("导出所选图块 Excel…")
        self.export_tiles.clicked.connect(self._export_selected_tiles)
        for button in (self.run_tiles, self.export_tiles):
            button.setEnabled(False)
            layout.addWidget(button)

    def _toggle_tile_pick(self, active):
        if active:
            self.canvas_tools.reset()
            self._preview_grid = None
            self.show_grid.setChecked(True)
            if self._layout_mode == "single":
                self.pane_button.setChecked(True)
        self._update_tile_grid()

    def _reset_tile_selection(self):
        self._tile_choices.clear()
        self._tile_boxes.clear()
        self._tile_selection_changed()
        self.pick_tiles.setChecked(False)
        self._update_tile_grid()

    def _sync_tile_boxes(self, boxes):
        remaining = self._tile_choices.intersection(boxes)
        if remaining != self._tile_choices or any(
                self._tile_boxes[key] is not boxes[key] for key in remaining):
            self._tile_choices = remaining
            self._tile_selection_changed()
        self._tile_boxes = boxes

    def _tile_clicked(self, x, y):
        if self.task and self.task.isRunning():
            return
        for key in self._tile_boxes:
            name, left, top, right, bottom = key
            if left <= x < right and top <= y < bottom:
                self._tile_choices.symmetric_difference_update({key})
                self._tile_selection_changed()
                break

    def _tile_selection_changed(self):
        self._tile_result = None
        self.roi_counter.schedule()
        self.tile_overlay.selected = [key[1:] for key in self._tile_choices]
        self.tile_overlay.update()
        self.tile_count.setText(f"已选 {len(self._tile_choices)} 块")
        self.run_tiles.setEnabled(bool(self._tile_choices))
        self.export_tiles.setEnabled(bool(self._tile_choices and self.project and self.image and
                                         self.project.has_cells(self.image)))
        for layer in list(self.viewer.layers):
            if layer.name.startswith("选块识别-"):
                self.viewer.layers.remove(layer)

    def _run_selected_tiles(self):
        if not self._require_scope() or not self._tile_choices:
            return
        project, image = self.project, self.image
        selected = sorted(self._tile_choices)
        sections = {key[0]: self._tile_boxes[key] for key in selected}
        names = list(project.channel_names)
        coloc_params = deepcopy(project.coloc)
        self.pick_tiles.setChecked(False)
        self._tile_selection_changed()

        def job(task):
            slide = image.open_slide(z_mode=project.z_mode)
            groups = defaultdict(list)
            rows, frames = [], []
            for number, (section, *box) in enumerate(selected, 1):
                groups[section].append((number, box))
            try:
                prepare(project, slide, image)
                for index, (name, tiles) in enumerate(groups.items()):
                    if task.is_cancelled():
                        raise InterruptedError
                    task.report(index, len(groups), f"正在识别 {name} 的所选图块")
                    section = sections[name]
                    settings = effective_settings(project.detection, image, section, project)
                    df = det.detect_section(slide, section, settings, cancel=task.is_cancelled,
                            detail=task.detection_event.emit, channel_names=names, image_name=image.display_name,
                            selected_boxes=[box for _, box in tiles])
                    df = drop_excluded(df, image)
                    df.attrs.clear()
                    for number, (l,t,r,b) in tiles:
                        active = [c for c, params in settings.channels.items() if params.detect]
                        row, cells = tile_data(df, image, name, number, (l,t,r,b), names, active)
                        rows.append(row)
                        frames.append(cells)
                cells = colocalize(pd.concat(frames, ignore_index=True), slide.pixel_size_um,
                                   coloc_params, nucleus_channel(names))
                cells["cell_id"] = np.arange(len(cells))
                cells["通道名称"] = cells.channel.map(dict(enumerate(names)))
                return pd.DataFrame(rows), cells
            finally:
                slide.close()

        def done(result):
            if self.image is not image or sorted(self._tile_choices) != selected:
                return
            self._tile_selection_changed()
            self._tile_result = result
            self.roi_counter.source.setCurrentIndex(2)
            self.roi_counter.schedule()
            cells = result[1]
            for c, name in enumerate(names):
                found = cells[cells.channel == c]
                if len(found):
                    add_cell_layer(self, found, c, "选块识别-"+name, editable=False)
            self.export_tiles.setEnabled(True)
            self.det_summary.setText(f"所选 {len(selected)} 块共检出 {len(cells)} 个通道对象，可导出 Excel。")
        self.run_task(job, done, label="正在识别所选图块…", detection=True)

    def _export_selected_tiles(self):
        result = self._tile_result
        source = "所选图块识别"
        if result is None:
            if not self._tile_choices or self.image is None:
                self.info("请先在④点选要导出的图块。")
                return
            saved = self.cells_for(self.image)
            if not {key[0] for key in self._tile_choices} <= processed_section_names(self.image, saved):
                self.info("所选图块中有尚无识别结果的脑片，请先点击“识别所选图块”。")
                return
            names = self.project.channel_names
            source = "项目已保存的识别结果"
            rows, frames = [], []
            for number, (section, *bounds) in enumerate(sorted(self._tile_choices), 1):
                frame = drop_excluded(saved[saved.section == section], self.image)
                models = saved.attrs.get("section_models", {}).get(section, {})
                channels = sorted({int(c) for c in models} | set(frame.channel.astype(int)))
                row, cells = tile_data(frame, self.image, section, number, bounds, names, channels)
                rows.append(row)
                frames.append(cells)
            detail = pd.concat(frames, ignore_index=True)
            detail["通道名称"] = detail.channel.map(dict(enumerate(names)))
            result = pd.DataFrame(rows), detail
        path, _ = QFileDialog.getSaveFileName(self, "导出所选图块", str(self.project.folder /
                    f"{self.image.key}_所选图块.xlsx"), "Excel (*.xlsx)")
        if not path:
            return
        path = Path(path).with_suffix(".xlsx")
        summary, cells = result
        summary = summary.assign(数据来源=source)
        detail = cells.drop(columns=["contour"], errors="ignore").copy()
        detail.insert(0, "图片", self.image.display_name)
        try:
            with tempfile.TemporaryDirectory(dir=path.parent) as folder:
                output = Path(folder)/"tiles.xlsx"
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    summary.to_excel(writer, sheet_name="图块汇总", index=False)
                    detail.to_excel(writer, sheet_name="细胞明细", index=False)
                output.replace(path)
            self.statusBar().showMessage(f"已导出 {path}", 10000)
        except (OSError, ValueError) as exc:
            self.error(str(exc))
