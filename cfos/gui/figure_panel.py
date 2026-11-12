"""⑨ 截图制图：在图上画截图框（ROI），放大裁剪、叠加比例尺后导出成论文用的图片。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..core import figure as fig

LAYER = "截图框"
LAYER_COLOR = "#2ee6d6"


def _pixmap(img) -> QPixmap:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    pm = QPixmap(); pm.loadFromData(buf.getvalue(), "PNG")
    return pm


class FigureMixin:
    # ---------------------------------------------------------------- 面板
    def _panel_figure(self):
        from .main_window import hint, big_button
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("把图上感兴趣的区域（ROI）放大截图，加上比例尺，导出成可以直接放进论文或 PPT 的图片。\n"
                         "1. 点“画新的截图框”，在图上拖出一个矩形（可以画多个），点“保存截图框”。\n"
                         "2. 在下面选择要显示的通道、叠加内容和比例尺样式，点“预览”看效果。\n"
                         "3. 点“导出”。图片保存在项目文件夹的“截图”子文件夹里，每张图附带一个参数文件。"))

        # ---- 截图框
        g = QGroupBox("截图框（ROI）"); gl = QVBoxLayout(g)
        b = big_button("画新的截图框（在图上拖出矩形）"); b.clicked.connect(self.fig_draw_mode); gl.addWidget(b)
        row = QHBoxLayout()
        self.fig_fixed = QCheckBox("固定大小（µm）"); self.fig_fixed.setToolTip(
            "勾选后，保存时每个截图框都会以框中心为准调整成这个大小，方便多张图之间比较。")
        self.fig_fw = QDoubleSpinBox(); self.fig_fw.setRange(10, 20000); self.fig_fw.setDecimals(0); self.fig_fw.setPrefix("宽 ")
        self.fig_fh = QDoubleSpinBox(); self.fig_fh.setRange(10, 20000); self.fig_fh.setDecimals(0); self.fig_fh.setPrefix("高 ")
        row.addWidget(self.fig_fixed); row.addWidget(self.fig_fw); row.addWidget(self.fig_fh); gl.addLayout(row)
        b = big_button("保存截图框（画完 / 调整完后点）"); b.clicked.connect(self.fig_save_rois); gl.addWidget(b)
        self.fig_list = QListWidget(); self.fig_list.setMaximumHeight(130)
        self.fig_list.setToolTip("双击名称可以改名；双击左边的编号放大到这个框。")
        self.fig_list.itemChanged.connect(self._fig_rename)
        self.fig_list.itemDoubleClicked.connect(self._fig_zoom)
        gl.addWidget(self.fig_list)
        grid = QGridLayout()
        b = QPushButton("放大到选中的框"); b.clicked.connect(lambda: self._fig_zoom(self.fig_list.currentItem())); grid.addWidget(b, 0, 0)
        b = QPushButton("删除选中的框"); b.clicked.connect(self.fig_delete_roi); grid.addWidget(b, 0, 1)
        b = QPushButton("改名"); b.clicked.connect(lambda: self.fig_list.currentItem() and self.fig_list.editItem(self.fig_list.currentItem())); grid.addWidget(b, 1, 0)
        b = QPushButton("复制这些框到所有图片"); b.setToolTip("按相对于脑片的位置复制：其他图片上的每张脑片，在同样的相对位置各加一个同样大小的框。")
        b.clicked.connect(self.fig_copy_rois); grid.addWidget(b, 1, 1)
        gl.addLayout(grid)
        L.addWidget(g)

        # ---- 显示内容
        g = QGroupBox("显示内容"); gl = QVBoxLayout(g)
        self.fig_ch_box = QHBoxLayout(); gl.addLayout(self.fig_ch_box)
        self.fig_ch_cbs: list[QCheckBox] = []
        self.fig_separate = QCheckBox("合并图之外，每个通道再单独导出一张")
        self.fig_cells = QCheckBox("叠加识别出的细胞轮廓")
        self.fig_regions = QCheckBox("叠加脑区边界（需要先做 ⑥ 配准）")
        self.fig_region_names = QCheckBox("标注脑区缩写")
        self.fig_excl = QCheckBox("显示排除区域")
        self.fig_inset = QCheckBox("左上角加定位小图（整张脑片 + 截图框位置）")
        for cb in (self.fig_separate, self.fig_cells, self.fig_regions, self.fig_region_names, self.fig_excl, self.fig_inset):
            gl.addWidget(cb)
        gl.addWidget(QLabel("亮度 / 对比度沿用“③ 亮度 / 对比度”里的设置。"))
        L.addWidget(g)

        # ---- 比例尺
        g = QGroupBox("比例尺"); f = QFormLayout(g)
        self.sb_len = QComboBox(); self.sb_len.setEditable(True)
        self.sb_len.addItem("自动", 0.0)
        for v in fig.NICE_UM:
            self.sb_len.addItem(f"{v:g} µm", float(v))
        self.sb_len.setToolTip("可以直接输入数字（µm）。自动 = 大约图宽的 1/5，取整到 50 / 100 / 200 / 500 µm。")
        self.sb_thick = QSpinBox(); self.sb_thick.setRange(0, 200); self.sb_thick.setSpecialValueText("自动"); self.sb_thick.setSuffix(" 像素")
        self.sb_color = QComboBox(); self.sb_color.addItems(["白", "黑"])
        self.sb_pos = QComboBox(); self.sb_pos.addItems(fig.POSITIONS)
        self.sb_text = QCheckBox("在比例尺上方标注长度（如 200 µm）"); self.sb_text.setChecked(True)
        self.sb_font = QSpinBox(); self.sb_font.setRange(0, 400); self.sb_font.setSpecialValueText("自动"); self.sb_font.setSuffix(" 像素")
        f.addRow("长度", self.sb_len); f.addRow("粗细", self.sb_thick); f.addRow("颜色", self.sb_color)
        f.addRow("位置", self.sb_pos); f.addRow(self.sb_text); f.addRow("字号", self.sb_font)
        L.addWidget(g)

        # ---- 导出
        g = QGroupBox("导出"); gl = QVBoxLayout(g)
        r = QHBoxLayout(); r.addWidget(QLabel("分辨率"))
        self.fig_res = QComboBox()
        for v, t in fig.RES_CHOICES:
            self.fig_res.addItem(t, v)
        r.addWidget(self.fig_res, 1); gl.addLayout(r)
        gl.addWidget(QLabel("格式（可多选）"))
        self.fig_png = QCheckBox("PNG（通用，PPT 用）"); self.fig_png.setChecked(True)
        self.fig_tif = QCheckBox("TIFF（无损，投稿用）"); self.fig_svg = QCheckBox("SVG（比例尺可在 Illustrator 中编辑）")
        for cb in (self.fig_png, self.fig_tif, self.fig_svg):
            gl.addWidget(cb)
        b = big_button("预览选中的框"); b.clicked.connect(self.fig_preview); gl.addWidget(b)
        b = big_button("导出当前图片的所有截图框"); b.clicked.connect(lambda: self.fig_export(False)); gl.addWidget(b)
        b = big_button("导出所有图片的所有截图框"); b.clicked.connect(lambda: self.fig_export(True)); gl.addWidget(b)
        b = QPushButton("打开“截图”文件夹"); b.clicked.connect(
            lambda: self.project and self._open_folder(self._fig_out_dir(True))); gl.addWidget(b)
        L.addWidget(g)
        L.addStretch(1)

        self._fig_setting_widgets = [self.fig_fixed, self.fig_fw, self.fig_fh, self.fig_separate, self.fig_cells,
                                     self.fig_regions, self.fig_region_names, self.fig_excl, self.fig_inset, self.sb_len,
                                     self.sb_thick, self.sb_color, self.sb_pos, self.sb_text, self.sb_font, self.fig_res,
                                     self.fig_png, self.fig_tif, self.fig_svg]
        self._fig_connect_settings(self._fig_setting_widgets)
        self._fig_loading = False
        return w

    def _fig_connect_settings(self, widgets):
        for wd in widgets:
            if isinstance(wd, QCheckBox):
                wd.toggled.connect(self._fig_settings_changed)
            elif isinstance(wd, QComboBox):
                wd.currentIndexChanged.connect(self._fig_settings_changed)
                if wd.isEditable():
                    wd.editTextChanged.connect(self._fig_settings_changed)
            else:
                wd.valueChanged.connect(self._fig_settings_changed)

    def _fig_out_dir(self, create=False) -> Path:
        d = self.project.folder / fig.OUT_DIR
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------------- 设置 <-> 界面
    def _sync_figure_panel(self):
        if self.project is None:
            return
        self._fig_loading = True
        try:
            st = fig.FigureSettings.from_dict(self.project.figure)
            while self.fig_ch_box.count():
                it = self.fig_ch_box.takeAt(0)
                wd = it.widget()
                if wd is not None:
                    wd.hide(); wd.setParent(None); wd.deleteLater()
            self.fig_ch_cbs = []
            n_ch = len(self.slide.channels) if self.slide else len(self.project.channel_names)
            self.fig_ch_box.addWidget(QLabel("通道："))
            for c in range(n_ch):
                cb = QCheckBox(self.ch_name(c)); cb.setChecked(not st.channels or c in st.channels)
                cb.toggled.connect(self._fig_settings_changed)
                self.fig_ch_box.addWidget(cb); self.fig_ch_cbs.append(cb)
            self.fig_ch_box.addStretch(1)
            self.fig_fixed.setChecked(st.fixed_size); self.fig_fw.setValue(st.fixed_w_um); self.fig_fh.setValue(st.fixed_h_um)
            self.fig_separate.setChecked(st.separate); self.fig_cells.setChecked(st.overlay_cells)
            self.fig_regions.setChecked(st.overlay_regions); self.fig_region_names.setChecked(st.overlay_region_names)
            self.fig_excl.setChecked(st.overlay_exclusions); self.fig_inset.setChecked(st.inset)
            sb = st.scale
            idx = self.sb_len.findData(float(sb.length_um))
            if idx >= 0:
                self.sb_len.setCurrentIndex(idx)
            else:
                self.sb_len.setEditText(f"{sb.length_um:g}")
            self.sb_thick.setValue(int(sb.thickness_px)); self.sb_color.setCurrentText(sb.color)
            self.sb_pos.setCurrentText(sb.position if sb.position in fig.POSITIONS else fig.POSITIONS[0])
            self.sb_text.setChecked(sb.show_text); self.sb_font.setValue(int(sb.font_px))
            idx = self.fig_res.findData(float(st.pixel_um)); self.fig_res.setCurrentIndex(max(idx, 0))
            self.fig_png.setChecked("png" in st.formats); self.fig_tif.setChecked("tif" in st.formats)
            self.fig_svg.setChecked("svg" in st.formats)
        finally:
            self._fig_loading = False
        self._refresh_fig_list()
        self._draw_figure_rois(force=True)

    def _fig_settings(self) -> fig.FigureSettings:
        st = fig.FigureSettings()
        chans = [c for c, cb in enumerate(self.fig_ch_cbs) if cb.isChecked()]
        st.channels = [] if len(chans) == len(self.fig_ch_cbs) else chans
        st.fixed_size = self.fig_fixed.isChecked(); st.fixed_w_um = self.fig_fw.value(); st.fixed_h_um = self.fig_fh.value()
        st.separate = self.fig_separate.isChecked(); st.overlay_cells = self.fig_cells.isChecked()
        st.overlay_regions = self.fig_regions.isChecked(); st.overlay_region_names = self.fig_region_names.isChecked()
        st.overlay_exclusions = self.fig_excl.isChecked(); st.inset = self.fig_inset.isChecked()
        txt = self.sb_len.currentText().replace("µm", "").replace("um", "").strip()
        try:
            length = 0.0 if txt in ("", "自动") else float(txt)
        except ValueError:
            length = 0.0
        st.scale = fig.ScaleBar(length_um=max(0.0, length), thickness_px=self.sb_thick.value(),
                                color=self.sb_color.currentText(), position=self.sb_pos.currentText(),
                                show_text=self.sb_text.isChecked(), font_px=self.sb_font.value())
        st.pixel_um = float(self.fig_res.currentData() or 0.0)
        st.formats = [t for t, cb in (("png", self.fig_png), ("tif", self.fig_tif), ("svg", self.fig_svg)) if cb.isChecked()] or ["png"]
        return st

    def _fig_settings_changed(self, *_):
        if getattr(self, "_fig_loading", False) or self.project is None:
            return
        self.project.figure = self._fig_settings().to_dict()
        self._save_later()

    # ---------------------------------------------------------------- 截图框图层
    def _draw_figure_rois(self, force=False):
        if LAYER in self.viewer.layers:
            self.viewer.layers.remove(LAYER)
        if self.image is None or self.slide is None:
            return
        rois = self.image.figure_rois
        if not rois and not force:
            return
        rects = [np.array([[r["y"], r["x"]], [r["y"], r["x"] + r["w"]], [r["y"] + r["h"], r["x"] + r["w"]],
                           [r["y"] + r["h"], r["x"]]], float) for r in rois]
        lw = max(self.slide.level_shapes[0]) / 400
        kw = dict(name=LAYER, edge_color=LAYER_COLOR, face_color="transparent", edge_width=lw,
                  text={"string": "{name}", "color": LAYER_COLOR, "size": 14, "anchor": "upper_left"})
        if rects:
            layer = self.viewer.add_shapes(rects, shape_type="rectangle", features={"name": [r["name"] for r in rois]}, **kw)
        else:
            layer = self.viewer.add_shapes(ndim=2, **kw)
        try:
            layer.feature_defaults["name"] = ""   # 新画的框先没有名字，保存时自动编号
        except Exception:  # noqa: BLE001
            pass
        layer.current_edge_color = LAYER_COLOR; layer.current_face_color = "transparent"
        layer.current_edge_width = lw
        return layer

    def fig_draw_mode(self):
        if self.slide is None:
            return
        layer = self.viewer.layers[LAYER] if LAYER in self.viewer.layers else self._draw_figure_rois(force=True)
        self.viewer.layers.selection.active = layer
        layer.mode = "add_rectangle"
        msg = "在图上按住鼠标拖出一个矩形；画完后点“保存截图框”。"
        if self.fig_fixed.isChecked():
            msg += f"（已勾选固定大小：保存时会以框中心为准调整为 {self.fig_fw.value():g} × {self.fig_fh.value():g} µm）"
        self.statusBar().showMessage(msg, 15000)

    def fig_save_rois(self):
        if self.image is None or LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[LAYER]
        old = list(self.image.figure_rois)
        px = self.slide.pixel_size_um
        H0, W0 = self.slide.level_shapes[0][:2]
        new, used = [], set()
        for i, poly in enumerate(layer.data):
            ys, xs = poly[:, -2], poly[:, -1]
            x, y, w, h = float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())
            if self.fig_fixed.isChecked():
                fw, fh = self.fig_fw.value() / px, self.fig_fh.value() / px
                x, y, w, h = x + w / 2 - fw / 2, y + h / 2 - fh / 2, fw, fh
            x, y = min(max(0.0, x), W0 - 1), min(max(0.0, y), H0 - 1)
            w, h = min(w, W0 - x), min(h, H0 - y)
            if w < 5 or h < 5:
                continue
            prev = old[i] if i < len(old) else None
            name = prev["name"] if prev else ""
            new.append({"name": name, "x": x, "y": y, "w": w, "h": h})
            if name:
                used.add(name)
        k = len(used) + 1   # 新框从已有数量 +1 开始编号（改过名的框不会让编号重复使用）
        for r in new:
            if not r["name"]:
                while f"ROI{k}" in used:
                    k += 1
                r["name"] = f"ROI{k}"; used.add(r["name"])
        self.image.figure_rois = new
        self.project.save()
        self._draw_figure_rois(force=True)
        self._refresh_fig_list()
        self.statusBar().showMessage(f"已保存 {len(new)} 个截图框。", 5000)

    def _refresh_fig_list(self):
        self.fig_list.blockSignals(True)
        self.fig_list.clear()
        px = self.slide.pixel_size_um if self.slide else 0.25
        for r in (self.image.figure_rois if self.image else []):
            it = QListWidgetItem(r["name"])
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            it.setToolTip(f"{r['w'] * px:.0f} × {r['h'] * px:.0f} µm")
            self.fig_list.addItem(it)
        self.fig_list.blockSignals(False)

    def _fig_rename(self, it):
        i = self.fig_list.row(it)
        if self.image is None or not (0 <= i < len(self.image.figure_rois)):
            return
        name = it.text().strip() or self.image.figure_rois[i]["name"]
        self.image.figure_rois[i]["name"] = name
        self.fig_list.blockSignals(True); it.setText(name); self.fig_list.blockSignals(False)
        self.project.save(); self._draw_figure_rois()

    def _fig_zoom(self, it):
        if it is None or self.image is None:
            return
        i = self.fig_list.row(it)
        if not (0 <= i < len(self.image.figure_rois)):
            return
        r = self.image.figure_rois[i]
        self.viewer.camera.center = (r["y"] + r["h"] / 2, r["x"] + r["w"] / 2)
        sz = (self.qt_viewer.height(), self.qt_viewer.width())
        self.viewer.camera.zoom = 0.8 * min(sz[0] / r["h"], sz[1] / r["w"])

    def fig_delete_roi(self):
        i = self.fig_list.currentRow()
        if self.image is None or not (0 <= i < len(self.image.figure_rois)):
            return
        del self.image.figure_rois[i]
        self.project.save(); self._draw_figure_rois(force=True); self._refresh_fig_list()

    def fig_copy_rois(self):
        if self.image is None or not self.image.figure_rois:
            self.info("当前图片还没有截图框。"); return
        n = 0
        src = self.image
        for r in src.figure_rois:
            s = fig.section_for(src, r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
            if s is None:
                continue
            sx, sy, sw, sh = s.bbox0
            fx, fy = (r["x"] + r["w"] / 2 - sx) / sw, (r["y"] + r["h"] / 2 - sy) / sh
            for im in self.project.images:
                for t in im.sections:
                    if im is src and t is s:
                        continue
                    tx, ty, tw, th = t.bbox0
                    cx, cy = tx + fx * tw, ty + fy * th
                    name = f"{r['name']}_{t.name}" if im is src else r["name"] + ("" if len(im.sections) == 1 else f"_{t.name}")
                    if any(q["name"] == name for q in im.figure_rois):
                        continue
                    im.figure_rois.append({"name": name, "x": cx - r["w"] / 2, "y": cy - r["h"] / 2, "w": r["w"], "h": r["h"]})
                    n += 1
        self.project.save(); self._draw_figure_rois(force=True); self._refresh_fig_list()
        self.info(f"已按相对位置复制，新增 {n} 个截图框。请到各张图片上检查位置是否合适（不合适可以拖动调整后再保存）。")

    # ---------------------------------------------------------------- 预览与导出
    def fig_preview(self):
        if self.image is None or self.slide is None:
            return
        i = self.fig_list.currentRow()
        if i < 0 and self.image.figure_rois:
            i = 0
        if not (0 <= i < len(self.image.figure_rois)):
            self.info("请先画一个截图框并保存。"); return
        st = self._fig_settings()
        roi = self.image.figure_rois[i]
        try:
            df = self.project.load_cells(self.image) if st.overlay_cells else None
            img, meta = fig.render_roi(self.project, self.image, self.slide, roi, st, df=df, max_side=1400)
        except Exception as e:  # noqa: BLE001
            self.error(f"预览失败：{e}"); return
        dlg = QDialog(self); dlg.setWindowTitle(f"预览：{roi['name']}"); dlg.resize(960, 760)
        L = QVBoxLayout(dlg)
        pm = _pixmap(img)
        if pm.width() > 920 or pm.height() > 660:
            pm = pm.scaled(920, 660, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lb = QLabel(); lb.setPixmap(pm); lb.setAlignment(Qt.AlignCenter); L.addWidget(lb, 1)
        sb = meta.get("scale_bar", {})
        x, y, w, h = fig.clamp_roi(roi, self.slide.level_shapes[0])
        ds = self.slide.downsample(fig.choose_level(self.slide, w, h, st))
        L.addWidget(QLabel(f"预览已缩小显示。实际导出约 {int(w / ds)} × {int(h / ds)} 像素"
                           f"（{self.slide.pixel_size_um * ds:.3g} µm/像素，{w * self.slide.pixel_size_um:.0f} × "
                           f"{h * self.slide.pixel_size_um:.0f} µm）；比例尺 {sb.get('length_um', 0):g} µm。"))
        b = QPushButton("关闭"); b.clicked.connect(dlg.accept); L.addWidget(b)
        dlg.show(); self._fig_dialog = dlg

    def fig_export(self, all_images: bool):
        if self.project is None or self.image is None:
            return
        images = list(self.project.images) if all_images else [self.image]
        n = sum(len(im.figure_rois) for im in images)
        if n == 0:
            self.info("还没有截图框。先点“画新的截图框”在图上画，再点“保存截图框”。"); return
        st = self._fig_settings()
        self.project.figure = st.to_dict(); self.project.save()
        project = self.project
        out_dir = self._fig_out_dir(True)

        def done(res):
            files, errors = res
            msg = f"已导出 {len(files)} 个文件到：{out_dir}"
            if errors:
                msg += "\n\n以下截图框出错：\n" + "\n".join(errors[:10])
            self.info(msg)
            if files:
                self._open_folder(out_dir)
            self._refresh_step_marks()

        self.run_task(lambda task: fig.export_images(project, images, st, progress=task.report,
                                                     cancel=task.is_cancelled, out_dir=out_dir),
                      done, label="正在导出截图…")
