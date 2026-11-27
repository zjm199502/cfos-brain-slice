"""⑥ 脑图谱配准面板（作为 MainWindow 的一部分）。"""
from __future__ import annotations

from ..core.atlas_choice import has_registration
import numpy as np
from .layout import fit_window
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget, QSpinBox,
)

from ..core import registration as R
from ..core.atlas import load_atlas, region_boundaries
from ..core.atlas_contours import registration_paths
from ..core.edge_drag import deform_paths, is_fine, snap_path
from ..core.guided_edge import DOT_UM, contour_edits, fluorescence_edge
from ..core.coloc import nucleus_channel
from ..core.regions import _thumb_geometry
from .atlas_outline import set_outline_paths

# 滑块：(属性名, 显示名, 最小, 最大, 每格的值, 单位)
SLIDERS = [
    ("ap_shift", "前后位置", -560, 560, 1.0, "× 25 µm"),
    ("tilt_dv", "背腹倾斜", -150, 150, 0.1, "°"),
    ("tilt_ml", "左右倾斜", -150, 150, 0.1, "°"),
    ("scale_x", "宽度缩放", 70, 130, 0.01, ""),
    ("scale_y", "高度缩放", 70, 130, 0.01, ""),
    ("rotation", "旋转", -300, 300, 0.1, "°"),
    ("shift_x", "左右移动", -150, 150, 0.002, ""),
    ("shift_y", "上下移动", -150, 150, 0.002, ""),
]


def big_button(text):
    b = QPushButton(text)
    b.setMinimumHeight(32)
    return b


class RegistrationMixin:
    # ------------------------------------------------------------ 面板
    def _panel_registration(self):
        from .main_window import hint
        w = QWidget(); L = QVBoxLayout(w)
        from ..core.atlas_choice import CCF, STAM
        self.reg_atlas = QComboBox()
        self.reg_atlas.addItem("Allen CCFv3（双侧分区）", CCF)
        self.reg_atlas.addItem("STAM（官方单侧分区）", STAM)
        form = QFormLayout()
        form.addRow("配准图谱", self.reg_atlas)
        L.addLayout(form)
        self.reg_atlas.activated.connect(self._change_registration_atlas)
        L.addWidget(hint("giRAff 在所选图谱的冠状切面中自动定位。白线显示配准后的脑区边界。"))
        self.reg_nonlinear_default = QCheckBox("自动配准后做组织外轮廓非线性贴合")
        self.reg_nonlinear_default.setChecked(True)
        L.addWidget(self.reg_nonlinear_default)
        button = big_button("自动配准所选图片和脑片")
        button.clicked.connect(lambda: self._require_scope() and self.auto_register(True))
        L.addWidget(button)

        row = QHBoxLayout(); row.addWidget(QLabel("检查 / 调整脑片："))
        self.reg_sec_combo = QComboBox(); self.reg_sec_combo.currentIndexChanged.connect(self._reg_section_changed)
        row.addWidget(self.reg_sec_combo, 1); L.addLayout(row)
        self.reg_status = QLabel(""); self.reg_status.setWordWrap(True); L.addWidget(self.reg_status)
        self.reg_ok_btn = QPushButton("已检查，这张没问题（去掉 ⚠）"); self.reg_ok_btn.setVisible(False)
        self.reg_ok_btn.clicked.connect(self._reg_mark_checked); L.addWidget(self.reg_ok_btn)

        box = QGroupBox("整体调整"); f = QFormLayout(box)
        self.atlas_move = QPushButton("拖动整张图谱")
        self.atlas_move.setCheckable(True)
        self.atlas_move.clicked.connect(lambda on: self.canvas_tools.begin("atlas") if on else self.canvas_tools.reset())
        f.addRow(self.atlas_move)
        tip = QLabel("右键拖出选框可调整局部轮廓。拖框内移动，拖上方圆柄旋转，拖角点等比缩放。松开鼠标保存，Esc 返回拖图。")
        tip.setWordWrap(True)
        f.addRow(tip)
        self.reg_orient = QComboBox(); self.reg_orient.addItems(["正放（背侧朝上）", "倒放（旋转 180°）"])
        self.reg_orient.currentIndexChanged.connect(self._reg_param_changed)
        f.addRow("切片方向", self.reg_orient)
        self.reg_mirror = QCheckBox("左右翻转（切片背面朝上时勾选，左右半脑互换）")
        self.reg_mirror.toggled.connect(self._reg_param_changed)
        f.addRow(self.reg_mirror)
        self.reg_sliders = {}
        for key, label, lo, hi, step, unit in SLIDERS:
            s = QSlider(Qt.Horizontal); s.setRange(lo, hi)
            lab = QLabel(""); lab.setMinimumWidth(70)
            s.valueChanged.connect(lambda v, key=key: self._reg_slider_moved(key))
            r = QHBoxLayout(); r.addWidget(s, 1); r.addWidget(lab)
            f.addRow(label, r)
            self.reg_sliders[key] = (s, lab, step, unit)
        self.reg_use_nl = QCheckBox("使用非线性贴合")
        self.reg_use_nl.toggled.connect(self._reg_param_changed)
        f.addRow(self.reg_use_nl)
        L.addWidget(box)

        box2 = QGroupBox("局部修正（手动矫正脑区边界）"); v2 = QVBoxLayout(box2)
        self.reg_landmark_btn = QPushButton("连续点规范点（沿边缘单击）")
        self.reg_landmark_btn.setCheckable(True)
        self.reg_landmark_btn.setMinimumHeight(34)
        self.reg_landmark_btn.setToolTip("沿同一段组织边缘按顺序单击，点的直径为 10 µm。布点后选择寻边通道并应用。")
        self.reg_landmark_btn.toggled.connect(lambda on: self._reg_set_drag_mode(on, landmark=True))
        v2.addWidget(self.reg_landmark_btn)
        self._trace_clicks = []
        trace_row = QHBoxLayout()
        trace_row.addWidget(QLabel("寻边通道"))
        self.reg_trace_channel = QComboBox()
        trace_row.addWidget(self.reg_trace_channel, 1)
        self.reg_trace_apply = QPushButton("沿点列寻边")
        self.reg_trace_apply.setEnabled(False)
        self.reg_trace_apply.clicked.connect(self._reg_apply_trace)
        trace_row.addWidget(self.reg_trace_apply)
        v2.addLayout(trace_row)
        self.reg_drag_btn = QPushButton("拖动边界修正（按住边界线拖到正确位置）"); self.reg_drag_btn.setCheckable(True)
        self.reg_drag_btn.setMinimumHeight(34); self.reg_drag_btn.toggled.connect(self._reg_set_drag_mode)
        v2.addWidget(self.reg_drag_btn)
        rr = QHBoxLayout(); rr.addWidget(QLabel("影响范围"))
        self.reg_radius = QSpinBox()
        self.reg_radius.setRange(10, 4000)
        self.reg_radius.setSingleStep(10)
        self.reg_radius.setValue(800)
        self.reg_radius.setSuffix(" µm")
        self.reg_radius.setToolTip("沿拖动轨迹的影响半径。放大后可输入 10 µm 微调细边。")
        rr.addWidget(self.reg_radius, 1); v2.addLayout(rr)
        r2 = QHBoxLayout()
        b = QPushButton("撤销上一个修正"); b.clicked.connect(self._reg_undo_point); r2.addWidget(b)
        b = QPushButton("添加修正点"); b.setToolTip("另一种方式：先在边界线上点一下加点，再切到“选择”模式把点拖走。")
        b.clicked.connect(self._reg_add_points_mode); r2.addWidget(b)
        b = QPushButton("清除修正点"); b.clicked.connect(self._reg_clear_points); r2.addWidget(b)
        v2.addLayout(r2)
        r3 = QHBoxLayout()
        b = QPushButton("重新非线性贴合"); b.clicked.connect(self._reg_refit_nonlinear); r3.addWidget(b)
        b = QPushButton("恢复自动结果"); b.clicked.connect(self._reg_reset); r3.addWidget(b)
        v2.addLayout(r3)
        L.addWidget(box2)
        bs = QPushButton("同一动物的连续切片：排序和联合校正…"); bs.clicked.connect(self.series_dialog)
        bs.setToolTip("同一个脑子的切片是用同一个角度切的。把所有切片的角度统一成中位数，可以去掉单张切片的角度误差；"
                      "知道切片间距时还可以让前后位置等间距排列。")
        L.addWidget(bs)
        be = QPushButton("导出配准图（整张脑片 + 脑区轮廓，论文用）")
        be.setToolTip("导出当前图片每张已配准脑片的图：荧光 + 脑区轮廓 + 缩写 + 比例尺（PNG / TIFF / 矢量 SVG）。"
                      "线条颜色、通道、分辨率等在“⑨ 截图制图”里设置。")
        be.clicked.connect(lambda: self._require_scope() and self._reg_export_figure())
        L.addWidget(be)
        self.reg_hover = QLabel("鼠标所在脑区："); self.reg_hover.setWordWrap(True)
        self.reg_hover.setToolTip("白线是图谱标签的平滑显示。鼠标所在脑区和细胞归属按原始配准标签判定。")
        L.addWidget(self.reg_hover)
        L.addStretch(1)

        self._reg_timer = QTimer(self); self._reg_timer.setSingleShot(True)
        self._reg_timer.timeout.connect(self._reg_redraw)
        self._reg_cache = {}
        self._ctrl_src = []
        self._reg_lab = None
        self._reg_drag_on = False
        self._drag = None
        self.viewer.mouse_move_callbacks.append(self._reg_mouse_move)
        self.viewer.mouse_drag_callbacks.append(self._reg_drag)
        return w

    # ------------------------------------------------------------ 自动配准
    def auto_register(self, all_images: bool):
        from ..core.scope import selected_sections
        targets = self._target_images(all_images)
        only = {im.key: {s.name for s in selected_sections(im, self._processing_scope)} for im in targets}
        if any(not im.sections for im in targets):
            self.info("有图片还没有拆分脑片，请先到“② 脑片拆分”。")
            return
        nuc = nucleus_channel(self.project.channel_names)
        do_nl = self.reg_nonlinear_default.isChecked()
        project = self.project


        def job(task):
            from ..core import autoreg
            return autoreg.run(project, targets, nuc, nonlinear=do_nl,
                               progress=task.report, cancelled=task.is_cancelled, only=only)

        def done(res):
            for im in targets:
                regs = res.get(im.source_id, {})
                for s in im.sections:
                    if s.name in regs:
                        s.registration = regs[s.name].to_dict()
                if regs:
                    enabled = [s for s in im.sections if s.enabled]
                    registered = sum(bool(has_registration(s.registration)) for s in enabled)
                    im.status = f"已配准 {registered}/{len(enabled)} 张脑片"
            self.project.save(); self._refresh_images()
            self._reg_cache.clear()
            self._sync_registration_panel()
            n_warn = sum(1 for regs in res.values() for r in regs.values() if r.warnings)
            self.statusBar().showMessage(
                f"配准完成，请逐张检查。其中 {n_warn} 张带 ⚠ 的脑片请重点检查。" if n_warn else "配准完成，请逐张检查。", 12000)

        self.run_task(job, done, label="正在配准…")

    # ------------------------------------------------------------ 面板同步
    @staticmethod
    def _reg_combo_text(s, reg):
        if reg.done and reg.warnings:
            return f"⚠ {s.name}（已配准，请重点检查）"
        return f"{s.name}（{'已配准' if reg.done else '未配准'}）"

    def _reg_reviewed(self, reg):
        """手动调整过 = 已经看过这张脑片：去掉 ⚠。"""
        if reg.warnings:
            reg.warnings = []
            i = self.reg_sec_combo.currentIndex()
            if self.image is not None and 0 <= i < len(self.image.sections):
                self.reg_sec_combo.setItemText(i, self._reg_combo_text(self.image.sections[i], reg))

    def _reg_mark_checked(self):
        s, reg = self._reg_current()
        if s is None:
            return
        self._reg_reviewed(reg)
        s.registration = reg.to_dict(); self.project.save()
        self._reg_redraw()

    def _change_registration_atlas(self, index):
        if self.project is None:
            return
        if self.task is not None and self.task.isRunning():
            self.reg_atlas.setCurrentIndex(self.reg_atlas.findData(self.project.registration_atlas))
            self.statusBar().showMessage("当前任务结束后可切换图谱。", 5000)
            return
        self.project.select_atlas(self.reg_atlas.itemData(index))
        self.project.save()
        self._reg_cache.clear()
        self._atlas_curves = self._reg_lab = self._reg_bnd = None
        self.reg_hover.setText("鼠标所在脑区：—")
        self._refresh_images()
        self._refresh_step_marks()
        self._sync_registration_panel()
        if self._reg_current()[0] is None:
            self._reg_redraw()

    def _sync_registration_panel(self):
        if self.project is not None:
            self.reg_atlas.setCurrentIndex(self.reg_atlas.findData(self.project.registration_atlas))
        R.preload_elastix()    # 后台提前加载非线性贴合组件（首次约 10 秒），点“自动配准”时不用再等
        self.reg_sec_combo.blockSignals(True)
        self.reg_sec_combo.clear()
        secs = self.image.sections if self.image and self.slide else []
        focus = self._focus_section()
        chosen = []
        for i, s in enumerate(secs):
            reg = R.Registration.from_dict(s.registration)
            self.reg_sec_combo.addItem(self._reg_combo_text(s, reg))
            allowed = self._section_in_scope(self.image, s)
            self.reg_sec_combo.model().item(i).setEnabled(allowed)
            if allowed:
                chosen.append(i)
        self.reg_sec_combo.blockSignals(False)
        if chosen:
            index = secs.index(focus) if focus is not None and secs.index(focus) in chosen else chosen[0]
            self.reg_sec_combo.setCurrentIndex(index)
            self._reg_section_changed(index)
        else:
            self._reg_set_drag_mode(False)
            self.reg_status.clear()
            self.reg_ok_btn.setVisible(False)

    def _reg_current(self):
        i = self.reg_sec_combo.currentIndex()
        if self.slide is None or self.image is None or not (0 <= i < len(self.image.sections)):
            return None, None
        s = self.image.sections[i]
        if not self._section_in_scope(self.image, s):
            return None, None
        return s, R.Registration.from_dict(s.registration)

    def _reg_geometry(self, s):
        reg = R.Registration.from_dict(s.registration)
        key = (reg.fit_atlas, self.image.key, s.name, tuple(s.bbox0), reg.warp_file, reg.nonlinear)
        if key not in self._reg_cache:
            ds, (H, W), _ = _thumb_geometry(self.slide, s)
            warp = None
            wp = R.warp_path(self.project.results_dir, self.image.key, s.name, reg)
            if reg.nonlinear and wp.exists():
                w = np.load(wp)
                if w.shape[1:] == (H, W):
                    warp = (w[0], w[1])
            self._reg_cache[key] = (ds, H, W, warp)
        return self._reg_cache[key]

    def _reg_section_changed(self, i):
        if hasattr(self, "canvas_tools"):
            self.canvas_tools.reset()
        s, reg = self._reg_current()
        self._reg_cancel_drag()
        self._reg_clear_trace()
        if (s is None or not reg.done) and self._reg_drag_on:
            self._reg_set_drag_mode(False)
        if s is None:
            return
        chan = self.reg_trace_channel
        names = [self.ch_name(i) for i in range(len(self.slide.channels))]
        if getattr(self, "_trace_slide", None) is not self.slide or [chan.itemText(j) for j in range(chan.count())] != names:
            chan.clear()
            chan.addItems(names)
            default = next((j for j, n in enumerate(names) if "neun" in n.lower()), nucleus_channel(names))
            chan.setCurrentIndex(default if default is not None else 0)
            self._trace_slide = self.slide
        for j in range(self.focus_combo.count()):
            if self.focus_combo.itemData(j) == (self.image.key, s.name):
                self.focus_combo.setCurrentIndex(j)
                self._apply_focused_display()
                break
        self._zoom_section(i, 0)
        widgets = [self.reg_orient, self.reg_mirror, self.reg_use_nl] + [v[0] for v in self.reg_sliders.values()]
        for wdg in widgets:
            wdg.blockSignals(True)
        self.reg_orient.setCurrentIndex(1 if reg.orientation == "r180" else 0)
        self.reg_mirror.setChecked(reg.mirror)
        has_warp = reg.done and R.warp_path(self.project.results_dir, self.image.key, s.name, reg).is_file()
        self.reg_use_nl.setEnabled(reg.nonlinear or has_warp)
        self.reg_use_nl.setChecked(reg.nonlinear)
        for key, (sl, lab, step, unit) in self.reg_sliders.items():
            val = getattr(reg, key)
            if key.startswith("scale"):
                sl.setValue(int(round(val * 100)))
            else:
                sl.setValue(int(round(val / step)))
            self._reg_slider_label(key)
        for wdg in widgets:
            wdg.blockSignals(False)
        self._reg_draw_points(s, reg)
        self._reg_redraw()

    def _reg_slider_label(self, key):
        sl, lab, step, unit = self.reg_sliders[key]
        if key.startswith("scale"):
            lab.setText(f"{sl.value()}%")
        elif key.startswith("shift"):
            lab.setText(f"{sl.value() * step * 100:.1f}%")
        else:
            lab.setText(f"{sl.value() * step:.1f}{unit if unit != '× 25 µm' else ''}" if key != "ap_shift"
                        else f"{sl.value() * 25} µm")

    def _reg_slider_moved(self, key):
        self._reg_slider_label(key)
        self._reg_param_changed()

    def _reg_param_changed(self, *_):
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return
        self._reg_cancel_drag()
        self._reg_clear_trace()
        old_orient = reg.orientation
        reg.orientation = "r180" if self.reg_orient.currentIndex() == 1 else "r0"
        reg.mirror = self.reg_mirror.isChecked()
        reg.nonlinear = self.reg_use_nl.isChecked()
        for key, (sl, lab, step, unit) in self.reg_sliders.items():
            setattr(reg, key, sl.value() / 100.0 if key.startswith("scale") else sl.value() * step)
        if reg.orientation != old_orient:
            reg.warp_file = ""
            reg.nonlinear = False  # 方向变了，原来的非线性结果不再适用
            reg.control_points = []
            self.reg_use_nl.blockSignals(True); self.reg_use_nl.setChecked(False); self.reg_use_nl.blockSignals(False)
            self._reg_draw_points(s, reg)
        reg.status = "已手动调整"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        # Coalesce slider events while still redrawing during a long drag.
        if not self._reg_timer.isActive():
            self._reg_timer.start(60)

    # ------------------------------------------------------------ 显示
    def _reg_redraw(self):
        if hasattr(self, "canvas_tools") and self.canvas_tools.mode == "atlas" and not self.canvas_tools.committing:
            self.canvas_tools.reset()
        self._reg_timer.stop()
        s, reg = self._reg_current()
        if s is None or not reg.done:
            if "脑区轮廓" in self.viewer.layers:
                self.viewer.layers.remove("脑区轮廓")
            self.reg_status.setText("这张脑片还没有配准。")
            self.reg_status.setStyleSheet(""); self.reg_ok_btn.setVisible(False)
            self._reg_lab = None
            self._reg_bnd = None
            return
        try:
            ds, H, W, warp = self._reg_geometry(s)
            lab, hemi = R.label_maps(reg, W, H, warp)
        except (FileNotFoundError, ValueError) as exc:
            self._reg_lab = self._reg_bnd = None
            if "脑区轮廓" in self.viewer.layers:
                self.viewer.layers.remove("脑区轮廓")
            self.reg_status.setText(str(exc))
            return
        lab_t, hemi_t = R.orient(lab, reg.orientation), R.orient(hemi, reg.orientation)
        self._reg_lab = (s, ds, lab_t, hemi_t)
        self._reg_bnd = region_boundaries(lab_t)
        plane = R.adjusted_plane(reg)
        legacy = tuple(tuple(p) for p in reg.control_points if not is_fine(p))
        key = (id(self.project), reg.fit_atlas, self.image.key, s.name, W, H, ds, tuple(plane.to_list()), legacy, reg.nonlinear)
        cached = getattr(self, "_atlas_curves", None)
        if cached is None or cached[0] != key or cached[1] is not warp:
            self._atlas_curves = (key, warp, registration_paths(reg, W, H, warp, ds*self.slide.pixel_size_um))
        paths = self._atlas_curves[2]
        for point in reg.control_points:
            if is_fine(point):
                paths = deform_paths(paths, point, W, H)
        if reg.orientation == "r180":
            paths = [np.array([H, W], np.float32) - path for path in paths]
        self._reg_paths = paths
        rgba = np.zeros(lab_t.shape + (4,), np.uint8)  # transparent host for white contour paths
        bx, by = s.bbox0[0], s.bbox0[1]
        sc, tr = (ds, ds), (by + ds / 2, bx + ds / 2)
        if "脑区轮廓" in self.viewer.layers:
            # 已有图层时直接换数据（比删掉再新建快得多，拖滑块时不卡）
            ly = self.viewer.layers["脑区轮廓"]
            ly.scale, ly.translate = sc, tr
            ly.data = rgba
        else:
            ly = self.viewer.add_image(rgba, rgb=True, name="脑区轮廓", scale=sc, translate=tr,
                                       interpolation2d="linear", opacity=0.9)
        set_outline_paths(self.qt_viewer, ly, paths)
        for nm in ("修正箭头", "修正点"):
            if nm in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(nm), len(self.viewer.layers) - 1)
        ap = plane.ap_mm()
        txt = (f"{reg.status}；切片中心约在 Bregma {ap:+.2f} mm。"
               f"{' 已使用非线性贴合。' if reg.nonlinear and warp is not None else ''}")
        if reg.warnings:
            txt += "\n⚠ 请重点检查这张脑片：" + "；".join(reg.warnings) + "。手动调整后 ⚠ 会自动消失；确认没问题可以点下面的“已检查”。"
        self.reg_status.setText(txt)
        self.reg_status.setStyleSheet("color: #c05a00;" if reg.warnings else "")
        self.reg_ok_btn.setVisible(bool(reg.warnings))

    def _reg_export_figure(self):
        if self.image is None:
            return
        self._sync_figure_panel()
        if not self.fig_svg.isChecked() and not self.fig_tif.isChecked() and self.fig_res.currentIndex() == 0:
            self.fig_res.setCurrentIndex(max(0, self.fig_res.findData(1.0)))   # 整张脑片默认 1 µm/像素，文件不会太大
        self.fig_export_sections(False, registration_view=True)

    def _reg_mouse_move(self, viewer, event):
        if self._reg_lab is None or self.stack.currentIndex() != 5:
            return
        s, ds, lab, hemi = self._reg_lab
        y, x = event.position[-2], event.position[-1]
        ix, iy = int((x - s.bbox0[0]) / ds), int((y - s.bbox0[1]) / ds)
        if 0 <= iy < lab.shape[0] and 0 <= ix < lab.shape[1] and lab[iy, ix]:
            st = load_atlas(self.project.registration_atlas).structures
            r = st.iloc[int(lab[iy, ix]) - 1]
            h = {1: "左", 2: "右"}.get(int(hemi[iy, ix]), "")
            self.reg_hover.setText(f"鼠标所在脑区：{r['acronym']}（{r['name']}），{h}半脑")
        else:
            self.reg_hover.setText("鼠标所在脑区：—")

    # ------------------------------------------------------------ 修正点
    def _to_norm(self, s, reg, x, y):
        ds, H, W, _ = self._reg_geometry(s)
        px, py = (x - s.bbox0[0]) / ds, (y - s.bbox0[1]) / ds
        if reg.orientation == "r180":
            px, py = W - px, H - py
        return px / W, py / H

    def _from_norm(self, s, reg, u, v):
        ds, H, W, _ = self._reg_geometry(s)
        px, py = u * W, v * H
        if reg.orientation == "r180":
            px, py = W - px, H - py
        return s.bbox0[0] + px * ds, s.bbox0[1] + py * ds

    def _reg_draw_points(self, s, reg):
        for nm in ("修正点", "修正箭头"):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        pts, lines, self._ctrl_src = [], [], []
        for cp in reg.control_points:
            if len(cp) >= 8:  # a fitted contour is edited and undone as one unit
                continue
            ss, tt, ds_, dt = cp[:4]
            x, y = self._from_norm(s, reg, ds_, dt)
            sx, sy = self._from_norm(s, reg, ss, tt)
            pts.append([y, x]); self._ctrl_src.append((ss, tt, *cp[4:]))
            lines.append(np.array([[sy, sx], [y, x]], float))
        lw = max(2.0, 12 / self.slide.pixel_size_um)
        kw = dict(name="修正箭头", shape_type="line", edge_color="#ff3b30", edge_width=lw, opacity=0.9)
        arrows = self.viewer.add_shapes(lines, **kw) if lines else self.viewer.add_shapes(ndim=2, **kw)
        arrows.editable = False
        size = DOT_UM / self.slide.pixel_size_um
        layer = self.viewer.add_points(np.array(pts).reshape(-1, 2), name="修正点", size=size,
                                       face_color="#ff3b30", border_color="white")
        layer.events.data.connect(self._reg_points_changed)
        if getattr(self, "_reg_drag_on", False):
            layer.mode = "pan_zoom"; layer.mouse_pan = False   # 拖动模式下，鼠标拖动用来拖边界，不平移画面
        self.viewer.layers.selection.active = layer

    def _reg_clear_trace(self):
        self._trace_clicks.clear()
        if "规范点" in self.viewer.layers:
            self.viewer.layers.remove("规范点")
        self.reg_trace_apply.setEnabled(False)
        self.reg_trace_apply.setText("沿点列寻边")

    def _reg_plot_trace(self):
        clicks = np.asarray(self._trace_clicks)
        if not len(clicks):
            self._reg_clear_trace()
            return
        if "规范点" not in self.viewer.layers:
            layer = self.viewer.add_points(clicks[:, ::-1], name="规范点", size=DOT_UM/self.slide.pixel_size_um,
                                           face_color="#36e5d1", border_color="white", border_width=.08)
            layer.editable = False
            layer.mouse_pan = False
        else:
            self.viewer.layers["规范点"].data = clicks[:, ::-1]
        self.reg_trace_apply.setEnabled(len(clicks) >= 2)
        self.reg_trace_apply.setText(f"沿点列寻边（{len(clicks)} 点）")

    def _reg_add_trace_click(self, x, y):
        s, reg = self._reg_current()
        if s is None or not reg.done or self._reg_lab is None or self._reg_lab[0] is not s:
            return
        bx, by, width, height = s.bbox0
        if not (bx <= x < bx+width and by <= y < by+height):
            return
        if self._trace_clicks and np.linalg.norm(np.array([x, y])-self._trace_clicks[-1])*self.slide.pixel_size_um < DOT_UM/2:
            return
        self._trace_clicks.append((x, y))
        self._reg_plot_trace()

    def _reg_apply_trace(self):
        s, reg = self._reg_current()
        if s is None or not reg.done or len(self._trace_clicks) < 2:
            return
        if self._reg_timer.isActive():
            self._reg_redraw()
        if "脑区轮廓" not in self.viewer.layers or not self.viewer.layers["脑区轮廓"].visible:
            self.statusBar().showMessage("请先显示脑区轮廓。", 6000)
            return
        ds, H, W, _ = self._reg_geometry(s)
        try:
            curve = fluorescence_edge(self.slide, self._trace_clicks, self.reg_trace_channel.currentIndex(), s.bbox0)
            edits = contour_edits(self._reg_paths, (curve-s.bbox0[:2])/ds, W, H, ds*self.slide.pixel_size_um,
                                  self.reg_radius.value(), reg.orientation == "r180", reg.control_points)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 12000)
            return
        if not edits:
            self.statusBar().showMessage("这段边界已贴近点列。", 6000)
            return
        reg.control_points = [*reg.control_points, *edits]
        reg.status = "已按规范点寻边"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        self._reg_clear_trace()
        self._reg_draw_points(s, reg)
        self._reg_redraw()
        self.statusBar().showMessage("这段轮廓已更新。可继续点下一段，也可撤销整段。", 8000)

    def _reg_radius_norm(self, s):
        """当前“影响范围”（µm）→ 以缩略图宽度为单位的半径。"""
        ds, H, W, _ = self._reg_geometry(s)
        return self.reg_radius.value() / (self.slide.pixel_size_um * ds) / W

    # ---------------------------------------------------------- 拖动边界
    def _reg_cancel_drag(self):
        self._drag = None
        if "拖动中" in self.viewer.layers:
            self.viewer.layers.remove("拖动中")

    def _reg_set_drag_mode(self, on: bool, landmark=False):
        if on:
            self.canvas_tools.reset()
        self._reg_cancel_drag()
        self._reg_clear_trace()
        self._reg_drag_on = bool(on)
        for button, active in ((self.reg_landmark_btn, on and landmark), (self.reg_drag_btn, on and not landmark)):
            blocked = button.blockSignals(True)
            button.setChecked(active)
            button.blockSignals(blocked)
        # napari 的画面平移开关跟着当前图层走，所以要设在图层上
        if "修正点" in self.viewer.layers:
            self.viewer.layers["修正点"].mouse_pan = not self._reg_drag_on
        self.viewer.camera.mouse_pan = not self._reg_drag_on
        if self._reg_drag_on:
            s, reg = self._reg_current()
            if s is None or not reg.done:
                self.info("请先自动配准。"); self._reg_set_drag_mode(False); return
            if "修正点" not in self.viewer.layers:
                self._reg_draw_points(s, reg)
            layer = self.viewer.layers["修正点"]; layer.mode = "pan_zoom"; layer.mouse_pan = False
            self.viewer.layers.selection.active = layer
            self.viewer.camera.mouse_pan = False
            message = ("沿同一段组织边缘依次单击，再点“沿点列寻边”；撤销可移去最后一个点。"
                       if landmark else "按住白色边界线，拖到正确位置后松开。")
            self.statusBar().showMessage(message + "滚轮可缩放，再点一次按钮退出。", 20000)
        else:
            self.statusBar().showMessage("已退出边界修正。", 4000)

    def _reg_drag(self, viewer, event):
        if not self._reg_drag_on or self.stack.currentIndex() != 5 or getattr(event, "button", 1) != 1:
            return
        pos = event.position
        if self.reg_landmark_btn.isChecked():
            self._reg_add_trace_click(pos[-1], pos[-2])
            return
        if self._reg_begin_drag(pos[-1], pos[-2]) is None:
            return
        stroke = self._drag
        yield
        while self._drag is stroke and event.type == "mouse_move":
            p = event.position
            self._reg_update_drag(p[-1], p[-2])
            yield
        if self._drag is stroke:
            p = event.position
            self._reg_end_drag(p[-1], p[-2])

    def _reg_begin_drag(self, x, y):
        """Project onto the currently displayed outline in this section."""
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return None
        if self._reg_timer.isActive():
            self._reg_redraw()
        if self._reg_lab is None or self._reg_lab[0] is not s or "脑区轮廓" not in self.viewer.layers:
            return None
        if not self.viewer.layers["脑区轮廓"].visible:
            self.statusBar().showMessage("请先显示脑区轮廓。", 4000)
            return None
        _, ds, lab, _ = self._reg_lab
        H, W = lab.shape
        ix, iy = (x - s.bbox0[0]) / ds, (y - s.bbox0[1]) / ds
        if not (0 <= iy < H and 0 <= ix < W):
            return None
        radius_um = self.reg_radius.value()
        tol = min(150., radius_um) / (self.slide.pixel_size_um * ds)
        hit = snap_path(self._reg_paths, ix, iy, tol)
        if hit is None:
            self.statusBar().showMessage("请放大后按住要移动的白色轮廓线。", 4000)
            return None
        sx, sy = s.bbox0[0] + hit[0]*ds, s.bbox0[1] + hit[1]*ds
        self._drag = {"src": (sx, sy), "press": (x, y), "section": s}
        lw = min(8., radius_um * .12) / self.slide.pixel_size_um
        if "拖动中" in self.viewer.layers:
            self.viewer.layers.remove("拖动中")
        self.viewer.add_shapes(self._drag_shapes(sx, sy, sx, sy), shape_type=["line", "ellipse"], name="拖动中",
                               edge_color="#2ee6d6", face_color="transparent", edge_width=lw)
        self.viewer.layers.selection.active = self.viewer.layers["修正点"]
        return sx, sy

    def _reg_update_drag(self, x, y):
        if self._drag is None or "拖动中" not in self.viewer.layers:
            return
        sx, sy = self._drag["src"]
        px, py = self._drag["press"]
        self.viewer.layers["拖动中"].data = self._drag_shapes(sx, sy, sx+x-px, sy+y-py)

    def _drag_shapes(self, sx, sy, x, y):
        """拖动时的提示：起点到当前位置的线 + 以当前位置为中心的“影响范围”圆。"""
        r = self.reg_radius.value() / self.slide.pixel_size_um
        return [np.array([[sy, sx], [y, x]], float), np.array([[y-r,x-r],[y-r,x+r],[y+r,x+r],[y+r,x-r]])]

    def _reg_end_drag(self, x, y):
        if self._drag is None:
            return
        stroke = self._drag
        sx, sy = stroke["src"]
        px, py = stroke["press"]
        self._reg_cancel_drag()
        s, reg = self._reg_current()
        if s is not stroke["section"] or not reg.done:
            return
        x, y = sx + (x-px), sy + (y-py)
        if np.hypot(x-sx, y-sy) * self.slide.pixel_size_um < min(2., self.reg_radius.value()/8):
            return
        ss, tt = self._to_norm(s, reg, sx, sy)
        u, v = self._to_norm(s, reg, x, y)
        point = [float(ss), float(tt), float(u), float(v), float(self._reg_radius_norm(s)), 2]
        reg.control_points = [*reg.control_points, point]
        reg.status = "已手动调整"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        self._reg_draw_points(s, reg)
        self._reg_redraw()
        self.viewer.layers.selection.active = self.viewer.layers["修正点"]
        self.statusBar().showMessage(f"已添加第 {len(reg.control_points)} 个修正，边界已更新。统计、导出和截图都会按修正后的边界计算。", 8000)

    def _reg_undo_point(self):
        if self._trace_clicks:
            self._trace_clicks.pop()
            self._reg_plot_trace()
            return
        s, reg = self._reg_current()
        if s is None or not reg.control_points:
            return
        last = reg.control_points[-1]
        reg.control_points = ([p for p in reg.control_points if len(p) < 8 or p[7] != last[7]]
                              if len(last) > 7 else reg.control_points[:-1])
        reg.status = "已手动调整" if reg.control_points else "已配准"
        s.registration = reg.to_dict(); self.project.save()
        self._reg_draw_points(s, reg); self._reg_redraw()

    def _reg_add_points_mode(self):
        self.canvas_tools.reset()
        s, reg = self._reg_current()
        if s is None or not reg.done:
            self.info("请先自动配准。")
            return
        if self._reg_drag_on:
            self._reg_set_drag_mode(False)
        if "修正点" not in self.viewer.layers:
            self._reg_draw_points(s, reg)
        layer = self.viewer.layers["修正点"]
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self.statusBar().showMessage("在需要修正的边界线上点一下添加修正点；然后切到“选择”模式（按 S 键），把点拖到正确位置。", 15000)

    def _reg_points_changed(self, event):
        s, reg = self._reg_current()
        if s is None:
            return
        layer = self.viewer.layers["修正点"]
        action = getattr(event, "action", None)
        action = str(getattr(action, "value", action))
        data = layer.data
        rad = self._reg_radius_norm(s)
        saved = list(reg.control_points)
        if action == "added":
            for i in range(len(self._ctrl_src), len(data)):
                y, x = data[i][-2], data[i][-1]
                self._ctrl_src.append(self._to_norm(s, reg, x, y) + (rad, 2))
        elif action == "removed":
            idx = sorted(getattr(event, "data_indices", ()) or ())
            manual = [i for i, p in enumerate(saved) if len(p) < 8]
            for i in reversed(idx):
                if 0 <= i < len(self._ctrl_src):
                    self._ctrl_src.pop(i)
                    saved.pop(manual[i])
        elif action not in ("changed",):
            return
        self._ctrl_src = self._ctrl_src[:len(data)]
        while len(self._ctrl_src) < len(data):
            y, x = data[len(self._ctrl_src)][-2:]
            self._ctrl_src.append(self._to_norm(s, reg, x, y) + (rad, 2))
        cps = []
        for (ss, tt, *extra), p in zip(self._ctrl_src, data):
            u, v = self._to_norm(s, reg, p[-1], p[-2])
            cps.append([float(ss), float(tt), float(u), float(v)] + extra)
        # 箭头跟着点走
        if "修正箭头" in self.viewer.layers:
            lines = []
            for (ss, tt, *_), p in zip(self._ctrl_src, data):
                sx, sy = self._from_norm(s, reg, ss, tt)
                lines.append(np.array([[sy, sx], [p[-2], p[-1]]], float))
            try:
                self.viewer.layers["修正箭头"].data = lines
            except Exception:  # noqa: BLE001
                pass
        # Preserve the order of fitted contours interleaved with editable manual points.
        manual = iter(cps)
        reg.control_points = [p if len(p) >= 8 else next(manual) for p in saved]
        reg.control_points.extend(manual)
        reg.status = "已手动调整"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        self._reg_timer.start(200)

    def _reg_clear_points(self):
        self._reg_clear_trace()
        s, reg = self._reg_current()
        if s is None:
            return
        reg.control_points = []
        s.registration = reg.to_dict(); self.project.save()
        self._reg_draw_points(s, reg); self._reg_redraw()

    def _reg_reset(self):
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return
        for key in ("ap_shift", "tilt_dv", "tilt_ml", "rotation", "shift_x", "shift_y"):
            setattr(reg, key, 0.0)
        reg.scale_x = reg.scale_y = 1.0
        reg.control_points = []
        s.registration = reg.to_dict(); self.project.save()
        self._reg_section_changed(self.reg_sec_combo.currentIndex())

    def _reg_refit_nonlinear(self):
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return
        nuc = nucleus_channel(self.project.channel_names)
        project, image = self.project, self.image

        def job(task):
            sl = image.open_slide(z_mode=project.z_mode)
            try:
                img, mask, ds = R.section_thumbnail(sl, s, nuc)
                pixel_um = sl.pixel_size_um * ds
            finally:
                sl.close()
            R.refit_outline(reg, img, mask, project.results_dir, pixel_um)
            return reg

        def done(reg):
            s.registration = reg.to_dict(); project.save()
            self._reg_cache.clear()
            if self.image is image and self._reg_current()[0] is s:
                self._reg_section_changed(self.reg_sec_combo.currentIndex())

        self.run_task(job, done, label="正在非线性贴合…")


    # ------------------------------------------------------------ 连续切片
    def series_dialog(self):
        from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QComboBox,
                                    QCheckBox, QDoubleSpinBox, QFormLayout, QHeaderView)
        from ..core import series
        if not self._require_scope():
            return
        groups = {key: [(im, sec) for im, sec in items if self._section_in_scope(im, sec)]
                  for key, items in series.animal_sections(self.project).items()}
        groups = {key: items for key, items in groups.items() if items}
        if not groups:
            self.info("还没有脑片。")
            return
        dlg = QDialog(self); dlg.setWindowTitle("同一动物的连续切片"); fit_window(dlg, 760, 520)
        L = QVBoxLayout(dlg)
        tip = QLabel("下表按前后位置从前（Bregma 大）到后排列。动物编号在“⑦ 分组与统计”中填写；没填的按图片分组。\n"
                     "“统一切面角度”：把这只动物所有切片的倾斜角度统一为中位数，前后位置保持不变。\n"
                     "“切片间距”：知道相邻两张的实际间距时填写（例如 40 µm 切片每 4 张取 1 张 = 160 µm），"
                     "软件会让前后位置等间距排列；不确定就填 0。")
        tip.setWordWrap(True); L.addWidget(tip)
        f = QFormLayout()
        animal = QComboBox(); animal.addItems(list(groups.keys()))
        fix = QCheckBox("统一切面角度（推荐）"); fix.setChecked(True)
        sp = QDoubleSpinBox(); sp.setRange(0, 2000); sp.setSingleStep(20); sp.setSuffix(" µm"); sp.setSpecialValueText("不调整")
        nl = QCheckBox("校正后重新做非线性贴合（推荐，约每张 10 秒）"); nl.setChecked(True)
        f.addRow("动物", animal); f.addRow(fix); f.addRow("切片间距", sp); f.addRow(nl)
        L.addLayout(f)
        table = QTableWidget(); table.setEditTriggers(QTableWidget.NoEditTriggers); L.addWidget(table, 1)

        def fill(rows=None):
            items = groups[animal.currentText()]
            cols = ["图片", "脑片", "Bregma (mm)", "左右倾斜 (°)", "背腹倾斜 (°)", "状态"]
            if rows:
                cols = ["图片", "脑片"] + list(rows[0].keys())
            table.clear(); table.setColumnCount(len(cols)); table.setHorizontalHeaderLabels(cols)
            table.setRowCount(len(items))
            for i, (im, s_) in enumerate(items):
                reg = R.Registration.from_dict(s_.registration)
                if rows:
                    vals = [im.display_name, s_.name] + list(rows[i].values())
                elif reg.done:
                    ml, dv = series.angles(reg)
                    vals = [im.display_name, s_.name, f"{R.adjusted_plane(reg).ap_mm():+.2f}", f"{ml:.1f}", f"{dv:.1f}", reg.status]
                else:
                    vals = [im.display_name, s_.name, "未配准", "", "", ""]
                for j, v in enumerate(vals):
                    table.setItem(i, j, QTableWidgetItem(str(v)))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        animal.currentIndexChanged.connect(lambda _: fill())
        fill()
        bx = QDialogButtonBox()
        b_apply = bx.addButton("应用联合校正", QDialogButtonBox.AcceptRole)
        b_undo = bx.addButton("撤销联合校正", QDialogButtonBox.ResetRole)
        b_close = bx.addButton("关闭", QDialogButtonBox.RejectRole)
        L.addWidget(bx)
        b_close.clicked.connect(dlg.reject)

        def refit(items):
            project = self.project
            nuc = nucleus_channel(project.channel_names)
            results = project.results_dir

            def job(task):
                for k, (im, s_) in enumerate(items):
                    reg = R.Registration.from_dict(s_.registration)
                    if not reg.done:
                        continue
                    task.report(k, len(items), f"正在非线性贴合 {im.display_name} {s_.name}…")
                    sl = im.open_slide(z_mode=project.z_mode)
                    try:
                        img, mask, ds = R.section_thumbnail(sl, s_, nuc)
                        pixel_um = sl.pixel_size_um * ds
                    finally:
                        sl.close()
                    R.refit_outline(reg, img, mask, results, pixel_um)
                    s_.registration = reg.to_dict()
                return True

            def done(_):
                self.project.save(); self._reg_cache.clear()
                if self.stack.currentIndex() == 5:
                    self._sync_registration_panel()

            self.run_task(job, done, label="正在重新非线性贴合…")

        def apply():
            items = groups[animal.currentText()]
            regs = [R.Registration.from_dict(s_.registration) for _, s_ in items]
            done_items = [(it, r) for it, r in zip(items, regs) if r.done]
            try:
                rows = series.harmonize([r for _, r in done_items], sp.value(), fix.isChecked())
            except ValueError as e:
                self.error(str(e)); return
            for (im, s_), r in done_items:
                s_.registration = r.to_dict()
            self.project.save(); self._reg_cache.clear()
            fill(rows if len(rows) == len(items) else None)
            if nl.isChecked():
                refit([it for it, _ in done_items])
            elif self.stack.currentIndex() == 5:
                self._sync_registration_panel()

        def undo():
            items = groups[animal.currentText()]
            n = 0
            for im, s_ in items:
                reg = R.Registration.from_dict(s_.registration)
                if series.undo(reg):
                    s_.registration = reg.to_dict(); n += 1
            self.project.save(); self._reg_cache.clear(); fill()
            if n and nl.isChecked():
                refit(items)
            self.statusBar().showMessage(f"已撤销 {n} 张切片的联合校正。", 8000)

        b_apply.clicked.connect(apply); b_undo.clicked.connect(undo)
        self._series_dlg = dlg
        dlg.show()
