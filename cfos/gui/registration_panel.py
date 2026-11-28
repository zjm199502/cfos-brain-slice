"""⑥ 脑图谱配准面板（作为 MainWindow 的一部分）。"""
from __future__ import annotations

from ..core.atlas_choice import has_registration
import numpy as np
from .layout import fit_window
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QShortcut,
    QSlider, QVBoxLayout, QWidget,
)

from ..core import registration as R
from ..core.atlas import load_atlas, region_boundaries
from ..core.atlas_contours import registration_paths
from ..core.edge_drag import deform_paths, is_fine
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

        r3 = QHBoxLayout()
        b = QPushButton("重新非线性贴合"); b.setToolTip("按现在的切面重新贴合组织轮廓和 DAPI 纹理（有标志点时一并精细贴合）。")
        b.clicked.connect(lambda: self._reg_refit()); r3.addWidget(b)
        b = QPushButton("恢复自动结果"); b.setToolTip("切面滑块归零，清除标志点。")
        b.clicked.connect(self._reg_reset); r3.addWidget(b)
        f.addRow(r3)
        L.addWidget(box)

        box2 = QGroupBox("标志点（自动配准对不上时用）"); v2 = QVBoxLayout(box2)
        tip2 = QLabel("点“添加标志点”，在白线上按住要对齐的结构（DG 尖端、CA3 末端、脑室角等），拖到它在组织上的真实位置松开。"
                      "边界立即跟着变，统计也按新的边界算。")
        tip2.setWordWrap(True); v2.addWidget(tip2)
        self.reg_lm_btn = QPushButton("添加标志点"); self.reg_lm_btn.setCheckable(True); self.reg_lm_btn.setMinimumHeight(34)
        self.reg_lm_btn.toggled.connect(self._reg_set_landmark_mode)
        v2.addWidget(self.reg_lm_btn)
        self.reg_lm_list = QListWidget(); self.reg_lm_list.setMaximumHeight(132); self.reg_lm_list.setVisible(False)
        self.reg_lm_list.currentRowChanged.connect(self._reg_lm_selected)
        QShortcut(QKeySequence.Delete, self.reg_lm_list, self._reg_lm_delete)
        QShortcut(QKeySequence("Backspace"), self.reg_lm_list, self._reg_lm_delete)
        v2.addWidget(self.reg_lm_list)
        r2 = QHBoxLayout()
        self.reg_lm_del = QPushButton("删除所选"); self.reg_lm_del.clicked.connect(self._reg_lm_delete); r2.addWidget(self.reg_lm_del)
        self.reg_lm_undo = QPushButton("撤销"); self.reg_lm_undo.clicked.connect(self._reg_lm_undo_last); r2.addWidget(self.reg_lm_undo)
        self.reg_lm_clear = QPushButton("全部清除"); self.reg_lm_clear.clicked.connect(self._reg_lm_clear); r2.addWidget(self.reg_lm_clear)
        v2.addLayout(r2)
        self.reg_lm_state = QLabel(""); self.reg_lm_state.setWordWrap(True); v2.addWidget(self.reg_lm_state)
        self.reg_lm_fit = QPushButton("按标志点精细贴合（约 20 秒，可选）")
        self.reg_lm_fit.setToolTip("用 DAPI 纹理让标志点之间的结构也贴合，标志点作为硬约束。不点也可以：插值结果已经生效。")
        self.reg_lm_fit.clicked.connect(lambda: self._reg_refit("正在按标志点精细贴合…"))
        v2.addWidget(self.reg_lm_fit)
        self.reg_legacy = QWidget(); lay = QVBoxLayout(self.reg_legacy); lay.setContentsMargins(0, 0, 0, 0)
        self.reg_legacy_label = QLabel(""); self.reg_legacy_label.setWordWrap(True)
        self.reg_legacy_label.setStyleSheet("color: #c05a00;")
        lay.addWidget(self.reg_legacy_label)
        b = QPushButton("移除旧版修正"); b.clicked.connect(self._reg_remove_legacy); lay.addWidget(b)
        self.reg_legacy.setVisible(False)
        v2.addWidget(self.reg_legacy)
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
        self._reg_lab = None
        self._lm_on = False
        self._lm_pending = None
        self._lm_undo = []
        self._lm_timer = QTimer(self); self._lm_timer.setSingleShot(True)
        self._lm_timer.timeout.connect(self._reg_lm_replane)
        self.viewer.mouse_move_callbacks.append(self._reg_mouse_move)
        self.viewer.mouse_move_callbacks.append(self._reg_lm_hover)
        self.viewer.mouse_drag_callbacks.append(self._reg_lm_mouse)
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
        s.registration = reg.to_dict(); self._save_later()
        self._reg_redraw()

    def _change_registration_atlas(self, index):
        if self.project is None:
            return
        if self.task is not None and self.task.isRunning():
            self.reg_atlas.setCurrentIndex(self.reg_atlas.findData(self.project.registration_atlas))
            self.statusBar().showMessage("当前任务结束后可切换图谱。", 5000)
            return
        self.project.select_atlas(self.reg_atlas.itemData(index))
        self._save_later()
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
            self._reg_set_landmark_mode(False)
            self._reg_lm_panel(None, R.Registration())
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
        self._lm_timer.stop()
        if self._lm_on:
            self._reg_set_landmark_mode(s is not None and not self._reg_lm_blocker(reg))
        if s is None:
            self._reg_draw_landmarks(None, R.Registration())
            return
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
        self._reg_redraw()
        self._reg_draw_landmarks(s, reg)

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
        self._lm_pending = None
        self._reg_lm_rubber(None)
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
            reg.landmarks, reg.base_warp_file, reg.landmark_fit = [], "", ""
            self._lm_undo = [(sec, L) for sec, L in self._lm_undo if sec is not s]
            self.reg_use_nl.blockSignals(True); self.reg_use_nl.setChecked(False); self.reg_use_nl.blockSignals(False)
            self._reg_draw_landmarks(s, reg)
        reg.status = "已手动调整"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        # Coalesce slider events while still redrawing during a long drag.
        if not self._reg_timer.isActive():
            self._reg_timer.start(60)
        if reg.landmarks:
            self._lm_timer.start(450)   # 滑块停下后，标志点按新切面重新插值

    # ------------------------------------------------------------ 显示
    def _reg_redraw(self):
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
        for nm in ("标志点箭头", "标志点", "标志点（拖动中）"):
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

    # ------------------------------------------------------------ 标志点
    # 一个标志点 = “白线上的这个结构”（图谱体素坐标）+ “它在组织上的真实位置”（脑片归一化坐标）。
    # 每次改动都立即更新形变（自动结果 + 标志点插值），所见即所得，统计也按它算；精细贴合是可选的。
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

    def _reg_px(self, s, reg, x, y):
        """画布坐标 → 已校正方向的缩略图角点坐标。"""
        ds, H, W, _ = self._reg_geometry(s)
        u, v = self._to_norm(s, reg, x, y)
        return u * W, v * H

    def _reg_data(self, s, reg, px, py):
        ds, H, W, _ = self._reg_geometry(s)
        return self._from_norm(s, reg, px / W, py / H)

    def _reg_pixel_um(self, s):
        return self.slide.pixel_size_um * self._reg_geometry(s)[0]

    def _reg_lm_blocker(self, reg):
        if not reg.done:
            return "请先自动配准。"
        if reg.fit_atlas not in R.DEEP_ATLASES:
            return "标志点目前只支持 Allen CCFv3 图谱。"
        if reg.control_points:
            return "这张脑片还有旧版手动修正，请先点“移除旧版修正”。"
        return ""

    def _reg_set_landmark_mode(self, on: bool):
        if on:
            s, reg = self._reg_current()
            why = "请先选择脑片。" if s is None else self._reg_lm_blocker(reg)
            if why:
                self.statusBar().showMessage(why, 8000)
                on = False
            else:
                self.canvas_tools.reset()
        self._lm_on = bool(on)
        self._lm_pending = None
        self._reg_lm_rubber(None)
        blocked = self.reg_lm_btn.blockSignals(True)
        self.reg_lm_btn.setChecked(self._lm_on)
        self.reg_lm_btn.blockSignals(blocked)
        # napari 的画面平移开关跟着当前图层走，所以图层和相机都要设
        for nm in ("标志点", "标志点箭头"):
            if nm in self.viewer.layers:
                self.viewer.layers[nm].mouse_pan = not self._lm_on
        self.viewer.camera.mouse_pan = not self._lm_on
        if self._lm_on:
            self.statusBar().showMessage("在白线上按住要对齐的结构，拖到它在组织上的真实位置松开；"
                                         "也可以先点白线、再点组织。拖动已有的标志点可以挪位置。再点一次按钮或按 Esc 退出。", 20000)

    def _reg_lm_rubber(self, line):
        """拖动或等待第二下单击时的提示线（画布坐标 (x, y) 两点）。"""
        if line is None:
            if "标志点（拖动中）" in self.viewer.layers:
                self.viewer.layers.remove("标志点（拖动中）")
            return
        (x0, y0), (x1, y1) = line
        data = [np.array([[y0, x0], [y1, x1]], float)]
        if "标志点（拖动中）" in self.viewer.layers:
            self.viewer.layers["标志点（拖动中）"].data = data
        else:
            lw = max(2.0, 10 / self.slide.pixel_size_um)
            layer = self.viewer.add_shapes(data, shape_type="line", name="标志点（拖动中）", edge_color="#2ee6d6",
                                           edge_width=lw, opacity=.95)
            layer.mouse_pan = False
            self.viewer.layers.selection.active = self.viewer.layers[self.image_layers[0].name] \
                if self.image_layers and self.image_layers[0] in self.viewer.layers else layer

    def _reg_lm_targets(self, s, reg):
        ds, H, W, _ = self._reg_geometry(s)
        sec, _ = R.landmark_pixels(reg, W, H)
        return [self._reg_data(s, reg, x, y) for x, y in sec]

    def _reg_lm_hit(self, s, reg, x, y, tol):
        best, bd = None, tol
        for i, (tx, ty) in enumerate(self._reg_lm_targets(s, reg)):
            d = float(np.hypot(tx - x, ty - y))
            if d <= bd:
                best, bd = i, d
        return best

    def _reg_lm_mouse(self, viewer, event):
        if not self._lm_on or self.stack.currentIndex() != 5 or getattr(event, "button", 1) != 1:
            return
        s, reg = self._reg_current()
        if s is None or self._reg_lm_blocker(reg):
            return
        if self._reg_timer.isActive():
            self._reg_redraw()
        ds, H, W, _ = self._reg_geometry(s)
        y0, x0 = event.position[-2], event.position[-1]
        px, py = self._reg_px(s, reg, x0, y0)
        if not (0 <= px < W and 0 <= py < H):
            return
        tol = 10.0 / max(float(viewer.camera.zoom), 1e-9)   # 屏幕上 10 个像素
        hit = self._reg_lm_hit(s, reg, x0, y0, tol)
        start = self._lm_pending if self._lm_pending is not None and hit is None else (x0, y0)
        moved = False
        yield
        while event.type == "mouse_move":
            x1, y1 = event.position[-1], event.position[-2]
            moved = moved or np.hypot(x1 - x0, y1 - y0) > tol / 2
            if hit is not None:
                tx, ty = self._reg_lm_targets(s, reg)[hit]
                self._reg_lm_rubber(((tx, ty), (x1, y1)))
            elif moved or self._lm_pending is not None:
                self._reg_lm_rubber((start, (x1, y1)))
            yield
        x1, y1 = event.position[-1], event.position[-2]
        self._reg_lm_rubber(None)
        if hit is not None:
            if moved:
                self._reg_lm_move(s, reg, hit, x1, y1)
            else:
                self.reg_lm_list.setCurrentRow(hit)
            return
        if self._lm_pending is not None:
            self._lm_pending = None
            self._reg_lm_add(s, reg, start, (x1, y1) if moved else (x0, y0))
        elif moved:
            self._reg_lm_add(s, reg, (x0, y0), (x1, y1))
        else:
            self._lm_pending = (x0, y0)
            self._reg_lm_rubber(((x0, y0), (x0, y0)))
            self.statusBar().showMessage("再点它在组织上的真实位置（Esc 取消）。", 15000)

    def _reg_lm_hover(self, viewer, event):
        if self._lm_on and self._lm_pending is not None and self.stack.currentIndex() == 5:
            self._reg_lm_rubber((self._lm_pending, (event.position[-1], event.position[-2])))

    def _reg_lm_add(self, s, reg, src, dst):
        ds, H, W, warp = self._reg_geometry(s)
        sx, sy = self._reg_px(s, reg, *src)
        tx, ty = self._reg_px(s, reg, *dst)
        if not (0 <= tx < W and 0 <= ty < H):
            self.statusBar().showMessage("真实位置要点在这张脑片里面。", 6000)
            return
        if np.hypot(tx - sx, ty - sy) * self._reg_pixel_um(s) < 4:
            self.statusBar().showMessage("白线已经在这个位置上了，不需要标志点。", 6000)
            return
        xyz = R.landmark_atlas_xyz(reg, W, H, None if warp is None else np.stack(warp), sx, sy)
        new = [*reg.landmarks, [tx / W, ty / H, *xyz]]
        self._reg_lm_commit(s, reg, new, f"已添加第 {len(new)} 个标志点，边界已更新。", select=len(new) - 1)

    def _reg_lm_move(self, s, reg, i, x, y):
        ds, H, W, _ = self._reg_geometry(s)
        tx, ty = self._reg_px(s, reg, x, y)
        if not (0 <= tx < W and 0 <= ty < H):
            return
        new = [list(L) for L in reg.landmarks]
        new[i][0:2] = [tx / W, ty / H]
        self._reg_lm_commit(s, reg, new, f"已移动标志点 {i + 1}。", select=i)

    def _reg_lm_commit(self, s, reg, landmarks, message, select=None, undo=True):
        ds, H, W, _ = self._reg_geometry(s)
        before = [list(L) for L in reg.landmarks]
        try:
            R.set_landmarks(reg, landmarks, self.project.results_dir, W, H, self._reg_pixel_um(s))
        except R.LandmarkError as exc:
            self.statusBar().showMessage(str(exc), 10000)
            self._reg_draw_landmarks(s, reg)
            return False
        if undo:
            self._lm_undo = [*self._lm_undo[-49:], (s, before)]
        reg.status = "已手动调整"
        self._reg_reviewed(reg)
        s.registration = reg.to_dict()
        self._save_later()
        # 每次改动都会生成新的位移场文件；只留这张脑片当前用的那份缓存
        self._reg_cache = {k: v for k, v in self._reg_cache.items()
                           if k[1:3] != (self.image.key, s.name) or k[4] == reg.warp_file}
        self._reg_redraw()
        self._reg_draw_landmarks(s, reg, select)
        self.statusBar().showMessage(message, 8000)
        return True

    def _reg_draw_landmarks(self, s, reg, select=None):
        for nm in ("标志点", "标志点箭头"):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        self._reg_lm_panel(s, reg, select)
        if s is None or not reg.done or not reg.landmarks:
            return
        ds, H, W, warp = self._reg_geometry(s)
        sec, atl = R.landmark_pixels(reg, W, H)
        base = R.base_warp(reg, self.project.results_dir, W, H)
        from ..core.deep_refine import invert_at
        points, lines = [], []
        for (x, y), (ax, ay) in zip(sec, atl):
            tx, ty = self._reg_data(s, reg, x, y)
            # 箭头从“自动结果把这个结构画在哪里”指向“它真实在哪里”
            qx, qy = invert_at(base, (ax - .5, ay - .5)) + .5
            fx, fy = self._reg_data(s, reg, qx, qy)
            points.append([ty, tx])
            lines.append(np.array([[fy, fx], [ty, tx]], float))
        lw = max(2.0, 10 / self.slide.pixel_size_um)
        arrows = self.viewer.add_shapes(lines, shape_type="line", name="标志点箭头", edge_color="#ffb000",
                                        edge_width=lw, opacity=.85)
        size = max(6 * ds, 40 / self.slide.pixel_size_um)
        colors = ["#ff4d1f" if i == select else "#ffb000" for i in range(len(points))]
        layer = self.viewer.add_points(np.array(points), name="标志点", size=size, face_color=colors,
                                       border_color="white", features={"n": [str(i + 1) for i in range(len(points))]},
                                       text={"string": "{n}", "color": "white", "size": 10, "anchor": "upper_left"})
        try:
            layer.canvas_size_limits = (9, 24)   # 缩小看全片时也看得见，放大时不挡住组织
        except (AttributeError, ValueError):
            pass
        for ly in (arrows, layer):
            ly.mouse_pan = not self._lm_on
        active = self.image_layers[0] if self.image_layers and self.image_layers[0] in self.viewer.layers else None
        if active is not None:
            self.viewer.layers.selection.active = active

    def _reg_lm_panel(self, s, reg, select=None):
        """标志点列表、状态说明和按钮可用状态。"""
        lst = self.reg_lm_list
        lst.blockSignals(True)
        lst.clear()
        n = len(reg.landmarks) if s is not None and reg.done else 0
        legacy = len(reg.control_points) if s is not None else 0
        self.reg_legacy.setVisible(bool(legacy))
        self.reg_legacy_label.setText(f"这张脑片有 {legacy} 处旧版手动修正，仍然生效。要用标志点，请先移除。")
        if n:
            ds, H, W, warp = self._reg_geometry(s)
            pixel_um = self._reg_pixel_um(s)
            base = R.base_warp(reg, self.project.results_dir, W, H)
            offsets = R.landmark_offsets_um(reg, base, W, H, pixel_um)
            residual = R.landmark_offsets_um(reg, None if warp is None else np.stack(warp), W, H, pixel_um)
            try:
                names = R.landmark_regions(reg)
            except (OSError, ValueError, FileNotFoundError):
                names = [""] * n
            for i in range(n):
                text = f"{i + 1}.  {names[i] or '—'}　自动结果偏了 {offsets[i]:.0f} µm"
                if reg.landmark_fit == "fitted":
                    text += f"，现在差 {residual[i]:.0f} µm"
                lst.addItem(text)
            if select is not None and 0 <= select < n:
                lst.setCurrentRow(select)
            if reg.landmark_fit == "fitted":
                state = f"{n} 个标志点，已按 DAPI 纹理精细贴合（标志点最大偏差 {max(residual):.0f} µm）。"
            else:
                state = (f"{n} 个标志点，边界已按标志点插值更新，统计按现在的边界计算。"
                         "需要标志点之间的结构也贴合 DAPI 时，点“按标志点精细贴合”。")
        else:
            state = ("自动配准对不上的地方（组织变形、CA1/CA3 交界、切面定位失败）才需要标志点。"
                     if not legacy else "")
        lst.blockSignals(False)
        lst.setVisible(bool(n))
        self.reg_lm_state.setText(state)
        for b in (self.reg_lm_del, self.reg_lm_clear):
            b.setEnabled(bool(n))
        self.reg_lm_undo.setEnabled(any(sec is s for sec, _ in self._lm_undo))
        self.reg_lm_fit.setEnabled(bool(n) and reg.landmark_fit != "fitted")

    def _reg_lm_selected(self, row):
        s, reg = self._reg_current()
        if s is None or not (0 <= row < len(reg.landmarks)):
            return
        self._reg_draw_landmarks(s, reg, row)
        x, y = self._reg_lm_targets(s, reg)[row]
        cam = self.viewer.camera
        cam.center = (*tuple(cam.center)[:-2], y, x)

    def _reg_lm_delete(self):
        s, reg = self._reg_current()
        row = self.reg_lm_list.currentRow()
        if s is None or not (0 <= row < len(reg.landmarks)):
            self.statusBar().showMessage("先在列表里选中要删除的标志点。", 5000)
            return
        new = [L for i, L in enumerate(reg.landmarks) if i != row]
        self._reg_lm_commit(s, reg, new, f"已删除标志点 {row + 1}，边界已更新。",
                            select=min(row, len(new) - 1) if new else None)

    def _reg_lm_clear(self):
        s, reg = self._reg_current()
        if s is None or not reg.landmarks:
            return
        self._reg_lm_commit(s, reg, [], "已清除全部标志点，边界回到自动结果（可撤销）。")

    def _reg_lm_undo_last(self):
        s, reg = self._reg_current()
        for i in range(len(self._lm_undo) - 1, -1, -1):
            sec, landmarks = self._lm_undo[i]
            if sec is s:
                del self._lm_undo[i]
                self._reg_lm_commit(s, reg, landmarks, "已撤销上一步。", undo=False)
                return
        self.statusBar().showMessage("没有可撤销的标志点操作。", 5000)

    def _reg_lm_replane(self):
        """切面滑块停下后，标志点按新切面重新插值（标志点存的是图谱坐标，不会丢）。"""
        s, reg = self._reg_current()
        if s is None or not reg.done or not reg.landmarks or self._reg_lm_blocker(reg):
            return
        fitted = reg.landmark_fit == "fitted"
        ok = self._reg_lm_commit(s, reg, reg.landmarks, "标志点已按新的切面更新。", undo=False)
        if ok and fitted:
            self.statusBar().showMessage("切面变了，标志点改用插值；需要时再点“按标志点精细贴合”。", 10000)

    def _reg_remove_legacy(self):
        s, reg = self._reg_current()
        if s is None:
            return
        reg.control_points = []
        reg.status = "已手动调整"
        s.registration = reg.to_dict(); self._save_later()
        self._reg_redraw(); self._reg_draw_landmarks(s, reg)
        self.statusBar().showMessage("已移除旧版手动修正，现在可以添加标志点。", 8000)

    def _reg_reset(self):
        """恢复自动结果：切面滑块归零，清除标志点和旧版修正。切面动过时重新贴合，让形变也回到自动结果。"""
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return
        keys = ("ap_shift", "tilt_dv", "tilt_ml", "rotation", "shift_x", "shift_y")
        moved = any(getattr(reg, k) for k in keys) or (reg.scale_x, reg.scale_y) != (1.0, 1.0)
        for key in keys:
            setattr(reg, key, 0.0)
        reg.scale_x = reg.scale_y = 1.0
        reg.control_points = []
        ds, H, W, _ = self._reg_geometry(s)
        R.set_landmarks(reg, [], self.project.results_dir, W, H, self._reg_pixel_um(s))
        self._lm_undo = [(sec, L) for sec, L in self._lm_undo if sec is not s]
        s.registration = reg.to_dict(); self._save_later()
        self._reg_section_changed(self.reg_sec_combo.currentIndex())
        if moved and reg.nonlinear:
            self._reg_refit(label="正在恢复自动结果…")
        else:
            self.statusBar().showMessage("已恢复自动结果。", 6000)

    def _reg_refit(self, label="正在非线性贴合…"):
        s, reg = self._reg_current()
        if s is None or not reg.done:
            return
        self._reg_set_landmark_mode(False)
        nuc = nucleus_channel(self.project.channel_names)
        project, image = self.project, self.image

        def job(task):
            sl = image.open_slide(z_mode=project.z_mode)
            try:
                img, mask, ds = R.section_thumbnail(sl, s, nuc)
                pixel_um = sl.pixel_size_um * ds
            finally:
                sl.close()
            report = R.refit_outline(reg, img, mask, project.results_dir, pixel_um)
            return reg, report

        def done(result):
            reg, report = result
            s.registration = reg.to_dict(); project.save()
            self._reg_cache.clear()
            if self.image is image and self._reg_current()[0] is s:
                self._reg_section_changed(self.reg_sec_combo.currentIndex())
            if report is None:
                msg = "非线性贴合完成。"
            elif reg.landmark_fit == "fitted":
                msg = "已按标志点精细贴合。"
            else:
                msg = f"精细贴合没有通过（{report.get('reason', '')}），继续使用标志点插值，结果仍然有效。"
            self.statusBar().showMessage(msg, 12000)

        self.run_task(job, done, label=label)


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
                     "软件会让前后位置等间距排列；不确定就填 0。\n"
                     "填过的间距会记在项目里：之后自动配准时，按切片编号顺序推算每张的前后位置，偏离太多的会自动挪回并标 ⚠。")
        tip.setWordWrap(True); L.addWidget(tip)
        f = QFormLayout()
        animal = QComboBox(); animal.addItems(list(groups.keys()))
        fix = QCheckBox("统一切面角度（推荐）"); fix.setChecked(True)
        sp = QDoubleSpinBox(); sp.setRange(0, 2000); sp.setSingleStep(20); sp.setSuffix(" µm"); sp.setSpecialValueText("不调整")
        sp.setValue(float(getattr(self.project, "section_spacing_um", 0.0) or 0.0))
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
            self.project.section_spacing_um = float(sp.value())   # 之后的自动配准也按这个间距约束前后位置
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
