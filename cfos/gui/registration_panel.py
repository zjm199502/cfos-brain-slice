"""⑥ 脑图谱配准面板（作为 MainWindow 的一部分）。"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from ..core import registration as R
from ..core.atlas import load_atlas, region_boundaries, region_colors
from ..core.coloc import nucleus_channel
from ..core.regions import _thumb_geometry
from ..core.slide_io import open_slide

# 滑块：(属性名, 显示名, 最小, 最大, 每格的值, 单位)
SLIDERS = [
    ("ap_shift", "前后位置", -40, 40, 1.0, "× 25 µm"),
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
        L.addWidget(hint("把每张脑片对齐到 Allen 小鼠脑图谱（CCFv3）。\n"
                         "1. 点“自动配准”，软件会自动判断切片的前后位置、角度和是否倒放，并贴合轮廓。\n"
                         "2. 在下拉框中逐张检查：黄色线是脑区边界，应与组织结构对齐。\n"
                         "3. 不准时：先用滑块调整整体位置/角度/大小；局部不准时，点“添加修正点”，"
                         "在边界线上点一下，再把这个点拖到组织中真正对应的位置。"))
        self.reg_nonlinear_default = QCheckBox("自动配准后做非线性贴合（推荐）")
        self.reg_nonlinear_default.setChecked(True)
        L.addWidget(self.reg_nonlinear_default)
        b1 = big_button("自动配准当前图片的全部脑片"); b1.clicked.connect(lambda: self.auto_register(False))
        b2 = big_button("自动配准项目中所有图片"); b2.clicked.connect(lambda: self.auto_register(True))
        L.addWidget(b1); L.addWidget(b2)

        row = QHBoxLayout(); row.addWidget(QLabel("检查 / 调整脑片："))
        self.reg_sec_combo = QComboBox(); self.reg_sec_combo.currentIndexChanged.connect(self._reg_section_changed)
        row.addWidget(self.reg_sec_combo, 1); L.addLayout(row)
        self.reg_status = QLabel(""); self.reg_status.setWordWrap(True); L.addWidget(self.reg_status)

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
        L.addWidget(box)

        box2 = QGroupBox("局部修正"); v2 = QVBoxLayout(box2)
        r2 = QHBoxLayout()
        b = QPushButton("添加修正点"); b.clicked.connect(self._reg_add_points_mode); r2.addWidget(b)
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
        self.reg_hover = QLabel("鼠标所在脑区："); self.reg_hover.setWordWrap(True)
        L.addWidget(self.reg_hover)
        L.addStretch(1)

        self._reg_timer = QTimer(self); self._reg_timer.setSingleShot(True)
        self._reg_timer.timeout.connect(self._reg_redraw)
        self._reg_cache = {}
        self._ctrl_src = []
        self._reg_lab = None
        self.viewer.mouse_move_callbacks.append(self._reg_mouse_move)
        return w

    # ------------------------------------------------------------ 自动配准
    def auto_register(self, all_images: bool):
        targets = list(self.project.images) if all_images else [self.image]
        if any(not im.sections for im in targets):
            self.info("有图片还没有拆分脑片，请先到“② 脑片拆分”。")
            return
        nuc = nucleus_channel(self.project.channel_names)
        do_nl = self.reg_nonlinear_default.isChecked()
        project = self.project

        def job(task):
            out = {}
            for i, im in enumerate(targets):
                task.report(i * 10, len(targets) * 10, f"正在配准 {im.key}…")
                sl = open_slide(im.path)
                secs = [s for s in im.sections if s.enabled]
                regs = R.auto_register(sl, secs, nuc, str(project.results_dir),
                                       progress=lambda d, t, txt, i=i: task.report(i * 10 + d, len(targets) * 10, txt))
                if do_nl:
                    for k, s in enumerate(secs):
                        if task.is_cancelled():
                            raise InterruptedError
                        task.report(i * 10 + 4 + int(5 * k / max(len(secs), 1)), len(targets) * 10,
                                    f"正在非线性贴合 {im.key} 的 {s.name}…")
                        img, mask, ds = R.section_thumbnail(sl, s, nuc)
                        reg = regs[s.name]
                        w = R.nonlinear_fit(reg, R.orient(mask, reg.orientation))
                        np.save(R.warp_path(project.results_dir, im.key, s.name), np.stack(w))
                        reg.nonlinear = True
                out[im.path] = regs
                sl.close()
            return out

        def done(res):
            for im in targets:
                regs = res.get(im.path, {})
                for s in im.sections:
                    if s.name in regs:
                        s.registration = regs[s.name].to_dict()
                im.status = "已配准"
            self.project.save(); self._refresh_images()
            self._reg_cache.clear()
            self._sync_registration_panel()
            self.statusBar().showMessage("配准完成，请逐张检查。", 8000)

        self.run_task(job, done, label="正在配准…")

    # ------------------------------------------------------------ 面板同步
    def _sync_registration_panel(self):
        self.reg_sec_combo.blockSignals(True)
        self.reg_sec_combo.clear()
        secs = self.image.sections if self.image else []
        for s in secs:
            reg = R.Registration.from_dict(s.registration)
            self.reg_sec_combo.addItem(f"{s.name}（{'已配准' if reg.done else '未配准'}）")
        self.reg_sec_combo.blockSignals(False)
        if secs:
            self.reg_sec_combo.setCurrentIndex(0)
            self._reg_section_changed(0)

    def _reg_current(self):
        i = self.reg_sec_combo.currentIndex()
        if self.image is None or not (0 <= i < len(self.image.sections)):
            return None, None
        s = self.image.sections[i]
        return s, R.Registration.from_dict(s.registration)

    def _reg_geometry(self, s):
        key = (self.image.path, s.name, tuple(s.bbox0))
        if key not in self._reg_cache:
            ds, (H, W), _ = _thumb_geometry(self.slide, s)
            warp = None
            wp = R.warp_path(self.project.results_dir, self.image.key, s.name)
            if wp.exists():
                w = np.load(wp)
                if w.shape[1:] == (H, W):
                    warp = (w[0], w[1])
            self._reg_cache[key] = (ds, H, W, warp)
        return self._reg_cache[key]

    def _reg_section_changed(self, i):
        s, reg = self._reg_current()
        if s is None:
            return
        self._zoom_section(i, 0)
        widgets = [self.reg_orient, self.reg_mirror, self.reg_use_nl] + [v[0] for v in self.reg_sliders.values()]
        for wdg in widgets:
            wdg.blockSignals(True)
        self.reg_orient.setCurrentIndex(1 if reg.orientation == "r180" else 0)
        self.reg_mirror.setChecked(reg.mirror)
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
        old_orient = reg.orientation
        reg.orientation = "r180" if self.reg_orient.currentIndex() == 1 else "r0"
        reg.mirror = self.reg_mirror.isChecked()
        reg.nonlinear = self.reg_use_nl.isChecked()
        for key, (sl, lab, step, unit) in self.reg_sliders.items():
            setattr(reg, key, sl.value() / 100.0 if key.startswith("scale") else sl.value() * step)
        if reg.orientation != old_orient:
            reg.nonlinear = False  # 方向变了，原来的非线性结果不再适用
            reg.control_points = []
            self.reg_use_nl.blockSignals(True); self.reg_use_nl.setChecked(False); self.reg_use_nl.blockSignals(False)
            self._reg_draw_points(s, reg)
        reg.status = "已手动调整"
        s.registration = reg.to_dict()
        self._save_later()
        self._reg_timer.start(120)

    # ------------------------------------------------------------ 显示
    def _reg_redraw(self):
        s, reg = self._reg_current()
        for nm in ("脑区轮廓",):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        if s is None or not reg.done:
            self.reg_status.setText("这张脑片还没有配准。")
            self._reg_lab = None
            return
        ds, H, W, warp = self._reg_geometry(s)
        lab, hemi = R.label_maps(reg, W, H, warp)
        lab_t, hemi_t = R.orient(lab, reg.orientation), R.orient(hemi, reg.orientation)
        self._reg_lab = (s, ds, lab_t, hemi_t)
        b = region_boundaries(lab_t)
        rgba = np.zeros(lab_t.shape + (4,), np.uint8)
        rgba[..., :3] = region_colors(lab_t)
        rgba[..., 3] = np.where(lab_t > 0, 40, 0)
        rgba[b] = [255, 230, 0, 255]
        bx, by = s.bbox0[0], s.bbox0[1]
        self.viewer.add_image(rgba, rgb=True, name="脑区轮廓", scale=(ds, ds),
                              translate=(by + ds / 2, bx + ds / 2), opacity=0.9)
        if "修正点" in self.viewer.layers:
            self.viewer.layers.move(self.viewer.layers.index("修正点"), len(self.viewer.layers) - 1)
        ap = R.adjusted_plane(reg).ap_mm()
        self.reg_status.setText(f"{reg.status}；切片中心约在 Bregma {ap:+.2f} mm。"
                                f"{' 已使用非线性贴合。' if reg.nonlinear and warp is not None else ''}")

    def _reg_mouse_move(self, viewer, event):
        if self._reg_lab is None or self.stack.currentIndex() != 5:
            return
        s, ds, lab, hemi = self._reg_lab
        y, x = event.position[-2], event.position[-1]
        ix, iy = int((x - s.bbox0[0]) / ds), int((y - s.bbox0[1]) / ds)
        if 0 <= iy < lab.shape[0] and 0 <= ix < lab.shape[1] and lab[iy, ix]:
            st = load_atlas().structures
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
        if "修正点" in self.viewer.layers:
            self.viewer.layers.remove("修正点")
        pts, self._ctrl_src = [], []
        for ss, tt, ds_, dt in reg.control_points:
            x, y = self._from_norm(s, reg, ds_, dt)
            pts.append([y, x]); self._ctrl_src.append((ss, tt))
        size = 120 / self.slide.pixel_size_um
        layer = self.viewer.add_points(np.array(pts).reshape(-1, 2), name="修正点", size=size,
                                       face_color="#ff3b30", border_color="white")
        layer.events.data.connect(self._reg_points_changed)
        self.viewer.layers.selection.active = layer

    def _reg_add_points_mode(self):
        s, reg = self._reg_current()
        if s is None or not reg.done:
            self.info("请先自动配准。")
            return
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
        if action == "added":
            for i in range(len(self._ctrl_src), len(data)):
                y, x = data[i][-2], data[i][-1]
                self._ctrl_src.append(self._to_norm(s, reg, x, y))
        elif action == "removed":
            idx = sorted(getattr(event, "data_indices", ()) or ())
            for i in reversed(idx):
                if 0 <= i < len(self._ctrl_src):
                    self._ctrl_src.pop(i)
        elif action not in ("changed",):
            return
        self._ctrl_src = self._ctrl_src[:len(data)]
        while len(self._ctrl_src) < len(data):
            y, x = data[len(self._ctrl_src)][-2:]
            self._ctrl_src.append(self._to_norm(s, reg, x, y))
        cps = []
        for (ss, tt), p in zip(self._ctrl_src, data):
            u, v = self._to_norm(s, reg, p[-1], p[-2])
            cps.append([float(ss), float(tt), float(u), float(v)])
        reg.control_points = cps
        reg.status = "已手动调整"
        s.registration = reg.to_dict()
        self._save_later()
        self._reg_timer.start(200)

    def _reg_clear_points(self):
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
        path, key, results = self.image.path, self.image.key, self.project.results_dir

        def job(task):
            sl = open_slide(path)
            img, mask, ds = R.section_thumbnail(sl, s, nuc)
            sl.close()
            r0 = R.Registration.from_dict(reg.to_dict()); r0.control_points = []
            w = R.nonlinear_fit(r0, R.orient(mask, reg.orientation))
            np.save(R.warp_path(results, key, s.name), np.stack(w))
            return True

        def done(_):
            reg.nonlinear = True
            s.registration = reg.to_dict(); self.project.save()
            self._reg_cache.clear()
            self._reg_section_changed(self.reg_sec_combo.currentIndex())

        self.run_task(job, done, label="正在非线性贴合…")


    # ------------------------------------------------------------ 连续切片
    def series_dialog(self):
        from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QComboBox,
                                    QCheckBox, QDoubleSpinBox, QFormLayout, QHeaderView)
        from ..core import series
        groups = series.animal_sections(self.project)
        if not groups:
            self.info("还没有脑片。")
            return
        dlg = QDialog(self); dlg.setWindowTitle("同一动物的连续切片"); dlg.resize(760, 520)
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
                    vals = [im.key, s_.name] + list(rows[i].values())
                elif reg.done:
                    ml, dv = series.angles(reg)
                    vals = [im.key, s_.name, f"{R.adjusted_plane(reg).ap_mm():+.2f}", f"{ml:.1f}", f"{dv:.1f}", reg.status]
                else:
                    vals = [im.key, s_.name, "未配准", "", "", ""]
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
            nuc = nucleus_channel(self.project.channel_names)
            results = self.project.results_dir

            def job(task):
                for k, (im, s_) in enumerate(items):
                    reg = R.Registration.from_dict(s_.registration)
                    if not reg.done:
                        continue
                    task.report(k, len(items), f"正在非线性贴合 {im.key} {s_.name}…")
                    sl = open_slide(im.path)
                    try:
                        img, mask, ds = R.section_thumbnail(sl, s_, nuc)
                    finally:
                        sl.close()
                    r0 = R.Registration.from_dict(reg.to_dict()); r0.control_points = []
                    w = R.nonlinear_fit(r0, R.orient(mask, reg.orientation))
                    np.save(R.warp_path(results, im.key, s_.name), np.stack(w))
                    reg.nonlinear = True
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
