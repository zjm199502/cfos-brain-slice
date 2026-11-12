"""主窗口：左侧是分步向导，中间是看图窗口（napari），右侧是当前步骤的操作面板。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari.qt import QtViewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core import detection as det
from ..core.coloc import colocalize, merged_cells, summarize, nucleus_channel
from ..core.export import export_excel
from ..core.project import Project, ImageEntry, STANDARD_CHANNELS
from ..core.sections import Section, find_sections, contour_in_bbox
from ..core.slide_io import open_slide
from .worker import Task
from .registration_panel import RegistrationMixin
from .groups_panel import GroupsMixin
from .quality_panel import QualityMixin
from .relocate_panel import RelocateMixin
from .teaching import TeachingMixin
from .figure_panel import FigureMixin
from .accuracy_panel import AccuracyMixin

APP_NAME = "cFos 脑片分析"
IMAGE_FILTER = "荧光全片图像 (*.qptiff *.tif *.tiff *.ome.tif *.ome.tiff);;所有文件 (*)"
COLORMAP = {"DAPI": "blue", "Green": "green", "Red": "red", "Far red": "magenta"}
POINT_COLOR = {"DAPI": "#4f8cff", "Green": "#3ddc4a", "Red": "#ff4040", "Far red": "#ff40ff"}

STEPS = ["① 项目与图片", "② 脑片拆分", "③ 亮度 / 对比度", "④ 细胞识别", "⑤ 共标统计",
         "⑥ 脑图谱配准", "⑦ 分组与统计", "⑧ 导出 Excel", "⑨ 截图制图"]


def hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color:#e6edf3; background:#30455e; padding:8px; border-radius:6px; font-size:13px;")
    return lb


def big_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(34)
    return b


class MainWindow(TeachingMixin, FigureMixin, AccuracyMixin, RelocateMixin, QualityMixin, GroupsMixin, RegistrationMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.project: Project | None = None
        self.image: ImageEntry | None = None
        self.slide = None
        self.task: Task | None = None
        self._cells_cache: dict[str, pd.DataFrame] = {}

        # 看图窗口
        self.viewer = ViewerModel(title=APP_NAME)
        self.qt_viewer = QtViewer(self.viewer, show_welcome_screen=False)

        # 左侧步骤
        self.steps = QListWidget()
        self.steps.setFixedWidth(140)
        f = QFont(); f.setPointSize(11)
        self.steps.setFont(f)
        for s in STEPS:
            it = QListWidgetItem(s)
            it.setSizeHint(it.sizeHint().expandedTo(self.steps.sizeHint().__class__(150, 40)))
            self.steps.addItem(it)

        # 右侧面板
        self.stack = QStackedWidget()
        self.stack.addWidget(self._panel_project())
        self.stack.addWidget(self._panel_sections())
        self.stack.addWidget(self._panel_display())
        self.stack.addWidget(self._panel_detection())
        self.stack.addWidget(self._panel_coloc())
        self.stack.addWidget(self._panel_registration())
        self.stack.addWidget(self._panel_groups())
        self.stack.addWidget(self._panel_export())
        self.stack.addWidget(self._panel_figure())
        self.steps.currentRowChanged.connect(self._on_step)
        right = QScrollArea(); right.setWidgetResizable(True); right.setWidget(self.stack)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right.setMinimumWidth(380); right.setMaximumWidth(460)

        # 顶部：当前图片
        top = QHBoxLayout()
        top.addWidget(QLabel("当前图片："))
        self.image_combo = QComboBox(); self.image_combo.setMinimumWidth(380)
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        top.addWidget(self.image_combo, 1)
        self.project_label = QLabel("未打开项目")
        top.addWidget(self.project_label)
        self._teaching_toggle(top)

        # 图层列表和图层工具（选择、画矩形、删除等）
        tools = QWidget(); tl = QVBoxLayout(tools); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("图层工具"))
        from .tools_panel import LayerTools
        self.layer_tools = LayerTools(self.viewer)
        tl.addWidget(self.layer_tools, 1)
        self.qt_viewer.controls.hide()
        self.qt_viewer.layers.setContextMenuPolicy(Qt.NoContextMenu)
        tl.addWidget(QLabel("图层（点眼睛图标可显示/隐藏）"))
        tl.addWidget(self.qt_viewer.layers, 1)
        tools.setFixedWidth(220)
        view_row = QHBoxLayout(); view_row.addWidget(tools); view_row.addWidget(self.qt_viewer, 1)

        center = QWidget(); cl = QVBoxLayout(center); cl.setContentsMargins(0, 0, 0, 0)
        cl.addLayout(top); cl.addLayout(view_row, 1)

        split = QSplitter(); split.addWidget(self.steps); split.addWidget(center); split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        # 底部进度条
        self.progress = QProgressBar(); self.progress.setMaximumWidth(360); self.progress.hide()
        self.cancel_btn = QPushButton("取消"); self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(lambda: self.task and self.task.cancel())
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        from ..core.cpu import describe as _cpu_desc
        self.statusBar().showMessage("请先在“① 项目与图片”中新建或打开一个项目。　" + _cpu_desc())

        try:
            from napari.qt import get_stylesheet
            self.setStyleSheet(get_stylesheet("dark"))
        except Exception:  # noqa: BLE001
            pass
        self._add_lessons()
        self.steps.setCurrentRow(0)
        self._refresh_enabled()

    # ================================================================ 通用
    def _on_step(self, i):
        self.stack.setCurrentIndex(i)
        if i != 5 and getattr(self, "_reg_drag_on", False):
            self._reg_set_drag_mode(False)
        self._refresh_step_marks()
        if i == 1 and self.image is not None:
            self._refresh_section_table()
        if i == 2:
            self._sync_display_panel()
        if i == 3:
            self._sync_detection_panel()
        if i == 4:
            self._refresh_coloc_table()
        if i == 5:
            self._sync_registration_panel()
        if i == 6:
            self._sync_groups_panel()
        if i == 7 and self.project:
            self.exp_skip_layers.setChecked(self.project.export_skip_layers)
            self.exp_depth.setValue(self.project.export_max_depth)
        if i == 8:
            self._sync_figure_panel()

    def _refresh_enabled(self):
        has_p = self.project is not None
        has_img = has_p and bool(self.project.images)
        for i in range(1, self.steps.count()):
            it = self.steps.item(i)
            flags = it.flags()
            if has_img:
                it.setFlags(flags | Qt.ItemIsEnabled)
            else:
                it.setFlags(flags & ~Qt.ItemIsEnabled)
        self.btn_add_images.setEnabled(has_p)

    def error(self, text):
        QMessageBox.critical(self, APP_NAME, text)

    def info(self, text):
        QMessageBox.information(self, APP_NAME, text)

    def run_task(self, fn, on_done, *args, label="正在处理…"):
        if self.task is not None and self.task.isRunning():
            self.info("上一个任务还在运行，请等待完成或点击“取消”。")
            return
        self.task = Task(fn, *args)
        self.progress.setRange(0, 0); self.progress.show(); self.cancel_btn.show()
        self.statusBar().showMessage(label)

        def prog(done, total, text):
            self.progress.setRange(0, max(total, 1)); self.progress.setValue(done)
            if text:
                self.statusBar().showMessage(text)

        def ok(res):
            self.progress.hide(); self.cancel_btn.hide()
            self.statusBar().showMessage("完成。", 5000)
            on_done(res)

        def fail(msg):
            self.progress.hide(); self.cancel_btn.hide()
            self.statusBar().showMessage("任务未完成。", 5000)
            if msg == "任务已取消。":
                self.info("任务已取消。已经完成的部分（例如已识别的脑片）都已保存，下次会从没做完的地方继续。")
            else:
                from ..core.log import log_dir
                self.error(msg + f"\n\n详细的出错记录在：{log_dir() / '运行日志.txt'}"
                           + ("（项目文件夹里也有一份）" if self.project else ""))

        self.task.progress.connect(prog)
        self.task.finished_ok.connect(ok)
        self.task.failed.connect(fail)
        self.task.start()

    def ch_name(self, c):
        return self.project.channel_names[c] if self.project and c < len(self.project.channel_names) else f"通道{c + 1}"

    # ================================================================ ① 项目与图片
    def _panel_project(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("第一步：新建一个项目文件夹（用来保存设置和结果），然后添加要分析的图片。"
                         "一张图片上可以有多张脑片，下一步会自动拆分。\n"
                         "以后可以用“打开项目”继续上次的工作。"))
        row = QHBoxLayout()
        b1 = big_button("新建项目…"); b1.clicked.connect(self.new_project)
        b2 = big_button("打开项目…"); b2.clicked.connect(self.open_project)
        row.addWidget(b1); row.addWidget(b2); L.addLayout(row)
        self.btn_add_images = big_button("添加图片…"); self.btn_add_images.clicked.connect(self.add_images)
        L.addWidget(self.btn_add_images)
        bb = big_button("一键批量处理整个文件夹…"); bb.setStyleSheet("font-weight:bold;")
        bb.setToolTip("选一个放图片的文件夹，自动完成：拆分脑片 → 识别细胞 → 共标 → 配准 → 导出 Excel。")
        bb.clicked.connect(self.batch_dialog)
        L.addWidget(bb)

        L.addWidget(QLabel("项目中的图片："))
        self.images_table = QTableWidget(0, 3)
        self.images_table.setHorizontalHeaderLabels(["图片", "脑片数", "状态"])
        self.images_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.images_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.images_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.images_table.cellDoubleClicked.connect(lambda r, c: self.image_combo.setCurrentIndex(r))
        L.addWidget(self.images_table, 1)
        rm = QPushButton("从项目中移除选中的图片"); rm.clicked.connect(self.remove_image)
        L.addWidget(rm)
        self._relocate_widgets(L)

        self.channels_box = QGroupBox("通道名称（对应实验中的染色）")
        self.channels_form = QFormLayout(self.channels_box)
        L.addWidget(self.channels_box)

        zb = QGroupBox("z-stack（多层扫描）图片的合成方式"); zf = QVBoxLayout(zb)
        from ..core.slide_io import Z_MODE_NAMES
        self.z_combo = QComboBox()
        for k, v in Z_MODE_NAMES.items():
            self.z_combo.addItem(v, k)
        self.z_combo.currentIndexChanged.connect(self._set_z_mode)
        self.z_info = QLabel("单层扫描或扫描仪已做景深合成（EDF）的图片不受影响。"); self.z_info.setWordWrap(True)
        zf.addWidget(self.z_combo); zf.addWidget(self.z_info)
        L.addWidget(zb)

        row = QHBoxLayout()
        t1 = QPushButton("保存参数模板…"); t1.clicked.connect(self.save_template)
        t2 = QPushButton("导入参数模板…"); t2.clicked.connect(self.load_template)
        t1.setToolTip("把识别、共标、亮度等所有参数存成一个文件，发给同学或用在下一个项目，保证分析标准一致。")
        row.addWidget(t1); row.addWidget(t2); L.addLayout(row)
        stb = QPushButton("软件自检（安装后建议做一次，约 1–2 分钟）")
        stb.setToolTip("用软件自带的示例脑片把读图、细胞识别、图谱配准、导出各走一遍，"
                       "检查这台电脑上的安装是否完整、能否正常工作。不影响当前项目。")
        stb.clicked.connect(self.run_selftest)
        L.addWidget(stb)
        dg = QPushButton("出问题时：导出诊断包（发给开发者）…")
        dg.setToolTip("把运行日志、项目设置、电脑和软件版本打成一个 zip（不含图片和细胞数据，一般几百 KB），"
                      "出问题时把它发给开发者即可，不用截图、不用描述半天。")
        dg.clicked.connect(self.export_diagnostics)
        L.addWidget(dg)
        return w

    def run_selftest(self):
        from ..core import selftest

        def job(task):
            return selftest.run(task.report)

        def done(res):
            ok, text, _ = res
            from ..core.log import log_dir
            msg = text + f"\n\n结果已保存到：{log_dir() / '自检结果.txt'}"
            if ok:
                self.info(msg)
            else:
                self.error(msg + "\n请点“导出诊断包”，把 zip 文件发给开发者。")

        self.run_task(job, done, label="正在自检…")

    def export_diagnostics(self):
        from qtpy.QtWidgets import QInputDialog
        from ..core.diagnostics import make_bundle
        note, ok = QInputDialog.getMultiLineText(self, "导出诊断包",
                                                 "简单写一下出了什么问题（可以不写）：\n例如：点“自动配准”后卡住不动。", "")
        if not ok:
            return
        try:
            f = make_bundle(self.project, note=note)
        except Exception as e:  # noqa: BLE001
            from ..core.log import friendly
            self.error(f"诊断包导出失败：{friendly(e)}")
            return
        self.info(f"诊断包已保存：\n{f}\n\n请把这个 zip 文件发给开发者。")
        self._open_folder(f.parent)

    def _set_z_mode(self, _i=None):
        if self.project is None:
            return
        mode = self.z_combo.currentData()
        if mode == self.project.z_mode:
            return
        self.project.z_mode = mode; self.project.save()
        if self.image is not None:
            self._on_image_changed(self.project.images.index(self.image))
        self.statusBar().showMessage("z-stack 合成方式已修改。已识别的图片需要重新识别才会用新的方式。", 10000)

    def save_template(self):
        if self.project is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "保存参数模板", str(self.project.folder / "参数模板.json"), "参数模板 (*.json)")
        if f:
            self.project.save_template(f)
            self.statusBar().showMessage(f"参数模板已保存：{f}", 8000)

    def load_template(self):
        if self.project is None:
            self.info("请先新建或打开一个项目。")
            return
        f, _ = QFileDialog.getOpenFileName(self, "导入参数模板", str(self.project.folder), "参数模板 (*.json)")
        if not f:
            return
        try:
            notes = self.project.load_template(f)
        except Exception as e:  # noqa: BLE001
            self.error(f"模板无法导入：{e}")
            return
        self._refresh_channels_form(); self._sync_detection_panel()
        if self.slide is not None:
            self._show_image()
        self.info("参数模板已导入。" + ("\n" + "\n".join(notes) if notes else "")
                  + "\n已识别的图片需要重新识别（或点“④ 细胞识别”中的“按新阈值重新筛选”）才会使用新参数。")

    def new_project(self):
        d = QFileDialog.getExistingDirectory(self, "选择一个空文件夹作为项目文件夹")
        if not d:
            return
        if Project.is_project(d):
            if QMessageBox.question(self, APP_NAME, "这个文件夹里已经有项目了，要直接打开吗？") == QMessageBox.Yes:
                self._load_project(Project.load(d))
            return
        p = Project(folder=Path(d)); p.save()
        self._load_project(p)
        self.add_images()

    def open_project(self):
        d = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if not d:
            return
        if not Project.is_project(d):
            self.error("这个文件夹里没有找到项目文件（cfos_project.json）。")
            return
        try:
            self._load_project(Project.load(d))
        except Exception as e:  # noqa: BLE001
            self.error(f"项目文件无法读取：{e}")

    def _load_project(self, p: Project):
        self._q_levels = None
        from ..core.log import attach_project, log
        attach_project(p.folder); log.info("打开项目 %s", p.folder)
        self.project = p
        self._cells_cache.clear()
        self.project_label.setText(f"项目：{p.folder.name}")
        self._refresh_images()
        self._refresh_channels_form()
        self._refresh_enabled()
        self.statusBar().showMessage(f"已打开项目 {p.folder}")
        QTimer.singleShot(300, self._check_missing_images)
        self._refresh_step_marks()

    def add_images(self):
        if self.project is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片（可多选）", "", IMAGE_FILTER)
        if not files:
            return
        bad = []
        existing = {im.path for im in self.project.images}
        for f in files:
            if f in existing:
                continue
            try:
                sl = open_slide(f)
                if self.project.channel_names and len(sl.channels) != len(self.project.channel_names):
                    bad.append(f"{Path(f).name}：通道数（{len(sl.channels)}）与项目中其他图片不同")
                    sl.close(); continue
                self.project.ensure_channels(sl.channels)
                sl.close()
                self.project.images.append(ImageEntry(path=f))
            except Exception as e:  # noqa: BLE001
                bad.append(f"{Path(f).name}：{e}")
        self.project.save()
        self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
        if bad:
            self.error("以下图片无法加入：\n" + "\n".join(bad))
        if self.project.images and self.image is None:
            self.image_combo.setCurrentIndex(0)

    def add_images_programmatic(self, files):
        """测试和命令行使用。"""
        for f in files:
            sl = open_slide(f); self.project.ensure_channels(sl.channels); sl.close()
            self.project.images.append(ImageEntry(path=str(f)))
        self.project.save()
        self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
        self.image_combo.setCurrentIndex(0)

    def batch_dialog(self):
        from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLineEdit
        from ..core.pipeline import list_images
        dlg = QDialog(self); dlg.setWindowTitle("一键批量处理")
        L = QVBoxLayout(dlg)
        L.addWidget(hint("选择放图片的文件夹，软件会依次处理其中所有 .qptiff / .tif 图片，"
                         "全部使用当前项目里的识别参数（建议先用一张图片调好参数）。\n"
                         "完成后会在项目文件夹里生成 Excel，并列出需要人工检查配准的脑片。"))
        row = QHBoxLayout(); path = QLineEdit(); pick = QPushButton("选择文件夹…")
        row.addWidget(path, 1); row.addWidget(pick); L.addLayout(row)
        cnt = QLabel(""); L.addWidget(cnt)

        def choose():
            d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if d:
                path.setText(d)
                cnt.setText(f"找到 {len(list_images(d))} 张图片。")
        pick.clicked.connect(choose)
        c_reg = QCheckBox("配准到脑图谱"); c_reg.setChecked(True)
        c_nl = QCheckBox("配准后做非线性贴合"); c_nl.setChecked(True)
        c_redo = QCheckBox("已经处理过的图片也重新处理"); c_redo.setChecked(False)
        for c in (c_reg, c_nl, c_redo):
            L.addWidget(c)
        bx = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bx.button(QDialogButtonBox.Ok).setText("开始"); bx.button(QDialogButtonBox.Cancel).setText("取消")
        bx.accepted.connect(dlg.accept); bx.rejected.connect(dlg.reject); L.addWidget(bx)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            folder = path.text().strip()
            if not folder:
                return
            self.run_batch(folder, c_reg.isChecked(), c_nl.isChecked(), c_redo.isChecked())

    def run_batch(self, folder, do_reg=True, do_nl=True, redo=False):
        import datetime as _dt
        from ..core.pipeline import list_images, run_batch
        files = list_images(folder)
        if not files:
            self.error("这个文件夹里没有找到 .qptiff / .tif 图片。")
            return
        if self.project is None:
            pf = Path(folder) / "cFos分析项目"
            self._load_project(Project.load(pf) if Project.is_project(pf) else Project(folder=pf))
            self.project.save()
        project = self.project
        xlsx = project.folder / f"批量结果_{_dt.datetime.now():%Y%m%d_%H%M}.xlsx"

        def job(task):
            chk = run_batch(project, files, do_reg, do_nl, redo, report=task.report, cancel=task.is_cancelled)
            task.report(1, 1, "正在导出 Excel…")
            export_excel(project, xlsx)
            return chk

        def done(chk):
            self._cells_cache.clear(); self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
            self._q_levels = None; self._refresh_section_table()
            bad = [c for c in chk if c.get("质量") == "可疑" or c["需要检查"].startswith("出错") or "未配准" in c["需要检查"]]
            warn = [c for c in chk if c.get("质量") == "注意" and c not in bad]
            msg = f"批量处理完成：{len(files)} 张图片，{len(chk)} 张脑片。\n结果已保存到：\n{xlsx}\n\n"
            if bad:
                msg += "以下脑片质量“可疑”或出错，建议优先人工检查（Excel“按脑片汇总”中已标红）：\n" + "\n".join(
                    f"• {c['图片']} {c['脑片']}：{c['需要检查']}" for c in bad[:30]) + "\n\n"
            if warn:
                msg += f"另有 {len(warn)} 张脑片有轻微问题（标黄），可以在“② 脑片拆分”中点“检查所有脑片的质量”查看。\n"
            if not bad and not warn:
                msg += "自动质量检查没有发现问题（配准、模糊、过曝、组织缺损），仍建议抽查几张。"
            self.info(msg)

        self.run_task(job, done, label="正在批量处理…")

    def remove_image(self):
        r = self.images_table.currentRow()
        if r < 0 or self.project is None:
            return
        im = self.project.images[r]
        if QMessageBox.question(self, APP_NAME, f"从项目中移除 {im.key}？（不会删除原图）") != QMessageBox.Yes:
            return
        self.project.images.pop(r); self.project.save()
        self._refresh_images(); self._refresh_enabled()

    def _refresh_images(self):
        p = self.project
        self.images_table.setRowCount(len(p.images))
        cur = self.image_combo.currentIndex()
        self.image_combo.blockSignals(True); self.image_combo.clear()
        for i, im in enumerate(p.images):
            self.images_table.setItem(i, 0, QTableWidgetItem(Path(im.path).name))
            self.images_table.setItem(i, 1, QTableWidgetItem(str(len(im.sections)) if im.sections else "—"))
            st = QTableWidgetItem(im.status)
            if not Path(im.path).exists():
                st.setText("找不到原图"); st.setForeground(Qt.red)
                st.setToolTip(f"原路径：{im.path}\n点下方“重新定位找不到的图片…”找回。")
            elif im.error:
                st.setToolTip(im.error); st.setText(f"出错：{im.error[:40]}")
                st.setForeground(Qt.red)
            self.images_table.setItem(i, 2, st)
            self.image_combo.addItem(f"{i + 1}. {Path(im.path).name}")
        self.image_combo.blockSignals(False)
        if p.images:
            idx = cur if 0 <= cur < len(p.images) else 0
            if self.image is None or self.image not in p.images:
                self.image_combo.setCurrentIndex(idx); self._on_image_changed(idx)
            else:
                self.image_combo.blockSignals(True)
                self.image_combo.setCurrentIndex(p.images.index(self.image))
                self.image_combo.blockSignals(False)
        else:
            self.image = None; self.viewer.layers.clear()

    def _refresh_channels_form(self):
        while self.channels_form.rowCount():
            self.channels_form.removeRow(0)
        if not self.project:
            return
        self.z_combo.blockSignals(True)
        self.z_combo.setCurrentIndex(max(0, self.z_combo.findData(self.project.z_mode)))
        self.z_combo.blockSignals(False)
        for c, nm in enumerate(self.project.channel_names):
            cb = QComboBox(); cb.setEditable(True); cb.addItems(STANDARD_CHANNELS)
            cb.setCurrentText(nm)
            cb.currentTextChanged.connect(lambda t, c=c: self._rename_channel(c, t))
            self.channels_form.addRow(f"图像第 {c + 1} 通道：", cb)

    def _rename_channel(self, c, text):
        if not text.strip():
            return
        self.project.channel_names[c] = text.strip(); self.project.save()
        if self.slide is not None:
            self._show_image()

    # ================================================================ 看图
    def _on_image_changed(self, idx):
        if self.project is None or not (0 <= idx < len(self.project.images)):
            return
        self.image = self.project.images[idx]
        if self.slide is not None:
            self.slide.close()
        if not Path(self.image.path).exists():
            self.slide = None; self.viewer.layers.clear()
            self.statusBar().showMessage(f"找不到原图：{self.image.path}。请在“① 项目与图片”中点“重新定位找不到的图片…”。")
            return
        try:
            self.slide = open_slide(self.image.path)
        except Exception as e:  # noqa: BLE001
            self.slide = None
            self.error(f"图片无法打开：{e}")
            return
        self.z_info.setText(f"当前图片是 z-stack（{self.slide.n_z} 层），正在使用“{self.z_combo.currentText()}”。"
                            if self.slide.n_z > 1 else "当前图片是单层图像（或扫描仪已做景深合成），不需要设置。")
        self._show_image()
        if self.stack.currentIndex() == 1:
            self._refresh_section_table()
        if self.stack.currentIndex() == 5:
            self._sync_registration_panel()

    def _show_image(self):
        self.viewer.layers.clear()
        sl = self.slide
        levels = [sl._arr(lv) for lv in range(sl.n_levels)]
        names = [self.ch_name(c) for c in range(len(sl.channels))]
        cmaps = [COLORMAP.get(n, "gray") for n in names]
        if levels[0].ndim == 2:  # 单通道图像
            layers = [self.viewer.add_image(levels, multiscale=True, name=names[0], colormap=cmaps[0],
                                            blending="additive")]
        else:
            layers = self.viewer.add_image(levels, channel_axis=0, multiscale=True, name=names,
                                           colormap=cmaps, blending="additive")
        for c, layer in enumerate(layers):
            d = self.project.display.get(c)
            if d is not None and d.high > d.low:
                layer.contrast_limits = (d.low, d.high); layer.gamma = d.gamma; layer.visible = d.visible
            else:
                self._auto_contrast(c, layer)
        self.image_layers = layers
        self._draw_sections()
        self._draw_polygons()
        self._draw_cells()
        self._draw_figure_rois(force=self.stack.currentIndex() == 8)
        self.viewer.reset_view()

    def _auto_contrast(self, c, layer=None):
        sl = self.slide
        lv = sl.level_for_pixel_size(8.0)
        a = sl.read(lv, None, [c])[0]
        v = a[a > 0] if (a > 0).any() else a.ravel()
        lo, hi = np.percentile(v, [0.5, 99.7])
        hi = max(hi, lo + 1)
        d = self.project.display[c]
        d.low, d.high, d.gamma = float(lo), float(hi), 1.0
        if layer is None:
            layer = self.image_layers[c]
        layer.contrast_limits = (d.low, d.high); layer.gamma = 1.0
        self.project.save()

    def visible_bbox0(self):
        """当前视野的范围（第 0 层像素坐标 x, y, w, h）。"""
        cam = self.viewer.camera
        cy, cx = cam.center[-2], cam.center[-1]
        cw = self.qt_viewer.canvas.size  # (h, w)
        size = cw if isinstance(cw, tuple) else (self.qt_viewer.height(), self.qt_viewer.width())
        h, w = size[0] / cam.zoom, size[1] / cam.zoom
        return int(cx - w / 2), int(cy - h / 2), int(w), int(h)

    # ================================================================ ② 脑片拆分
    def _panel_sections(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("软件会自动找出图片上每一张脑片，并用黄色方框标出。\n"
                         "• 方框不准时：在左侧图层列表选中“脑片”图层，用“选择 / 移动方框”“拖动顶点调整大小”调整；"
                         "也可以“画新的矩形”，或选中后点“删除选中的方框”。\n"
                         "• 调整完点“保存方框修改”。\n• 破损或不想分析的脑片，在下表中取消“分析”勾选。"))
        b1 = big_button("自动识别当前图片的脑片"); b1.clicked.connect(lambda: self.auto_sections(False))
        b2 = big_button("自动识别所有图片的脑片"); b2.clicked.connect(lambda: self.auto_sections(True))
        b3 = big_button("保存方框修改"); b3.clicked.connect(self.save_section_edits)
        for b in (b1, b2, b3):
            L.addWidget(b)
        self.sec_table = QTableWidget(0, 4)
        self.sec_table.setHorizontalHeaderLabels(["脑片", "分析", "大小 (mm)", "质量"])
        self.sec_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sec_table.cellDoubleClicked.connect(self._zoom_section)
        self.sec_table.itemChanged.connect(self._section_item_changed)
        L.addWidget(self.sec_table, 1)
        L.addWidget(QLabel("提示：双击表格中的一行可以放大到这张脑片。"))
        self._quality_widgets(L)

        ex = QGroupBox("排除区域 / 自定义分析区域"); el = QVBoxLayout(ex)
        lb = QLabel("• 排除区域：折叠、气泡、破损、边缘发亮等，区域内的细胞和面积都不计入。\n"
                    "• 自定义区域：不依赖脑图谱，自己圈出要统计的区域（Excel 中多一张“按自定义区域”表）。\n"
                    "画法：点按钮后在图上逐个点击顶点，双击结束一个多边形；画完点“保存区域”。")
        lb.setWordWrap(True); el.addWidget(lb)
        r1 = QHBoxLayout()
        bx1 = QPushButton("画排除区域"); bx1.clicked.connect(lambda: self._edit_polygons("排除区域"))
        bx2 = QPushButton("画自定义区域"); bx2.clicked.connect(lambda: self._edit_polygons("自定义区域"))
        r1.addWidget(bx1); r1.addWidget(bx2); el.addLayout(r1)
        bx3 = QPushButton("保存区域"); bx3.clicked.connect(self.save_polygons); el.addWidget(bx3)
        el.addWidget(QLabel("自定义区域名称（双击可改名）："))
        self.roi_list = QListWidget(); self.roi_list.setMaximumHeight(90)
        self.roi_list.itemChanged.connect(self._rename_roi)
        el.addWidget(self.roi_list)
        L.addWidget(ex)
        return w

    def _draw_polygons(self):
        for nm in ("排除区域", "自定义区域"):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        if self.image is None or self.slide is None:
            return
        lw = max(self.slide.level_shapes[0]) / 400
        ex = [np.asarray(p, float)[:, ::-1] for p in self.image.exclusions if len(p) >= 3]
        self.viewer.add_shapes(ex, shape_type="polygon", name="排除区域", edge_color="#ffdc00",
                               face_color=[1, 0.86, 0, 0.25], edge_width=lw)
        rois = [r for r in self.image.rois if len(r["polygon"]) >= 3]
        self.viewer.add_shapes([np.asarray(r["polygon"], float)[:, ::-1] for r in rois], shape_type="polygon",
                               name="自定义区域", edge_color="cyan", face_color="transparent", edge_width=lw,
                               features={"name": [r["name"] for r in rois]},
                               text={"string": "{name}", "color": "cyan", "size": 14})
        self._refresh_roi_list()

    def _refresh_roi_list(self):
        self.roi_list.blockSignals(True); self.roi_list.clear()
        for r in (self.image.rois if self.image else []):
            it = QListWidgetItem(r["name"]); it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.roi_list.addItem(it)
        self.roi_list.blockSignals(False)

    def _rename_roi(self, item):
        i = self.roi_list.row(item)
        if self.image and 0 <= i < len(self.image.rois) and item.text().strip():
            self.image.rois[i]["name"] = item.text().strip()
            self.project.save(); self._draw_polygons()

    def _edit_polygons(self, name):
        if self.image is None:
            return
        if name not in self.viewer.layers:
            self._draw_polygons()
        layer = self.viewer.layers[name]
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon"
        self.statusBar().showMessage(f"正在画“{name}”：在图上逐个点击顶点，双击结束。画完点“保存区域”。", 15000)

    def save_polygons(self):
        if self.image is None:
            return
        def polys(nm):
            if nm not in self.viewer.layers:
                return []
            return [[(float(p[-1]), float(p[-2])) for p in d] for d in self.viewer.layers[nm].data if len(d) >= 3]
        self.image.exclusions = polys("排除区域")
        old = {i: r["name"] for i, r in enumerate(self.image.rois)}
        used = set(old.values())
        rois = []
        for i, poly in enumerate(polys("自定义区域")):
            nm = old.get(i)
            if not nm:
                k = 1
                while f"区域{k}" in used:
                    k += 1
                nm = f"区域{k}"; used.add(nm)
            rois.append({"name": nm, "polygon": poly})
        self.image.rois = rois
        self.project.save(); self._draw_polygons()
        self.statusBar().showMessage(f"已保存：{len(self.image.exclusions)} 个排除区域，{len(rois)} 个自定义区域。"
                                     "导出 Excel 时生效，不需要重新识别。", 10000)

    def auto_sections(self, all_images: bool):
        targets = list(self.project.images) if all_images else [self.image]

        def job(task):
            out = {}
            for i, im in enumerate(targets):
                task.report(i, len(targets), f"正在识别脑片：{Path(im.path).name}")
                sl = open_slide(im.path)
                out[im.path] = find_sections(sl)
                sl.close()
            return out

        def done(res):
            for im in targets:
                im.sections = res[im.path]
                im.status = f"已拆分 {len(im.sections)} 张脑片"
            self.project.save(); self._refresh_images()
            self._draw_sections(); self._refresh_section_table()
            n = sum(len(v) for v in res.values())
            self.statusBar().showMessage(f"共识别出 {n} 张脑片。", 8000)

        self.run_task(job, done, label="正在识别脑片…")

    def _draw_sections(self):
        if "脑片" in self.viewer.layers:
            self.viewer.layers.remove("脑片")
        if self.image is None:
            return
        rects, names = [], []
        for s in self.image.sections:
            x, y, w, h = s.bbox0
            rects.append(np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], float))
            names.append(s.name + ("" if s.enabled else "（不分析）"))
        lw = max(self.slide.level_shapes[0]) / 300
        self.viewer.add_shapes(rects, shape_type="rectangle", name="脑片", edge_color="yellow",
                               face_color="transparent", edge_width=lw,
                               features={"name": names},
                               text={"string": "{name}", "color": "yellow", "size": 14, "anchor": "upper_left"})

    def _refresh_section_table(self):
        self._q_levels = None  # 每次刷新重新读取质量检查结果（配准等改动后过期的会显示“—”）
        self.sec_table.blockSignals(True)
        secs = self.image.sections if self.image else []
        self.sec_table.setRowCount(len(secs))
        px = self.slide.pixel_size_um if self.slide else 0.25
        for i, s in enumerate(secs):
            self.sec_table.setItem(i, 0, QTableWidgetItem(s.name))
            chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if s.enabled else Qt.Unchecked)
            self.sec_table.setItem(i, 1, chk)
            self.sec_table.setItem(i, 2, QTableWidgetItem(f"{s.bbox0[2] * px / 1000:.1f} × {s.bbox0[3] * px / 1000:.1f}"))
            self.sec_table.setItem(i, 3, self._quality_cell(self.image, s))
        self.sec_table.blockSignals(False)

    def _section_item_changed(self, item):
        r, c = item.row(), item.column()
        s = self.image.sections[r]
        if c == 1:
            s.enabled = item.checkState() == Qt.Checked
        elif c == 0 and item.text().strip():
            s.name = item.text().strip()
        self.project.save(); self._draw_sections()

    def _zoom_section(self, r, _c):
        s = self.image.sections[r]
        x, y, w, h = s.bbox0
        self.viewer.camera.center = (y + h / 2, x + w / 2)
        sz = (self.qt_viewer.height(), self.qt_viewer.width())
        self.viewer.camera.zoom = 0.9 * min(sz[0] / h, sz[1] / w)

    def save_section_edits(self):
        if "脑片" not in self.viewer.layers:
            return
        layer = self.viewer.layers["脑片"]
        old = list(self.image.sections)
        new = []
        used = set()
        for i, poly in enumerate(layer.data):
            ys, xs = poly[:, -2], poly[:, -1]
            x0, y0 = max(0, xs.min()), max(0, ys.min())
            bbox = (int(x0), int(y0), int(xs.max() - x0), int(ys.max() - y0))
            if bbox[2] < 50 or bbox[3] < 50:
                continue
            prev = old[i] if i < len(old) else None
            if prev is not None and tuple(prev.bbox0) == bbox:
                s = prev
            else:
                s = Section(id=i + 1, bbox0=bbox, contour0=contour_in_bbox(self.slide, bbox),
                            name=prev.name if prev else "", enabled=prev.enabled if prev else True)
            new.append(s)
        for s in new:
            used.add(s.name)
        k = 1
        for s in new:
            if not s.name:
                while f"S{k:02d}" in used:
                    k += 1
                s.name = f"S{k:02d}"; used.add(s.name)
        self.image.sections = new
        self.project.save(); self._refresh_images(); self._draw_sections(); self._refresh_section_table()
        self.statusBar().showMessage("方框修改已保存。", 5000)

    # ================================================================ ③ 显示调整
    def _panel_display(self):
        w = QWidget(); self.display_layout = QVBoxLayout(w)
        self.display_layout.addWidget(hint("拖动滑块调整每个通道的亮度范围：\n"
                                           "• “最暗”以下显示为黑色，“最亮”以上显示为最亮。\n"
                                           "• “伽马”小于 1 会让暗处更亮。\n"
                                           "这些设置只影响显示和导出的图片，不会改变细胞识别结果。"))
        self.display_rows = QVBoxLayout(); self.display_layout.addLayout(self.display_rows)
        self.display_layout.addStretch(1)
        return w

    def _sync_display_panel(self):
        while self.display_rows.count():
            it = self.display_rows.takeAt(0)
            wd = it.widget()
            if wd is not None:
                wd.hide(); wd.setParent(None); wd.deleteLater()
        if not self.slide:
            return
        vmax = 255 if self.slide._arr(0).dtype == np.uint8 else 65535
        for c in range(len(self.slide.channels)):
            d = self.project.display[c]
            box = QGroupBox(self.ch_name(c)); f = QFormLayout(box)
            vis = QCheckBox("显示"); vis.setChecked(d.visible)
            lo = QSlider(Qt.Horizontal); lo.setRange(0, vmax); lo.setValue(int(d.low))
            hi = QSlider(Qt.Horizontal); hi.setRange(0, vmax); hi.setValue(int(d.high))
            gm = QDoubleSpinBox(); gm.setRange(0.2, 3.0); gm.setSingleStep(0.1); gm.setValue(d.gamma)
            auto = QPushButton("自动")
            lo_lab, hi_lab = QLabel(str(int(d.low))), QLabel(str(int(d.high)))
            r1 = QHBoxLayout(); r1.addWidget(lo, 1); r1.addWidget(lo_lab)
            r2 = QHBoxLayout(); r2.addWidget(hi, 1); r2.addWidget(hi_lab)
            f.addRow(vis, auto); f.addRow("最暗", r1); f.addRow("最亮", r2); f.addRow("伽马", gm)

            def apply(_=None, c=c, vis=vis, lo=lo, hi=hi, gm=gm, lo_lab=lo_lab, hi_lab=hi_lab):
                d = self.project.display[c]
                d.visible = vis.isChecked(); d.low = float(lo.value()); d.high = float(max(hi.value(), lo.value() + 1))
                d.gamma = float(gm.value())
                lo_lab.setText(str(int(d.low))); hi_lab.setText(str(int(d.high)))
                layer = self.image_layers[c]
                layer.visible = d.visible; layer.contrast_limits = (d.low, d.high); layer.gamma = d.gamma
                self._save_later()

            def do_auto(_=None, c=c, lo=lo, hi=hi, gm=gm):
                self._auto_contrast(c)
                d = self.project.display[c]
                for wdg, v in ((lo, d.low), (hi, d.high)):
                    wdg.blockSignals(True); wdg.setValue(int(v)); wdg.blockSignals(False)
                gm.setValue(1.0); apply()

            vis.toggled.connect(apply); lo.valueChanged.connect(apply); hi.valueChanged.connect(apply)
            gm.valueChanged.connect(apply); auto.clicked.connect(do_auto)
            self.display_rows.addWidget(box)

    def _save_later(self):
        if not hasattr(self, "_save_timer"):
            self._save_timer = QTimer(self); self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(lambda: self.project and self.project.save())
        self._save_timer.start(500)

    # ================================================================ ④ 细胞识别
    def _panel_detection(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("每个通道单独识别阳性细胞（使用 StarDist 深度学习模型，不需要显卡）。\n"
                         "建议流程：先把看图窗口放大到一块有代表性的区域，点“预览当前视野”，"
                         "对照图片检查圈出的细胞是否合适，再调整参数；满意后点“识别当前图片”或“批量识别”。"))
        tb = QGroupBox("切片厚度"); tl = QVBoxLayout(tb)
        self.thick_combo = QComboBox()
        self.thick_combo.addItem("薄切片（≤ 20 µm）", "thin")
        self.thick_combo.addItem("厚切片（30–50 µm，如 40 µm 漂片）", "thick")
        bt = QPushButton("使用这个厚度的推荐参数"); bt.clicked.connect(self.apply_thickness)
        tl.addWidget(self.thick_combo); tl.addWidget(bt)
        tl.addWidget(QLabel("厚切片推荐：识别前减背景 25 µm（去掉焦平面外的雾），去掉比周围亮不到 1.5 倍的模糊细胞，"
                            "共标距离放宽到 8 µm。z-stack 图片请在“① 项目与图片”中选择合成方式。"))
        tl.itemAt(2).widget().setWordWrap(True)
        L.addWidget(tb)
        self.det_rows = QVBoxLayout(); L.addLayout(self.det_rows)
        b0 = big_button("预览当前视野的减背景/去卷积效果"); b0.clicked.connect(self.preview_enhance)
        L.addWidget(b0)
        b1 = big_button("预览当前视野的细胞识别"); b1.clicked.connect(self.preview_detection)
        b2 = big_button("识别当前图片的全部脑片"); b2.clicked.connect(lambda: self.run_detection(False))
        b3 = big_button("批量识别项目中所有图片"); b3.clicked.connect(lambda: self.run_detection(True))
        b4 = big_button("按新阈值重新筛选（几秒钟，不用重新识别）"); b4.clicked.connect(self.refilter)
        b4.setToolTip("只改了面积、最低强度、自动阳性阈值、局部对比度时，点这里即可，不需要重新识别。\n"
                      "改了灵敏度、提亮、减背景、去卷积则需要重新识别。")
        for b in (b1, b2, b3, b4):
            L.addWidget(b)
        mb = QGroupBox("手动校正细胞（识别完成后）"); ml = QVBoxLayout(mb)
        lb = QLabel("先选通道，再点“补加”后在图上点击漏掉的细胞；或点“选择”后点击 / 拖框选中误识别的细胞，"
                    "按“删除选中”。改完点“保存手动校正”。校正会保留到以后（改阈值、重新识别后仍然有效），"
                    "每张脑片改了多少个会写进 Excel。")
        lb.setWordWrap(True); ml.addWidget(lb)
        r = QHBoxLayout(); r.addWidget(QLabel("通道：")); self.man_ch = QComboBox(); r.addWidget(self.man_ch, 1); ml.addLayout(r)
        r = QHBoxLayout()
        for t, fn in (("补加", lambda: self._man_mode("add")), ("选择", lambda: self._man_mode("select")),
                      ("删除选中", self._man_delete)):
            b = QPushButton(t); b.clicked.connect(fn); r.addWidget(b)
        ml.addLayout(r)
        b = QPushButton("保存手动校正"); b.clicked.connect(self.save_manual); ml.addWidget(b)
        self.man_info = QLabel(""); self.man_info.setWordWrap(True); ml.addWidget(self.man_info)
        L.addWidget(mb)
        self._accuracy_widgets(L)

        tb2 = QGroupBox("用自己的标注改进识别模型（可选）"); tl2 = QVBoxLayout(tb2)
        lb = QLabel("通用模型对你们的染色不够准时使用：①把看图窗口移到有代表性的地方，点“取一块并自动标注”；"
                    "②在左侧用画笔补画漏掉的细胞（每画一个新细胞前按 M 键或点“画新细胞”）、用橡皮擦掉错的；"
                    "③点“保存这块”。亮的、暗的、密集的区域各取几块，共 5–10 块后点“开始训练”（约 10 分钟）。")
        lb.setWordWrap(True); tl2.addWidget(lb)
        r = QHBoxLayout(); r.addWidget(QLabel("通道：")); self.ft_ch = QComboBox(); r.addWidget(self.ft_ch, 1); tl2.addLayout(r)
        self.ft_ch.currentIndexChanged.connect(lambda _: self._ft_refresh())
        r = QHBoxLayout()
        b = QPushButton("取一块并自动标注"); b.clicked.connect(self.ft_new_patch); r.addWidget(b)
        b = QPushButton("保存这块"); b.clicked.connect(self.ft_save_patch); r.addWidget(b)
        tl2.addLayout(r)
        b = QPushButton("开始训练"); b.clicked.connect(self.ft_train); tl2.addWidget(b)
        self.ft_info = QLabel(""); self.ft_info.setWordWrap(True); tl2.addWidget(self.ft_info)
        L.addWidget(tb2)
        self.det_summary = QLabel(""); self.det_summary.setWordWrap(True)
        L.addWidget(self.det_summary)
        L.addStretch(1)
        return w

    def _sync_detection_panel(self):
        while self.det_rows.count():
            it = self.det_rows.takeAt(0)
            wd = it.widget()
            if wd is not None:
                wd.hide(); wd.setParent(None); wd.deleteLater()
        if not self.project:
            return
        self.thick_combo.setCurrentIndex(max(0, self.thick_combo.findData(self.project.thickness)))
        for cb in (self.man_ch, self.ft_ch):
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(self.project.channel_names)
            if cur:
                cb.setCurrentText(cur)
            cb.blockSignals(False)
        self._ft_refresh()
        self._man_refresh_info()
        self._acc_sync()
        for c, nm in enumerate(self.project.channel_names):
            p = self.project.detection.channels[c]
            box = QGroupBox(nm); f = QFormLayout(box)
            on = QCheckBox("识别这个通道的细胞"); on.setChecked(p.detect)
            prob = QDoubleSpinBox(); prob.setRange(0.05, 0.95); prob.setSingleStep(0.05); prob.setValue(p.prob_thresh)
            prob.setToolTip("越小找到的细胞越多（包括较暗的），越大越严格。默认 0.5。")
            amin = QDoubleSpinBox(); amin.setRange(1, 5000); amin.setValue(p.min_area_um2); amin.setSuffix(" µm²")
            amax = QDoubleSpinBox(); amax.setRange(10, 20000); amax.setValue(p.max_area_um2); amax.setSuffix(" µm²")
            imin = QDoubleSpinBox(); imin.setRange(0, 65535); imin.setValue(p.min_intensity)
            imin.setToolTip("细胞内平均荧光强度低于此值的不计为阳性。0 表示不限制。")
            from ..core.finetune import list_models
            mdl = QComboBox(); mdl.addItem("通用模型（软件自带）", "")
            mdl.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); mdl.setMinimumContentsLength(12)
            for m in list_models(self.project):
                if m.get("channel") == nm:
                    mdl.addItem(f"{m['name']}（准确率 {m['f1_before']}→{m['f1_after']}）", m["name"])
            mdl.setCurrentIndex(max(0, mdl.findData(p.model)))
            mdl.setToolTip("“用自己的标注改进识别模型”训练出的模型会出现在这里。换模型后需要重新识别。")
            f.addRow(on); f.addRow("识别模型", mdl); f.addRow("识别灵敏度阈值", prob); f.addRow("最小面积", amin); f.addRow("最大面积", amax)
            f.addRow("最低平均强度", imin)
            gm = QDoubleSpinBox(); gm.setRange(0.2, 2.0); gm.setSingleStep(0.1); gm.setValue(p.gamma)
            gm.setToolTip("识别前把图像提亮的程度：小于 1 时暗的细胞更容易被识别（DAPI 建议 0.5）。只影响识别，不影响显示。")
            f.addRow("识别前提亮（伽马）", gm)
            bgf = QDoubleSpinBox(); bgf.setRange(0, 50); bgf.setSingleStep(0.5); bgf.setValue(p.bg_fold); bgf.setSuffix(" 倍背景")
            bgf.setSpecialValueText("不使用")
            bgf.setToolTip("自动阳性阈值：细胞内平均亮度至少是本脑片组织背景亮度的多少倍才算阳性。"
                           "适合 cFos 等活性标记，同一批实验请使用同一数值。0 表示不使用。")
            f.addRow("自动阳性阈值", bgf)
            mct = QDoubleSpinBox(); mct.setRange(0, 10); mct.setSingleStep(0.1); mct.setValue(p.min_contrast)
            mct.setSuffix(" 倍周围"); mct.setSpecialValueText("不使用")
            mct.setToolTip("局部对比度：细胞内平均亮度 ÷ 细胞外面一圈（3 µm 宽，不含其他细胞）的平均亮度。\n"
                           "厚切片中焦平面以外的细胞看起来是一团模糊的光晕，和周围差别小，设 1.3–1.8 可以去掉它们。\n"
                           "0 表示不使用。改这个值后点“按新阈值重新筛选”即可，不需要重新识别。")
            f.addRow("局部对比度下限", mct)
            bgr = QDoubleSpinBox(); bgr.setRange(0, 500); bgr.setSingleStep(5); bgr.setValue(p.bg_radius_um)
            bgr.setSuffix(" µm"); bgr.setSpecialValueText("不减背景")
            bgr.setToolTip("识别前去掉大范围的雾状背景。半径要比细胞大，一般 20–50 µm。只影响识别，强度测量仍用原图。")
            f.addRow("识别前减背景", bgr)
            dcv = QCheckBox("识别前去卷积（让细胞边界更清楚，会变慢）"); dcv.setChecked(p.deconv)
            dcb = QDoubleSpinBox(); dcb.setRange(0.1, 3.0); dcb.setSingleStep(0.1); dcb.setValue(p.deconv_blur_um); dcb.setSuffix(" µm")
            dcb.setToolTip("图像的模糊程度。太大会让细胞变成空心圆环，样图建议 0.4 µm。")
            dci = QDoubleSpinBox(); dci.setDecimals(0); dci.setRange(1, 50); dci.setValue(p.deconv_iter)
            f.addRow(dcv); f.addRow("去卷积模糊程度", dcb); f.addRow("去卷积迭代次数", dci)

            def apply(_=None, c=c, on=on, prob=prob, amin=amin, amax=amax, imin=imin, gm=gm, bgf=bgf,
                      bgr=bgr, dcv=dcv, dcb=dcb, dci=dci, mct=mct, mdl=mdl):
                p = self.project.detection.channels[c]
                p.min_contrast = mct.value()
                p.model = mdl.currentData() or ""
                p.gamma = gm.value(); p.bg_fold = bgf.value()
                p.bg_radius_um = bgr.value(); p.deconv = dcv.isChecked()
                p.deconv_blur_um = dcb.value(); p.deconv_iter = int(dci.value())
                p.detect = on.isChecked(); p.prob_thresh = prob.value()
                p.min_area_um2 = amin.value(); p.max_area_um2 = amax.value(); p.min_intensity = imin.value()
                self._save_later()

            for wdg in (prob, amin, amax, imin, gm, bgf, bgr, dcb, dci, mct):
                wdg.valueChanged.connect(apply)
            on.toggled.connect(apply); dcv.toggled.connect(apply); mdl.currentIndexChanged.connect(apply)
            self.det_rows.addWidget(box)

    def preview_detection(self):
        if self.slide is None:
            self.info("请先在顶部选择一张图片。")
            return
        x, y, w, h = self.visible_bbox0()
        sl = self.slide
        maxw = int(2048 * self.project.detection.pixel_um / sl.pixel_size_um)
        if w > maxw * 1.5 or h > maxw * 1.5:
            cx, cy = x + w / 2, y + h / 2
            w = h = maxw
            x, y = int(cx - w / 2), int(cy - h / 2)
            self.statusBar().showMessage("视野太大，只预览视野中央的一块区域（黄色虚线框）。", 8000)
        H0, W0 = sl.level_shapes[0]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W0 - x), min(h, H0 - y)
        sec = Section(id=0, bbox0=(x, y, w, h), contour0=[], name="预览")
        path = self.image.path
        settings = self.project.detection

        def job(task):
            s = open_slide(path)
            try:
                return det.detect_section(s, sec, settings, progress=lambda d, t: task.report(d, t, "正在预览识别…"),
                                          cancel=task.is_cancelled)
            finally:
                s.close()

        def done(df):
            for nm in [l.name for l in self.viewer.layers if l.name.startswith("预览")]:
                self.viewer.layers.remove(nm)
            rect = np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], float)
            self.viewer.add_shapes([rect], shape_type="rectangle", name="预览区域", edge_color="yellow",
                                   face_color="transparent", edge_width=max(w, h) / 400, opacity=0.6)
            msg = []
            for c, nm in enumerate(self.project.channel_names):
                g = df[df["channel"] == c]
                msg.append(f"{nm}：{len(g)} 个")
                if len(g):
                    polys = [np.asarray(cn)[:, ::-1] for cn in g["contour"] if len(cn) > 2]
                    self.viewer.add_shapes(polys, shape_type="polygon", name=f"预览-{nm}",
                                           edge_color=POINT_COLOR.get(nm, "white"), face_color="transparent",
                                           edge_width=max(w, h) / 800)
            self.det_summary.setText("预览结果：" + "，".join(msg) + "。\n不满意可以调整参数后再预览一次。")

        self.run_task(job, done, label="正在预览识别…")

    def preview_enhance(self):
        if self.slide is None:
            self.info("请先在顶部选择一张图片。")
            return
        from ..core.deconv import enhance
        x, y, w, h = self.visible_bbox0()
        sl = self.slide
        H0, W0 = sl.level_shapes[0]
        lv = 0
        while w / sl.downsample(lv) > 2048 or h / sl.downsample(lv) > 2048:
            lv += 1
            if lv >= sl.n_levels - 1:
                break
        x, y = max(0, x), max(0, y)
        w, h = min(w, W0 - x), min(h, H0 - y)
        if w <= 0 or h <= 0:
            self.info("当前视野不在图片范围内，请先把图片移到视野中间再预览。")
            return
        chans = [c for c, p in self.project.detection.channels.items() if p.deconv or p.bg_radius_um > 0]
        if not chans:
            self.info("还没有任何通道勾选“去卷积”或设置“减背景”。")
            return
        params = {c: self.project.detection.channels[c] for c in chans}
        path = self.image.path
        ds = sl.downsample(lv)
        pix = sl.pixel_size_um * ds

        def job(task):
            s = open_slide(path)
            a = s.read(lv, (x, y, w, h), chans)
            s.close()
            out = []
            for k, c in enumerate(chans):
                task.report(k, len(chans), f"正在处理 {self.ch_name(c)}…")
                p = params[c]
                out.append(enhance(a[k], pix, p.deconv, p.deconv_blur_um, p.deconv_iter, p.bg_radius_um))
            return out

        def done(res):
            for nm in [l.name for l in self.viewer.layers if l.name.startswith("预处理-")]:
                self.viewer.layers.remove(nm)
            for c, im in zip(chans, res):
                nm = self.ch_name(c)
                lo, hi = np.percentile(im, [1, 99.7])
                self.viewer.add_image(im, name=f"预处理-{nm}", colormap=COLORMAP.get(nm, "gray"),
                                      blending="additive", scale=(ds, ds), translate=(y + ds / 2, x + ds / 2),
                                      contrast_limits=(float(lo), float(max(hi, lo + 1))))
                if c < len(self.image_layers):
                    self.image_layers[c].visible = False
            self.statusBar().showMessage("已显示预处理后的图像（“预处理-”图层）。在左侧图层列表中点眼睛图标可与原图对比。", 15000)

        self.run_task(job, done, label="正在预览预处理效果…")

    def run_detection(self, all_images: bool):
        targets = list(self.project.images) if all_images else [self.image]
        if not all_images and not self.image.sections:
            self.info("当前图片还没有拆分脑片，请先到“② 脑片拆分”。")
            return
        settings = self.project.detection
        coloc_p = self.project.coloc
        project = self.project

        def job(task):
            from ..core.pipeline import detect_image
            from ..core.log import friendly, log
            import traceback
            summary, errors, reused = {}, [], 0
            for i, im in enumerate(targets):
                try:
                    r = detect_image(project, im, report=task.report, cancel=task.is_cancelled,
                                     progress_range=(i * 100, i * 100 + 100), total=len(targets) * 100)
                    reused += r["reused"]
                    summary[im.key] = project.load_cells(im)
                except InterruptedError:
                    project.save()
                    raise
                except Exception as e:  # noqa: BLE001  一张图出错不影响其他图片
                    log.error("识别 %s 出错\n%s", im.path, traceback.format_exc())
                    im.error = friendly(e); im.status = "出错"
                    errors.append(f"{Path(im.path).name}：{im.error}")
            project.save()
            return summary, errors, reused

        def done(out):
            res, errors, reused = out
            self.project.save(); self._refresh_images()
            for k_, df in res.items():
                self._cells_cache[k_] = df
            self._draw_cells()
            lines = []
            if reused:
                lines.append(f"（有 {reused} 张脑片的识别参数没有变化，直接使用了上次的结果）")
            if errors:
                self.error("以下图片识别出错，已跳过，其他图片正常完成：\n" + "\n".join(errors))
            for k_, df in res.items():
                if df is None:
                    continue
                cnt = ", ".join(f"{self.ch_name(c)} {int((df['channel'] == c).sum())}" for c in range(len(self.project.channel_names))) if len(df) else "无"
                lines.append(f"{k_}：{cnt}")
            self.det_summary.setText("识别完成：\n" + "\n".join(lines))
            for nm in [l.name for l in self.viewer.layers if l.name.startswith("预览")]:
                self.viewer.layers.remove(nm)

        self.run_task(job, done, label="正在识别细胞…")

    def apply_thickness(self):
        th = self.thick_combo.currentData()
        self.project.apply_thickness_preset(th)
        self._sync_detection_panel()
        self.info(("已使用厚切片推荐参数" if th == "thick" else "已使用薄切片推荐参数")
                  + "。\n建议先放大到一块区域点“预览当前视野的细胞识别”检查效果，再识别全部脑片。")

    def refilter(self):
        from ..core.pipeline import refilter_all
        project = self.project

        def done(n):
            self._cells_cache.clear(); self._draw_cells(); self._refresh_coloc_table()
            lines = []
            for im in project.images:
                df = self.cells_for(im)
                if df is not None and len(df):
                    lines.append(f"{im.key}：" + ", ".join(f"{self.ch_name(c)} {int((df['channel'] == c).sum())}"
                                                         for c in range(len(project.channel_names))))
            self.det_summary.setText(f"已按新阈值重新筛选 {n} 张图片：\n" + "\n".join(lines))

        self.run_task(lambda task: refilter_all(project, task.report), done, label="正在重新筛选…")

    # ---------------- 手动校正
    def _man_layer(self, create=True):
        nm = self.man_ch.currentText()
        if not nm or self.image is None:
            return None
        name = f"细胞-{nm}"
        if name not in self.viewer.layers:
            if not create:
                return None
            size = 10 / self.slide.pixel_size_um
            layer = self.viewer.add_points(np.zeros((0, 2)), name=name, size=size, face_color="transparent",
                                           border_color=POINT_COLOR.get(nm, "white"), border_width=0.15,
                                           features={"cid": np.zeros(0, int)})
            layer.feature_defaults = {"cid": -1}
        return self.viewer.layers[name]

    def _man_mode(self, mode):
        if self.cells_for(self.image) is None:
            self.info("当前图片还没有识别结果，请先识别细胞。")
            return
        layer = self._man_layer()
        if layer is None:
            return
        layer.visible = True
        self.viewer.layers.selection.active = layer
        layer.mode = mode
        self.statusBar().showMessage("补加：在图上点击细胞中心。" if mode == "add" else
                                     "选择：点击细胞，或按住鼠标拖框选中多个，然后点“删除选中”。", 15000)

    def _man_delete(self):
        layer = self._man_layer(create=False)
        if layer is not None and layer.selected_data:
            n = len(layer.selected_data)
            layer.remove_selected()
            self.statusBar().showMessage(f"已删除 {n} 个（还没保存）。", 8000)

    def save_manual(self):
        from ..core.manual import load_manual, save_manual, section_of, measure_added
        from ..core.pipeline import refilter_all
        if self.image is None:
            return
        df = self.cells_for(self.image)
        if df is None:
            return
        man = load_manual(self.project, self.image)
        px = self.slide.pixel_size_um
        n_add = n_del = n_out = 0
        for c, nm in enumerate(self.project.channel_names):
            name = f"细胞-{nm}"
            if name not in self.viewer.layers:
                continue
            layer = self.viewer.layers[name]
            cid = layer.features["cid"].to_numpy() if "cid" in layer.features else np.full(len(layer.data), -1)
            orig = df[df["channel"] == c]
            gone = orig.index.difference(pd.Index(cid[cid >= 0]))
            for i in gone:
                r = df.loc[i]
                if int(r.get("manual", 0)) == 1:  # 删掉的是之前手动补加的：撤销那条补加记录
                    man["added"] = [a for a in man["added"] if not (int(a["channel"]) == c and
                                    abs(a["x"] - r["x"]) < 1 and abs(a["y"] - r["y"]) < 1)]
                else:
                    man["removed"].append({"channel": c, "x": float(r["x"]), "y": float(r["y"]), "section": r["section"]})
                n_del += 1
            for (y, x) in layer.data[cid < 0]:
                sec = section_of(self.image, x, y)
                if not sec:
                    n_out += 1; continue
                means, cont = measure_added(self.slide, x, y, px)
                man["added"].append({"channel": c, "x": float(x), "y": float(y), "section": sec,
                                     "means": means, "contour": cont})
                n_add += 1
        save_manual(self.project, self.image, man)
        project = self.project

        def done(_):
            self._cells_cache.clear(); self._draw_cells(); self._refresh_coloc_table(); self._man_refresh_info()
            msg = f"已保存：补加 {n_add} 个，删除 {n_del} 个。"
            if n_out:
                msg += f"\n有 {n_out} 个补加的点不在任何脑片方框内，已忽略。"
            self.statusBar().showMessage(msg.replace("\n", " "), 10000)

        self.run_task(lambda task: refilter_all(project, task.report), done, label="正在更新细胞和共标…")

    # ---------------- 微调模型
    def _ft_refresh(self):
        if self.project is None or not self.ft_ch.currentText():
            return
        from ..core.finetune import list_patches, list_models
        c = self.ft_ch.currentIndex()
        n = len(list_patches(self.project, c))
        ms = [m for m in list_models(self.project) if m.get("channel") == self.ft_ch.currentText()]
        t = f"“{self.ft_ch.currentText()}”已有 {n} 个标注块。"
        if ms:
            m = ms[-1]
            t += f"\n最近训练的模型：{m['name']}，准确率（F1）{m['f1_before']} → {m['f1_after']}。"
        self.ft_info.setText(t)

    def ft_new_patch(self):
        if self.slide is None:
            self.info("请先在顶部选择一张图片。"); return
        from ..core.finetune import make_patch
        c = self.ft_ch.currentIndex()
        cam = self.viewer.camera
        center = (cam.center[-1], cam.center[-2])
        project, image, path = self.project, self.image, self.image.path

        def job(task):
            task.report(0, 1, "正在自动识别这一块…")
            sl = open_slide(path)
            try:
                return make_patch(project, sl, image, center, c)
            finally:
                sl.close()

        def done(res):
            x, lab, meta = res
            self._ft_patch = (c, x, meta)
            for nm in ("标注块", "标注块-图像"):
                if nm in self.viewer.layers:
                    self.viewer.layers.remove(nm)
            ds = meta["ds"]
            tr = (meta["y0"] + ds / 2, meta["x0"] + ds / 2)
            self.viewer.add_image(x, name="标注块-图像", colormap="gray", scale=(ds, ds), translate=tr,
                                  contrast_limits=(0, 1), opacity=1.0)
            lay = self.viewer.add_labels(lab, name="标注块", scale=(ds, ds), translate=tr, opacity=0.45)
            lay.brush_size = max(3, int(4 / (self.slide.pixel_size_um * ds)))
            self.viewer.layers.selection.active = lay
            n = meta["n"] * ds
            self.viewer.camera.center = (meta["y0"] + n / 2, meta["x0"] + n / 2)
            self.viewer.camera.zoom = 0.9 * min(self.qt_viewer.height(), self.qt_viewer.width()) / n
            self.statusBar().showMessage(f"自动圈出了 {int(lab.max())} 个细胞。请修改后点“保存这块”。", 15000)

        self.run_task(job, done, label="正在准备标注块…")

    def ft_save_patch(self):
        if not getattr(self, "_ft_patch", None) or "标注块" not in self.viewer.layers:
            self.info("请先点“取一块并自动标注”。"); return
        from ..core.finetune import save_patch
        c, x, meta = self._ft_patch
        lab = np.asarray(self.viewer.layers["标注块"].data)
        p = save_patch(self.project, c, x, lab, meta)
        for nm in ("标注块", "标注块-图像"):
            self.viewer.layers.remove(nm)
        self._ft_patch = None
        self._ft_refresh()
        self.statusBar().showMessage(f"已保存 {p.name}（{len(np.unique(lab)) - 1} 个细胞）。", 10000)

    def ft_train(self):
        from ..core.finetune import finetune, list_patches
        c = self.ft_ch.currentIndex()
        n = len(list_patches(self.project, c))
        if n < 3:
            self.info(f"至少需要 3 个标注块（现在有 {n} 个），建议 5–10 个。"); return
        project = self.project

        def done(info):
            self._ft_refresh(); self._sync_detection_panel()
            better = info["f1_after"] > info["f1_before"] + 0.01
            msg = (f"训练完成：{info['name']}\n在留出的 {info['n_val']} 个标注块上，准确率（F1，越接近 1 越好）"
                   f"从 {info['f1_before']} 变为 {info['f1_after']}。\n\n")
            if better:
                msg += "新模型更准。要让“" + info["channel"] + "”通道使用新模型吗？（使用后需要重新识别）"
                if QMessageBox.question(self, APP_NAME, msg) == QMessageBox.Yes:
                    p = self.project.detection.channels[c]
                    p.model = info["name"]; p.prob_thresh = round(info["prob_thresh"], 2)
                    self.project.save(); self._sync_detection_panel()
            else:
                self.info(msg + "新模型没有明显变好，建议继续使用原来的模型，或者多标注几块、覆盖更多不同的区域后再训练。")

        self.run_task(lambda task: finetune(project, c, progress=task.report, cancel=task.is_cancelled),
                      done, label="正在训练识别模型…")

    def _man_refresh_info(self):
        if self.image is None or self.project is None:
            self.man_info.setText(""); return
        from ..core.manual import load_manual
        m = load_manual(self.project, self.image)
        self.man_info.setText(f"当前图片已有手动校正：补加 {len(m['added'])} 个，删除 {len(m['removed'])} 个。"
                              if m["added"] or m["removed"] else "当前图片还没有手动校正。")

    def cells_for(self, im: ImageEntry):
        if im.key not in self._cells_cache:
            df = self.project.load_cells(im)
            if df is None:
                return None
            self._cells_cache[im.key] = df
        return self._cells_cache[im.key]

    def _draw_cells(self):
        for nm in [l.name for l in self.viewer.layers if l.name.startswith("细胞-")]:
            self.viewer.layers.remove(nm)
        if self.image is None:
            return
        df = self.cells_for(self.image)
        if df is None or df.empty:
            return
        size = 10 / self.slide.pixel_size_um
        for c, nm in enumerate(self.project.channel_names):
            g = df[df["channel"] == c]
            if g.empty:
                continue
            layer = self.viewer.add_points(g[["y", "x"]].to_numpy(), name=f"细胞-{nm}", size=size,
                                           face_color="transparent", border_color=POINT_COLOR.get(nm, "white"),
                                           border_width=0.15, visible=True,
                                           features={"cid": g.index.to_numpy()})
            layer.feature_defaults = {"cid": -1}

    # ================================================================ ⑤ 共标统计
    def _panel_coloc(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("不同通道识别到的细胞，如果位置重合，就算作同一个细胞（共标）。\n"
                         "判定标准：两个细胞中心距离小于下面的数值，或一个细胞的中心落在另一个细胞的轮廓内。\n"
                         "修改参数后点“重新计算共标”，不需要重新识别细胞。"))
        f = QFormLayout()
        self.coloc_dist = QDoubleSpinBox(); self.coloc_dist.setRange(1, 30); self.coloc_dist.setSuffix(" µm")
        self.coloc_contour = QCheckBox("中心落在轮廓内也算共标")
        self.coloc_nuc_int = QCheckBox("标记物细胞内 DAPI 信号够强，也算有核")
        self.coloc_req_nuc = QCheckBox("只统计有细胞核（DAPI+）的细胞（推荐）")
        self.coloc_req_nuc.setToolTip("切片切到胞体边缘时，会出现没有细胞核的标记物信号碎片，勾选后这些不计入细胞数，"
                                      "但会在 Excel 中单独列出数量。")
        f.addRow("中心距离上限", self.coloc_dist); f.addRow(self.coloc_contour)
        f.addRow(self.coloc_nuc_int); f.addRow(self.coloc_req_nuc)
        L.addLayout(f)
        b = big_button("重新计算共标（所有图片）"); b.clicked.connect(self.recompute_coloc)
        L.addWidget(b)
        gb = QGroupBox("阳性细胞强度分级（弱 / 中 / 强）"); gl = QVBoxLayout(gb)
        lb = QLabel("强度 = 细胞平均亮度 ÷ 该脑片组织背景（倍）。同一批实验请使用同一组分界值。"
                    "设置后 Excel 和组间统计中会多出“弱 / 中 / 强”的细胞数，可以比较强阳性细胞的变化。")
        lb.setWordWrap(True); gl.addWidget(lb)
        r = QHBoxLayout(); r.addWidget(QLabel("通道：")); self.gr_ch = QComboBox(); r.addWidget(self.gr_ch, 1); gl.addLayout(r)
        self.gr_on = QCheckBox("对这个通道分级")
        self.gr_c1 = QDoubleSpinBox(); self.gr_c1.setRange(0.1, 1000); self.gr_c1.setSingleStep(0.5); self.gr_c1.setSuffix(" 倍")
        self.gr_c2 = QDoubleSpinBox(); self.gr_c2.setRange(0.1, 1000); self.gr_c2.setSingleStep(0.5); self.gr_c2.setSuffix(" 倍")
        f2 = QFormLayout(); f2.addRow(self.gr_on); f2.addRow("弱 / 中 分界", self.gr_c1); f2.addRow("中 / 强 分界", self.gr_c2)
        gl.addLayout(f2)
        r = QHBoxLayout()
        b = QPushButton("按全部细胞三等分建议"); b.clicked.connect(self.grading_suggest); r.addWidget(b)
        b = QPushButton("应用并画分布图"); b.clicked.connect(self.grading_apply); r.addWidget(b)
        gl.addLayout(r)
        self.gr_hist = QLabel(); self.gr_hist.setAlignment(Qt.AlignCenter); gl.addWidget(self.gr_hist)
        self.gr_ch.currentIndexChanged.connect(lambda _: self._grading_sync())
        self.coloc_table = QTableWidget(); self.coloc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.coloc_table.setMinimumHeight(220)
        L.addWidget(self.coloc_table, 1)
        L.addWidget(gb)
        return w

    def recompute_coloc(self):
        self.project.coloc.max_dist_um = self.coloc_dist.value()
        self.project.coloc.use_contour = self.coloc_contour.isChecked()
        self.project.coloc.nucleus_by_intensity = self.coloc_nuc_int.isChecked()
        self.project.coloc.require_nucleus = self.coloc_req_nuc.isChecked()
        self.project.save()
        project = self.project

        def job(task):
            from ..core.pipeline import refilter_all
            return refilter_all(project, task.report)

        def done(_):
            self._cells_cache.clear(); self._refresh_coloc_table()

        self.run_task(job, done, label="正在计算共标…")

    def _grading_sync(self):
        nm = self.gr_ch.currentText()
        cuts = (self.project.grading or {}).get(nm) if self.project else None
        self.gr_on.setChecked(bool(cuts))
        self.gr_c1.setValue(cuts[0] if cuts else 3.0); self.gr_c2.setValue(cuts[1] if cuts else 6.0)
        self.gr_hist.clear()

    def grading_suggest(self):
        from ..core.grading import project_folds, suggest_cuts
        nm = self.gr_ch.currentText()
        if not nm:
            return
        v = project_folds(self.project, self.project.channel_names.index(nm))
        if not len(v):
            self.info("还没有这个通道的识别结果。"); return
        a, b = suggest_cuts(v)
        self.gr_c1.setValue(a); self.gr_c2.setValue(b); self.gr_on.setChecked(True)
        self.grading_apply()

    def grading_apply(self):
        from ..core.grading import project_folds, histogram_png
        nm = self.gr_ch.currentText()
        if not nm:
            return
        g = dict(self.project.grading or {})
        if self.gr_on.isChecked():
            a, b = sorted([self.gr_c1.value(), self.gr_c2.value()])
            g[nm] = [round(a, 2), round(b, 2)]
        else:
            g.pop(nm, None)
        self.project.grading = g; self.project.save()
        self._refresh_coloc_table()
        v = project_folds(self.project, self.project.channel_names.index(nm))
        if len(v):
            path = self.project.results_dir / f"强度分布_{nm}.png"
            histogram_png(v, g.get(nm), f"{nm}+ 细胞强度分布（全部图片）", path)
            from qtpy.QtGui import QPixmap
            self.gr_hist.setPixmap(QPixmap(str(path)).scaledToWidth(360, Qt.SmoothTransformation))

    def _refresh_coloc_table(self):
        if not self.project:
            return
        self.coloc_dist.setValue(self.project.coloc.max_dist_um)
        self.coloc_contour.setChecked(self.project.coloc.use_contour)
        self.coloc_nuc_int.setChecked(self.project.coloc.nucleus_by_intensity)
        self.coloc_req_nuc.setChecked(self.project.coloc.require_nucleus)
        nuc = nucleus_channel(self.project.channel_names)
        cur = self.gr_ch.currentText()
        self.gr_ch.blockSignals(True); self.gr_ch.clear()
        self.gr_ch.addItems([n for i, n in enumerate(self.project.channel_names) if i != nuc])
        if cur:
            self.gr_ch.setCurrentText(cur)
        self.gr_ch.blockSignals(False); self._grading_sync()
        from ..core.grading import add_grades
        frames = []
        for im in self.project.images:
            df = self.cells_for(im)
            if df is None or df.empty or "merged_id" not in df.columns:
                continue
            enabled = {s.name for s in im.sections if s.enabled}
            from ..core.pipeline import drop_excluded
            cells = merged_cells(drop_excluded(df[df["section"].isin(enabled)], im), len(self.project.channel_names))
            cells = add_grades(cells, self.project.channel_names, self.project.grading)
            cells.insert(0, "图片", im.key)
            frames.append(summarize(cells, self.project.channel_names, ("图片", "section"), require_nucleus=self.project.coloc.require_nucleus))
        t = pd.concat(frames, ignore_index=True).rename(columns={"section": "脑片"}) if frames else pd.DataFrame()
        self.coloc_table.clear()
        self.coloc_table.setRowCount(len(t)); self.coloc_table.setColumnCount(len(t.columns))
        self.coloc_table.setHorizontalHeaderLabels([str(c) for c in t.columns])
        for i, row in enumerate(t.itertuples(index=False)):
            for j, v in enumerate(row):
                self.coloc_table.setItem(i, j, QTableWidgetItem(str(v)))
        self.coloc_table.resizeColumnsToContents()

    # ================================================================ ⑦ 导出
    def _panel_export(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("把所有图片的统计结果导出成一个 Excel 文件，包括三张表：\n"
                         "• 按脑片汇总：每张脑片各通道阳性细胞数和共标数\n"
                         "• 按脑区×脑片：每张脑片每个脑区（分左右半脑，含上级脑区）的细胞数、面积和密度（需先配准）\n"
                         "• 分析参数：方便以后复现\n"
                         "如果在“② 脑片拆分”中画了自定义区域，会多一张“按自定义区域”表。"))
        ob = QGroupBox("导出选项"); of = QVBoxLayout(ob)
        self.exp_skip_layers = QCheckBox("按脑区表不列出皮层各层（layer 1、2/3…）")
        self.exp_depth = QDoubleSpinBox(); self.exp_depth.setDecimals(0); self.exp_depth.setRange(0, 12)
        self.exp_depth.setSpecialValueText("不限（全部脑区）"); self.exp_depth.setSuffix(" 级")
        self.exp_depth.setToolTip("脑区按层级排列：例如 5 级大约到“Isocortex / 海马 / 丘脑”这一层，"
                                  "7–8 级到具体脑区。细胞数已经包含所有下级脑区，所以少列几级不会漏数。")
        self.exp_masks = QCheckBox("同时保存细胞 mask（1/3 分辨率）"); self.exp_masks.setChecked(True)
        self.exp_qc = QCheckBox("同时生成每张脑片的质控图（PNG，方便快速检查）"); self.exp_qc.setChecked(True)
        self.exp_depth.setPrefix("第 ")
        dr = QHBoxLayout(); dr.addWidget(QLabel("按脑区表最多列到：")); dr.addWidget(self.exp_depth, 1)
        of.addWidget(self.exp_skip_layers); of.addLayout(dr)
        for wdg in (self.exp_masks, self.exp_qc):
            of.addWidget(wdg)
        L.addWidget(ob)
        b = big_button("导出 Excel…"); b.clicked.connect(self.export)
        L.addWidget(b)
        b2 = big_button("打开项目文件夹"); b2.clicked.connect(lambda: self._open_folder(self.project.folder))
        L.addWidget(b2)
        L.addStretch(1)
        return w

    def export(self):
        default = str(self.project.folder / "细胞统计结果.xlsx")
        f, _ = QFileDialog.getSaveFileName(self, "保存 Excel", default, "Excel 文件 (*.xlsx)")
        if not f:
            return
        project = self.project
        project.export_skip_layers = self.exp_skip_layers.isChecked()
        project.export_max_depth = int(self.exp_depth.value())
        project.save()
        masks, qc = self.exp_masks.isChecked(), self.exp_qc.isChecked()

        def done(p):
            extra = []
            if masks:
                extra.append(f"• 细胞 mask：“细胞mask_{Path(p).stem}”文件夹（原图 1/3 分辨率）")
            if qc:
                extra.append(f"• 质控图：“质控图_{Path(p).stem}”文件夹，每张脑片一张，可以快速翻看识别和配准是否正常")
            self.info(f"已导出：\n{p}\n\n同一文件夹中还有：\n" + "\n".join(extra) if extra else f"已导出：\n{p}")

        self.run_task(lambda task: export_excel(project, f, masks=masks, qc=qc, progress=task.report),
                      done, label="正在导出 Excel…")

    def _open_folder(self, path):
        path = str(path)
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
