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
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider, QSizePolicy, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core import detection as det
from ..core import cellpose4
from ..core.coloc import colocalize, merged_cells, summarize, nucleus_channel
from ..core.export import export_excel, companion_dir
from ..core.project import Project, ImageEntry, STANDARD_CHANNELS, DYE_LABELS, UNUSED_LABEL, EXPORT_HEMISPHERES
from ..core.sections import Section, find_sections, contour_in_bbox
from .processing_scope import ProcessingScopeMixin, ScopeDialog
from .layout import fit_window, PanelScroll, ProjectLabel
from .worker import Task
from .registration_panel import RegistrationMixin
from .groups_panel import GroupsMixin
from .quality_panel import QualityMixin
from .relocate_panel import IMAGE_FILTER, RelocateMixin
from .teaching import TeachingMixin
from .figure_panel import FigureMixin
from .accuracy_panel import AccuracyMixin
from .update_panel import UpdateMixin

from .. import APP_NAME  # noqa: E402
COLORMAP = {"DAPI": "blue", "Green": "green", "Red": "red", "Far red": "magenta"}
POINT_COLOR = {"DAPI": "#4f8cff", "Green": "#3ddc4a", "Red": "#ff4040", "Far red": "#ff40ff"}


def rgb_colormap(rgb):
    """把 [R, G, B] 做成 napari 的“黑 → 该颜色”色表（v0.8.3 自定义显示颜色）。"""
    from napari.utils.colormaps import Colormap
    r, g, b = (max(0, min(255, int(v))) / 255 for v in rgb[:3])
    return Colormap([[0, 0, 0, 1], [r, g, b, 1]], name="fosmap-%02x%02x%02x" % tuple(int(v * 255) for v in (r, g, b)))

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


class MainWindow(ProcessingScopeMixin, UpdateMixin, TeachingMixin, FigureMixin, AccuracyMixin, RelocateMixin, QualityMixin, GroupsMixin, RegistrationMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        from ..core.updater import version_text
        self.setWindowTitle(f"{APP_NAME} {version_text()}")
        fit_window(self, 1500, 920)
        self.project: Project | None = None
        self.image: ImageEntry | None = None
        self.slide = None
        self.task: Task | None = None
        self._after_task = None
        self._cells_cache: dict[str, pd.DataFrame] = {}

        # 看图窗口
        self.viewer = ViewerModel(title=APP_NAME)
        self.qt_viewer = QtViewer(self.viewer, show_welcome_screen=False)

        # 左侧步骤
        self.steps = QListWidget()
        f = QFont(); f.setPointSize(11)
        self.steps.setFont(f)
        fm = self.steps.fontMetrics()
        nav_width = max(fm.horizontalAdvance(s+"  ✓") for s in STEPS)+32
        self.steps.setFixedWidth(nav_width)
        for s in STEPS:
            it = QListWidgetItem(s)
            it.setSizeHint(QSize(0, max(36, fm.height()+12)))
            self.steps.addItem(it)

        # 右侧面板
        self.stack = QStackedWidget()
        for panel in (self._panel_project, self._panel_sections, self._panel_display,
                      self._panel_detection, self._panel_coloc, self._panel_registration,
                      self._panel_groups, self._panel_export, self._panel_figure):
            self.stack.addWidget(PanelScroll(panel()))
        self.steps.currentRowChanged.connect(self._on_step)
        self.stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # 顶部：当前图片
        top = QHBoxLayout()
        top.addWidget(QLabel("当前图片："))
        self.image_combo = QComboBox()
        self.image_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.image_combo.setMinimumContentsLength(7)
        self.image_combo.currentTextChanged.connect(self.image_combo.setToolTip)
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        top.addWidget(self.image_combo, 1)
        self.project_label = ProjectLabel()
        self.project_label.set_project("未打开项目")
        self.project_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(self.project_label, 1)
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
        self.tools_scroll = PanelScroll(tools)
        self.tools_scroll.setFixedWidth(max(208, tools.minimumSizeHint().width()+24))
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self.tools_scroll)
        view_row.addWidget(self.qt_viewer, 1)
        self.view_pane = QWidget()
        self.view_pane.setLayout(view_row)
        self.split = QSplitter()
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.steps)
        self.split.addWidget(self.view_pane)
        self.split.addWidget(self.stack)
        self.split.setStretchFactor(1, 1)

        self.step_picker = QComboBox()
        self.step_picker.addItems(STEPS)
        self.step_picker.activated.connect(self.steps.setCurrentRow)
        self.layers_button = QPushButton("图层工具")
        self.layers_button.setCheckable(True)
        self.layers_button.toggled.connect(self._toggle_tools)
        self.pane_button = QPushButton("查看图像")
        self.pane_button.setCheckable(True)
        self.pane_button.toggled.connect(self._toggle_pane)
        self.navigation = QWidget()
        nav = QHBoxLayout(self.navigation)
        nav.setContentsMargins(0, 0, 0, 0)
        for widget in (self.step_picker, self.pane_button, self.layers_button):
            nav.addWidget(widget)
        nav.addStretch(1)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(4, 2, 4, 2)
        cl.addLayout(top)
        cl.addLayout(self._scope_bar())
        cl.addWidget(self.navigation)
        cl.addWidget(self.split, 1)
        self.setCentralWidget(center)
        self._layout_mode = None
        self._build_help_menu()

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
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_layout_mode"):
            self._adapt_layout()

    def _adapt_layout(self):
        width = self.width()
        mode = "single" if width < 900 else "compact" if width < 1280 else "wide"
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        self.navigation.setVisible(mode != "wide")
        self.steps.setVisible(mode == "wide")
        self.pane_button.setVisible(mode == "single")
        self._toggle_pane(self.pane_button.isChecked())
        if mode != "single":
            panel = min(590, max(545, int(width*.36)))
            self.split.setSizes([self.steps.width() if mode == "wide" else 0,
                                 max(1, width-panel-(self.steps.width() if mode == "wide" else 0)), panel])

    def _toggle_tools(self, checked):
        self.tools_scroll.setVisible(self._layout_mode == "wide" or checked)

    def _toggle_pane(self, show_image):
        single = self._layout_mode == "single"
        self.view_pane.setVisible(not single or show_image)
        self.stack.setVisible(not single or not show_image)
        self.pane_button.setText("查看步骤" if show_image else "查看图像")
        self.layers_button.setVisible(not single or show_image)
        self._toggle_tools(self.layers_button.isChecked())

    def _on_step(self, i):
        self.step_picker.setCurrentIndex(i)
        self.stack.setCurrentIndex(i)
        if self._layout_mode == "single":
            self.pane_button.setChecked(False)
        if i != 5 and getattr(self, "_reg_drag_on", False):
            self._reg_set_drag_mode(False)
        self._refresh_step_marks()
        if i == 0:
            self._refresh_project_progress()
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
            self._sync_export_panel()
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
            self.step_picker.model().item(i).setEnabled(bool(has_img))
        self.btn_add_images.setEnabled(has_p)
        self._refresh_scope()

    def error(self, text):
        QMessageBox.critical(self, APP_NAME, text)

    def info(self, text):
        QMessageBox.information(self, APP_NAME, text)

    def run_task(self, fn, on_done, *args, label="正在处理…", lock_export=False):
        if self.task is not None and self.task.isRunning():
            self.info("上一个任务还在运行，请等待完成或点击“取消”。")
            return
        self.task = Task(fn, *args)
        if hasattr(self, 'scope_button'):
            self.scope_button.setEnabled(False)
            self.task.finished.connect(lambda: self.scope_button.setEnabled(bool(self.project and self.project.images)))
        task = self.task
        if lock_export:
            export_task = self.task
            self._export_task = export_task
            self.stack.widget(7).setEnabled(False)

            def unlock_export():
                if self._export_task is export_task:
                    self._export_task = None
                    self.stack.widget(7).setEnabled(True)

            export_task.finished.connect(unlock_export)
        self.device_box.set_busy(True)
        self.progress.setRange(0, 0); self.progress.show(); self.cancel_btn.show()
        self.statusBar().showMessage(label)

        def prog(done, total, text):
            if self.task is not task or self._after_task is not None:
                return
            self.progress.setRange(0, max(total, 1)); self.progress.setValue(done)
            if text:
                self.statusBar().showMessage(text)

        def ok(res):
            if self.task is not task:
                return
            self.device_box.set_busy(False)
            self.progress.hide(); self.cancel_btn.hide()
            if self._after_task is not None:
                return
            self.statusBar().showMessage("完成。", 5000)
            on_done(res)

        def fail(msg):
            if self.task is not task:
                return
            self.device_box.set_busy(False)
            self.progress.hide(); self.cancel_btn.hide()
            if self._after_task is not None:
                return
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
        if hasattr(self, "progress_table"):
            self.task.finished.connect(self._refresh_project_progress, Qt.QueuedConnection)
        self.task.finished.connect(self._finish_transition, Qt.QueuedConnection)
        self.task.start()

    def _when_task_stops(self, action):
        if self.task is not None and self.task.isRunning():
            self._after_task = action
            self.task.cancel()
            self.centralWidget().setEnabled(False)
            self.statusBar().showMessage("正在结束当前任务…")
            return
        action()

    def _finish_transition(self):
        if self._after_task is not None and self.sender() is self.task:
            action, self._after_task = self._after_task, None
            self.centralWidget().setEnabled(True)
            action()

    def _release_project(self):
        self.task = None
        self._after_task = None
        self.centralWidget().setEnabled(True)
        self.device_box.set_busy(False)
        self.progress.hide(); self.cancel_btn.hide()
        cellpose4.close()
        if self.slide is not None:
            self.slide.close()
            self.slide = None
        self.image = None
        self._ft_patch = None
        if getattr(self, "_save_timer", None) and self._save_timer.isActive() and self.project:
            self._save_timer.stop()
            self.project.save()

    def closeEvent(self, event):
        if self.task is not None and self.task.isRunning():
            event.ignore()
            self._when_task_stops(self.close)
            return
        self._release_project()
        super().closeEvent(event)

    def ch_name(self, c):
        return self.project.channel_names[c] if self.project and c < len(self.project.channel_names) else f"通道{c + 1}"

    # ================================================================ ① 项目与图片
    def _panel_project(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("第一步：新建一个项目文件夹（用来保存设置和结果），然后添加要分析的图片。"
                         "一张图片上可以有多张脑片，下一步会自动拆分。\n"
                         "Olympus VSI 的扫描区、ZEISS CZI 的场景会分别列出。旧项目可重新添加原 VSI，补齐其他扫描区。"))
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
        self.images_table = QTableWidget(0, 4)
        self.images_table.setHorizontalHeaderLabels(["图片", "脑片数", "最近操作", "来源文件夹"])
        header = self.images_table.horizontalHeader()
        header.moveSection(3, 1)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.images_table.setColumnWidth(0, 150)
        self.images_table.setColumnWidth(3, 120)
        self.images_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.images_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.images_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.images_table.cellDoubleClicked.connect(lambda r, c: self.image_combo.setCurrentIndex(r))
        L.addWidget(self.images_table, 1)
        rm = QPushButton("从项目中移除选中的图片"); rm.clicked.connect(self.remove_image)
        L.addWidget(rm)
        self._relocate_widgets(L)

        self.source_folder_label = QLabel()
        self.source_folder_label.setWordWrap(True)
        self.source_folder_label.setTextFormat(Qt.PlainText)
        self.source_folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        L.addWidget(self.source_folder_label)
        L.addWidget(QLabel("所选图片的逐脑片处理进度"))
        self.progress_table = QTableWidget()
        self.progress_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.progress_table.setMinimumHeight(220)
        self.progress_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        L.addWidget(self.progress_table, 1)
        self.images_table.currentCellChanged.connect(lambda *_: self._refresh_project_progress())

        L.addWidget(QLabel("各通道对应哪种染色、哪些通道不用，在“② 脑片拆分”里设置。"))

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
        row = QHBoxLayout()
        ab = QPushButton("关于本软件"); ab.clicked.connect(self.show_about)
        up = QPushButton("检查更新"); up.clicked.connect(lambda: self.check_updates(silent=False))
        up.setToolTip("有新版本时下载补丁包（约 1 MB），重启后生效。也可以在“帮助”菜单里从文件安装补丁。")
        row.addWidget(ab); row.addWidget(up); L.addLayout(row)
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
                  + "\n要应用新参数，请到“④ 细胞识别”点“选择处理范围并识别…”。")

    def new_project(self):
        d = QFileDialog.getExistingDirectory(self, "选择一个空文件夹作为项目文件夹")
        if not d:
            return
        if Project.is_project(d):
            if QMessageBox.question(self, APP_NAME, "这个文件夹里已经有项目了，要直接打开吗？") == QMessageBox.Yes:
                self._when_task_stops(lambda: self._open_saved_project(d))
            return
        def create():
            p = Project(folder=Path(d)); p.save()
            self._load_project(p)
            self.add_images()
        self._when_task_stops(create)

    def open_project(self):
        d = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if not d:
            return
        if not Project.is_project(d):
            self.error("这个文件夹里没有找到项目文件（cfos_project.json）。")
            return
        self._when_task_stops(lambda: self._open_saved_project(d))

    def _open_saved_project(self, d):
        try:
            self._load_project(Project.load(d))
        except Exception as e:  # noqa: BLE001
            self.error(f"项目文件无法读取：{e}")

    def _load_project(self, p: Project):
        if self.task is not None and self.task.isRunning():
            self._when_task_stops(lambda: self._load_project(p))
            return
        self._release_project()
        self._processing_scope = None
        self._q_levels = None
        from ..core.log import attach_project, log
        attach_project(p.folder); log.info("打开项目 %s", p.folder)
        self.project = p
        self._sync_export_panel()
        self._cells_cache.clear()
        self._reg_cache.clear()
        self.project_label.set_project(f"项目：{p.folder.name}")
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
        bad, notes, imported = [], [], []
        for f in files:
            try:
                previous_count = len(self.project.images)
                entries, messages = self.project.import_images(f)
                added = len(self.project.images) - previous_count
                if len(entries) > 1 and added:
                    imported.append(f"{Path(f).name} 已加入项目的扫描区共 {len(entries)} 个，本次新增 {added} 个")
                notes.extend(f"{Path(f).name}：{msg}" for msg in messages)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{Path(f).name}：{e}")
        self.project.save()
        self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
        if imported:
            self.statusBar().showMessage("；".join(imported) + "。可在“当前图片”中切换扫描区。", 20000)
        if bad:
            self.error("以下图片无法加入：\n" + "\n".join(bad))
        if notes:
            self.info("请留意：\n" + "\n".join(notes))
        if self.project.images and self.image is None:
            self.image_combo.setCurrentIndex(0)

    def add_images_programmatic(self, files):
        """测试和命令行使用。"""
        for f in files:
            self.project.import_images(f)
        self.project.save()
        self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
        self.image_combo.setCurrentIndex(0)

    def batch_dialog(self):
        from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLineEdit
        from ..core.pipeline import list_images
        dlg = QDialog(self); dlg.setWindowTitle("一键批量处理")
        L = QVBoxLayout(dlg)
        L.addWidget(hint("可直接使用项目图片，也可先选文件夹导入。下一步勾选要处理的图片、扫描场景或脑片，"
                         "全部使用当前项目里的识别参数（建议先用一张图片调好参数）。\n"
                         "完成后会在项目文件夹里生成 Excel，并列出需要人工检查配准的脑片。"))
        row = QHBoxLayout(); path = QLineEdit(); pick = QPushButton("选择文件夹…")
        row.addWidget(path, 1); row.addWidget(pick); L.addLayout(row)
        cnt = QLabel(""); L.addWidget(cnt)

        def choose():
            d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if d:
                path.setText(d)
                cnt.setText(f"找到 {len(list_images(d))} 个图片文件。VSI 扫描区和 CZI 场景会分别处理。")
        pick.clicked.connect(choose)
        c_reg = QCheckBox("配准到脑图谱"); c_reg.setChecked(True)
        c_nl = QCheckBox("配准后做非线性贴合"); c_nl.setChecked(True)
        c_redo = QCheckBox("已经处理过的图片也重新处理"); c_redo.setChecked(False)
        for c in (c_reg, c_nl, c_redo):
            L.addWidget(c)
        bx = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bx.button(QDialogButtonBox.Ok).setText("选择处理范围…"); bx.button(QDialogButtonBox.Cancel).setText("取消")
        bx.accepted.connect(dlg.accept); bx.rejected.connect(dlg.reject); L.addWidget(bx)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            folder = path.text().strip()
            candidates = list(self.project.images) if self.project else []
            if folder:
                files = list_images(folder)
                if not files:
                    self.info("文件夹中没有可导入的图片。")
                    return
                if self.project is None:
                    pf = Path(folder) / "cFos分析项目"
                    self._load_project(Project.load(pf) if Project.is_project(pf) else Project(folder=pf))
                candidates, notes = [], []
                for file in files:
                    try:
                        entries, warnings = self.project.import_images(file)
                        candidates.extend(entries)
                        notes.extend(warnings)
                    except Exception as exc:
                        notes.append(f"{Path(file).name}：{exc}")
                self.project.save()
                self._refresh_images(); self._refresh_enabled()
                if notes:
                    self.info("导入提示\n" + "\n".join(notes))
            if not candidates:
                self.info("请先添加图片或选择文件夹。")
                return
            dlg = ScopeDialog(candidates, self._processing_scope or {}, self)
            if dlg.exec() != QDialog.Accepted:
                return
            selection = dlg.selection()
            self._apply_scope(selection)
            self.run_batch(folder, c_reg.isChecked(), c_nl.isChecked(), c_redo.isChecked(), selection)

    def run_batch(self, folder, do_reg=True, do_nl=True, redo=False, selection=None):
        import datetime as _dt
        from ..core.pipeline import list_images, run_batch
        from ..core.scope import scoped_project, selected_images
        files = (list_images(folder) if selection is None else
                 list(dict.fromkeys(im.path for im in selected_images(self.project, selection))))
        if not files:
            self.error("没有可处理的图片。")
            return
        if self.project is None:
            pf = Path(folder) / "cFos分析项目"
            self._load_project(Project.load(pf) if Project.is_project(pf) else Project(folder=pf))
            self.project.save()
        project = self.project
        xlsx = project.folder / f"批量结果_{_dt.datetime.now():%Y%m%d_%H%M%S_%f}.xlsx"

        def job(task):
            kwargs = {"selection": selection} if selection is not None else {}
            chk = run_batch(project, files, do_reg, do_nl, redo, report=task.report, cancel=task.is_cancelled, **kwargs)
            task.report(1, 1, "正在导出 Excel…")
            export_excel(scoped_project(project, selection), xlsx)
            return chk

        def done(chk):
            self._cells_cache.clear(); self._refresh_images(); self._refresh_channels_form(); self._refresh_enabled()
            self._q_levels = None; self._refresh_section_table()
            imports = [c for c in chk if c.get("类型") == "导入"]
            results = [c for c in chk if c.get("类型") != "导入"]
            bad = [c for c in results if c.get("质量") == "可疑" or c["需要检查"].startswith("出错") or "未配准" in c["需要检查"]]
            warn = [c for c in results if (c.get("质量") == "注意" or c["需要检查"]) and c not in bad]
            sections = [c for c in results if c.get("脑片")]
            unchecked = sum(not c.get("质量") for c in sections)
            msg = f"批量处理完成：{len(files)} 个图片文件，{len(sections)} 张脑片有处理记录。\n结果已保存到：\n{xlsx}\n\n"
            import_lines = [f"• {c['图片']}：{c['需要检查']}" for c in imports]
            if imports:
                msg += f"导入提示（{len(imports)} 条）：\n" + "\n".join(import_lines[:8]) + "\n\n"
                if len(imports) > 8:
                    msg += "展开详细信息可查看全部导入提示。\n\n"
            if bad:
                msg += "以下图片处理出错、脑片未配准或质量可疑，请优先检查：\n" + "\n".join(
                    f"• {c['图片']} {c['脑片']}：{c['需要检查']}" for c in bad[:30]) + "\n\n"
            if warn:
                msg += f"另有 {len(warn)} 张脑片需要留意，可到“② 脑片拆分”查看质量，或到“⑥ 脑图谱配准”检查位置。\n"
            if unchecked:
                msg += f"{unchecked} 张脑片没有自动质量评分，可在“② 脑片拆分”重新检查。\n"
            if sections and not (imports or bad or warn or unchecked):
                msg += "自动质量检查没有发现问题（配准、模糊、过曝、组织缺损），仍建议抽查几张。"
            if len(imports) > 8:
                dialog = QMessageBox(QMessageBox.Information, APP_NAME, msg, QMessageBox.Ok, self)
                dialog.setDetailedText("\n".join(import_lines))
                dialog.exec_()
            else:
                self.info(msg)

        self.run_task(job, done, label="正在批量处理…", lock_export=True)

    def remove_image(self):
        r = self.images_table.currentRow()
        if r < 0 or self.project is None:
            return
        im = self.project.images[r]
        if QMessageBox.question(self, APP_NAME, f"从项目中移除 {im.display_name}？（不会删除原图）") != QMessageBox.Yes:
            return
        self.project.images.pop(r); self.project.save()
        self._refresh_images(); self._refresh_enabled()

    def _refresh_images(self):
        p = self.project
        selected = self.images_table.currentRow()
        self.images_table.blockSignals(True)
        self.images_table.setRowCount(len(p.images))
        cur = self.image_combo.currentIndex()
        self.image_combo.blockSignals(True); self.image_combo.clear()
        from ..core.workflow_status import source_folders
        folders = source_folders(p.images)
        for i, im in enumerate(p.images):
            item = QTableWidgetItem(im.display_name)
            item.setToolTip(f"{im.display_name}\n{im.path}")
            self.images_table.setItem(i, 0, item)
            self.images_table.setItem(i, 1, QTableWidgetItem(str(len(im.sections)) if im.sections else "—"))
            st = QTableWidgetItem(im.status)
            if not Path(im.path).exists():
                st.setText("找不到原图"); st.setForeground(Qt.red)
                st.setToolTip(f"原路径：{im.path}\n点下方“重新定位找不到的图片…”找回。")
            elif im.error:
                st.setToolTip(im.error); st.setText(f"出错：{im.error[:40]}")
                st.setForeground(Qt.red)
            self.images_table.setItem(i, 2, st)
            short, full = folders[i]
            folder = QTableWidgetItem(short)
            folder.setToolTip(full)
            self.images_table.setItem(i, 3, folder)
            self.image_combo.addItem(f"{i + 1}. {im.display_name}")
            self.image_combo.setItemData(i, item.toolTip(), Qt.ToolTipRole)
        self._refresh_scope()
        if p.images:
            idx = cur if 0 <= cur < len(p.images) else 0
            if self._processing_scope and p.images[idx].key not in self._processing_scope:
                idx = next(i for i, im in enumerate(p.images) if im.key in self._processing_scope)
            current = next((i for i, im in enumerate(p.images) if im is self.image), None)
            self.image_combo.setCurrentIndex(idx if current is None else current)
            self.image_combo.blockSignals(False)
            if current is None:
                self._on_image_changed(idx)
        else:
            self.image_combo.blockSignals(False)
            self.image = None
            self._clear_image_view()
            self._refresh_image_panel()

        if p.images:
            self.images_table.setCurrentCell(selected if 0 <= selected < len(p.images) else self.image_combo.currentIndex(), 0)
        self.images_table.blockSignals(False)
        self._refresh_project_progress()

    def _refresh_project_progress(self):
        if self.stack.currentIndex() != 0:
            return
        table = self.progress_table
        row = self.images_table.currentRow()
        if not self.project or not (0 <= row < len(self.project.images)):
            self.source_folder_label.clear()
            table.setColumnCount(0)
            table.setRowCount(0)
            return
        im = self.project.images[row]
        folder = self.images_table.item(row, 3)
        if folder is not None:
            self.source_folder_label.setText(f"来源文件夹  {folder.toolTip()}")
        from ..core.workflow_status import section_progress
        progress = section_progress(self.project, im)
        if not progress:
            progress = [("尚无脑片", ("已导入", "待拆分") + ("—",) * (len(STEPS) - 2))]
        table.setRowCount(len(STEPS))
        table.setColumnCount(len(progress))
        table.setVerticalHeaderLabels(STEPS)
        table.setHorizontalHeaderLabels([name for name, _ in progress])
        for column, (name, states) in enumerate(progress):
            for step, state in enumerate(states):
                item = QTableWidgetItem(state)
                item.setToolTip(f"{im.display_name} / {name}\n{STEPS[step]}  {state}")
                table.setItem(step, column, item)

    def _clear_image_view(self):
        self.viewer.layers.clear()
        self.image_layers = []
        self._ft_patch = None
        self._blind_context = None
        self._drag = self._reg_lab = self._reg_bnd = self._atlas_curves = None
        self._reg_paths = []
        self._reg_timer.stop()
        if self._reg_drag_on:
            self._reg_set_drag_mode(False)
        self.det_summary.clear()
        self.z_info.clear()
        if self.slide is not None:
            self.slide.close()
            self.slide = None

    def _refresh_image_panel(self):
        step = self.stack.currentIndex()
        if step == 1:
            self._refresh_section_table()
            self._refresh_roi_list()
        elif step == 2:
            self._sync_display_panel()
        elif step == 3:
            self._sync_detection_panel()
        elif step == 5:
            self._sync_registration_panel()
        elif step == 8:
            self._refresh_fig_list()

    def _refresh_channels_form(self):
        while self.channels_form.rowCount():
            self.channels_form.removeRow(0)
        if not self.project:
            return
        self.z_combo.blockSignals(True)
        self.z_combo.setCurrentIndex(max(0, self.z_combo.findData(self.project.z_mode)))
        self.z_combo.blockSignals(False)
        p = self.project
        file_names = [ch.name for ch in self.slide.channels] if self.slide is not None else []
        tissue = p.tissue_channels()
        for c, nm in enumerate(p.channel_names):
            cb = QComboBox()
            labels = list(DYE_LABELS.values())
            unused = c in p.unused_channels
            cur = UNUSED_LABEL if unused else DYE_LABELS.get(nm, nm)
            if cur not in labels and cur != UNUSED_LABEL:
                labels.append(cur)      # 以前自己填的名字（如“NeuN”）也保留在列表里
            cb.addItems(labels + [UNUSED_LABEL])
            cb.setCurrentText(cur)
            cb.currentTextChanged.connect(lambda t, c=c, cb=cb: self._rename_channel(c, t, cb))
            use = QCheckBox("拆分"); use.setEnabled(not unused)
            use.setChecked(not unused and (tissue is None or c in tissue))
            use.setToolTip("用这个通道找组织轮廓、拆分脑片。改了以后请重新点“自动识别脑片”。")
            use.toggled.connect(lambda on, c=c: self._set_tissue_channel(c, on))
            row = QHBoxLayout(); row.addWidget(cb, 1); row.addWidget(use)
            fn = f"（{file_names[c]}）" if c < len(file_names) else ""
            cb.setToolTip(f"图片文件里第 {c + 1} 个通道" + (f"，扫描仪写的名字是“{file_names[c]}”" if fn else "") + "。")
            self.channels_form.addRow(f"通道 {c + 1}{fn}", row)

    def _set_tissue_channel(self, c, on):
        p = self.project
        cur = p.tissue_channels() or p.used_channels()
        chs = sorted(set(cur) | {c}) if on else [x for x in cur if x != c]
        if not chs:
            self.error("至少要用一个通道来拆分脑片。")
            self._refresh_channels_form(); return
        p.section_channels = [] if chs == p.used_channels() else chs
        p.save()
        self.statusBar().showMessage("拆分用的通道已改。请重新点“自动识别脑片”，按新设置找组织轮廓。", 10000)

    def _rename_channel(self, c, label, cb=None):
        p = self.project
        was_unused = c in p.unused_channels
        old_name = p.channel_names[c]
        why = p.set_channel_role(c, label)
        if why:
            self.error(why)
            QTimer.singleShot(0, self._refresh_channels_form)
            return
        detected = any(p.has_cells(im) for im in p.images)
        if detected:
            notes = []
            if label == UNUSED_LABEL or was_unused:
                notes.append("启用 / 停用通道后，请在“④ 细胞识别”里重新识别（没变的通道会直接用上次的结果）。")
            if "DAPI" in (old_name, p.channel_names[c]):
                notes.append("细胞核（DAPI）通道变了，请在“⑤ 共标统计”中重新计算所选脑片的共标。")
            if notes:
                self.info("通道设置已保存。\n\n" + "\n".join(notes))
        QTimer.singleShot(0, self._refresh_channels_form)
        self._refresh_detection_panel_safe()
        if self.slide is not None:
            self._show_image()

    def _refresh_detection_panel_safe(self):
        try:
            self._sync_detection_panel()
        except Exception:  # noqa: BLE001  面板还没建好时忽略
            pass

    # ================================================================ 看图
    def _on_image_changed(self, idx):
        if self.project is None or not (0 <= idx < len(self.project.images)):
            return
        target = self.project.images[idx]
        if self._processing_scope is not None and target.key not in self._processing_scope:
            if self.image in self.project.images:
                self.image_combo.blockSignals(True)
                self.image_combo.setCurrentIndex(self.project.images.index(self.image))
                self.image_combo.blockSignals(False)
            return
        self._clear_image_view()
        self.image = target
        if not self.images_table.signalsBlocked():
            self.images_table.setCurrentCell(idx, 0)
        if self._processing_scope is not None:
            focused = self.focus_combo.currentData()
            if not focused or focused[0] != target.key:
                for j in range(self.focus_combo.count()):
                    if self.focus_combo.itemData(j)[0] == target.key:
                        self.focus_combo.setCurrentIndex(j)
                        break
        if not Path(self.image.path).exists():
            self.statusBar().showMessage(f"找不到原图：{self.image.path}。请在“① 项目与图片”中点“重新定位找不到的图片…”。")
            self._refresh_channels_form(); self._refresh_image_panel()
            return
        try:
            self.slide = self.image.open_slide(z_mode=self.project.z_mode)
        except Exception as e:  # noqa: BLE001
            self._refresh_channels_form(); self._refresh_image_panel()
            self.error(f"{self.image.display_name} 无法打开：{e}")
            return
        self.z_info.setText(f"当前图片是 z-stack（{self.slide.n_z} 层），正在使用“{self.z_combo.currentText()}”。"
                            if self.slide.n_z > 1 else "当前图片是单层图像（或扫描仪已做景深合成），不需要设置。")
        self._refresh_channels_form()
        self._show_image()
        self._refresh_image_panel()

    # ---------------- 通道显示颜色（v0.8.3）
    def _ch_index(self, nm) -> int:
        names = self.project.channel_names if self.project else []
        return names.index(nm) if nm in names else -1

    def ch_cmap(self, c):
        """看图时通道 c 的色表：项目里选的颜色优先，否则按通道名（DAPI 蓝、Green 绿……）。"""
        rgb = self._display_color(c) if self.project and c >= 0 else None
        if rgb is not None:
            return rgb_colormap(rgb)
        nm = self.project.channel_names[c] if self.project and 0 <= c < len(self.project.channel_names) else ""
        return COLORMAP.get(nm, "gray")

    def ch_point_color(self, c):
        rgb = self._display_color(c) if self.project and c >= 0 else None
        if rgb is not None:      # 细胞圈用浅一点的同色，压在同色荧光上也看得清
            return "#%02x%02x%02x" % tuple(min(255, int(v * 0.5 + 128)) for v in rgb)
        nm = self.project.channel_names[c] if self.project and 0 <= c < len(self.project.channel_names) else ""
        return POINT_COLOR.get(nm, "white")

    def _show_image(self):
        self.viewer.layers.clear()
        sl = self.slide
        levels = [sl._arr(lv) for lv in range(sl.n_levels)]
        names = [self.ch_name(c) for c in range(len(sl.channels))]
        cmaps = [self.ch_cmap(c) for c in range(len(names))]
        if levels[0].ndim == 2:  # 单通道图像
            layers = [self.viewer.add_image(levels, multiscale=True, name=names[0], colormap=cmaps[0],
                                            blending="additive")]
        else:
            layers = self.viewer.add_image(levels, channel_axis=0, multiscale=True, name=names,
                                           colormap=cmaps, blending="additive")
        self.image_layers = layers
        self._apply_focused_display()
        self._draw_sections()
        self._draw_polygons()
        self._draw_cells()
        self._draw_figure_rois(force=self.stack.currentIndex() == 8)
        self.viewer.reset_view()

    def _display_color(self, c):
        return self.project.custom_color(c, self.image, self._focus_section())

    def _apply_focused_display(self):
        section = self._focus_section()
        for c, layer in enumerate(self.image_layers):
            d = self.project.channel_display(c, self.image, section)
            if d.high > d.low:
                layer.contrast_limits = (d.low, d.high)
            layer.gamma = d.gamma
            layer.visible = d.visible and c not in self.project.unused_channels
            self._refresh_channel_color(c)

    def _auto_contrast(self, c):
        sl = self.slide
        lv = sl.level_for_pixel_size(8.0)
        section = self._focus_section()
        a = sl.read(lv, section.bbox0 if section else None, [c])[0]
        v = a[a > 0] if (a > 0).any() else a.ravel()
        lo, hi = np.percentile(v, [0.5, 99.7])
        hi = max(hi, lo + 1)
        self.project.update_display(self._processing_scope, c, low=float(lo), high=float(hi), gamma=1.0)
        self._apply_focused_display()
        self._save_later()

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
        self.channels_box = QGroupBox("通道设置（对应实验中的染色）")
        cbl = QVBoxLayout(self.channels_box)
        cbl.addWidget(hint("每个通道选一种染色；没用的通道选“不使用”（不识别、不显示、不导出）。\n"
                           "勾选“拆分”的通道用来找组织轮廓、拆分脑片；某个通道背景太亮、把载玻片也当成组织时，取消勾选它。"))
        self.channels_form = QFormLayout(); cbl.addLayout(self.channels_form)
        L.addWidget(self.channels_box)
        b1 = big_button("拆分所选图片 / 更新所选脑片轮廓"); b1.clicked.connect(lambda: self._require_scope() and self.auto_sections(True))
        b3 = big_button("保存方框修改"); b3.clicked.connect(lambda: self._require_scope() and self.save_section_edits())
        for b in (b1, b3):
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
        bx3 = QPushButton("保存区域"); bx3.clicked.connect(lambda: self._require_scope() and self.save_polygons()); el.addWidget(bx3)
        el.addWidget(QLabel("自定义区域名称（双击可改名）："))
        self.roi_list = QListWidget(); self.roi_list.setMaximumHeight(90)
        self.roi_list.itemChanged.connect(self._rename_roi)
        el.addWidget(self.roi_list)
        L.addWidget(ex)
        return w

    def _polygon_in_scope(self, polygon):
        if (self._processing_scope or {}).get(self.image.key) is None:
            return True
        for sec in self.image.sections:
            if self._section_in_scope(self.image, sec):
                x, y, w, h = sec.bbox0
                if all(x <= px <= x + w and y <= py <= y + h for px, py in polygon):
                    return True
        return False

    def _draw_polygons(self):
        for nm in ("排除区域", "自定义区域"):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        if self.image is None or self.slide is None:
            return
        lw = max(self.slide.level_shapes[0]) / 400
        ex = [np.asarray(p, float)[:, ::-1] for p in self.image.exclusions if len(p) >= 3 and self._polygon_in_scope(p)]
        self.viewer.add_shapes(ex, shape_type="polygon", name="排除区域", edge_color="#ffdc00",
                               face_color=[1, 0.86, 0, 0.25], edge_width=lw)
        rois = [r for r in self.image.rois if len(r["polygon"]) >= 3 and self._polygon_in_scope(r["polygon"])]
        self.viewer.add_shapes([np.asarray(r["polygon"], float)[:, ::-1] for r in rois], shape_type="polygon",
                               name="自定义区域", edge_color="cyan", face_color="transparent", edge_width=lw,
                               features={"name": [r["name"] for r in rois]},
                               text={"string": "{name}", "color": "cyan", "size": 14})
        self._refresh_roi_list()

    def _refresh_roi_list(self):
        self.roi_list.blockSignals(True); self.roi_list.clear()
        for i, r in enumerate(self.image.rois if self.image else []):
            if not self._polygon_in_scope(r["polygon"]):
                continue
            it = QListWidgetItem(r["name"]); it.setData(Qt.UserRole, i); it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.roi_list.addItem(it)
        self.roi_list.blockSignals(False)

    def _rename_roi(self, item):
        i = item.data(Qt.UserRole)
        if self.image and 0 <= i < len(self.image.rois) and item.text().strip():
            self.image.rois[i]["name"] = item.text().strip()
            self.project.save(); self._draw_polygons()

    def _edit_polygons(self, name):
        if self.slide is None:
            return
        if name not in self.viewer.layers:
            self._draw_polygons()
        layer = self.viewer.layers[name]
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon"
        self.statusBar().showMessage(f"正在画“{name}”：在图上逐个点击顶点，双击结束。画完点“保存区域”。", 15000)

    def save_polygons(self):
        if self.slide is None:
            return
        def polys(nm):
            if nm not in self.viewer.layers:
                return []
            return [[(float(p[-1]), float(p[-2])) for p in d] for d in self.viewer.layers[nm].data if len(d) >= 3]
        ex, custom = polys("排除区域"), polys("自定义区域")
        if any(not self._polygon_in_scope(poly) for poly in ex + custom):
            self.info("请将整个区域画在所选脑片内，或在处理范围中勾选整张图片。")
            return
        self.image.exclusions = [p for p in self.image.exclusions if not self._polygon_in_scope(p)] + ex
        layer = self.viewer.layers["自定义区域"] if "自定义区域" in self.viewer.layers else None
        old = dict(enumerate(layer.features["name"])) if layer is not None else {}
        used = {r["name"] for r in self.image.rois}
        rois = [r for r in self.image.rois if not self._polygon_in_scope(r["polygon"])]
        keep = {r["name"] for r in rois}
        for i, poly in enumerate(custom):
            nm = old.get(i)
            if not nm or nm in keep:
                k = 1
                while f"区域{k}" in used:
                    k += 1
                nm = f"区域{k}"; used.add(nm)
            keep.add(nm)
            rois.append({"name": nm, "polygon": poly})
        self.image.rois = rois
        self.project.save(); self._draw_polygons()
        self.statusBar().showMessage(f"已保存：{len(self.image.exclusions)} 个排除区域，{len(rois)} 个自定义区域。"
                                     "导出 Excel 时生效，不需要重新识别。", 10000)

    def auto_sections(self, all_images: bool):
        selection = self._processing_scope
        targets = self._target_images(all_images)
        project = self.project
        channels = project.tissue_channels()

        def job(task):
            out = {}
            for i, im in enumerate(targets):
                if task.is_cancelled():
                    raise InterruptedError
                task.report(i, len(targets), f"正在识别脑片：{im.display_name}")
                sl = im.open_slide(z_mode=project.z_mode)
                try:
                    names = selection.get(im.key) if selection is not None else None
                    if names is None:
                        out[im.key] = find_sections(sl, channels=channels)
                    else:
                        out[im.key] = [(s, contour_in_bbox(sl, s.bbox0, channels)) for s in im.sections if s.name in names]
                finally:
                    sl.close()
            return out

        def done(res):
            for im in targets:
                if selection is not None and selection.get(im.key) is not None:
                    for section, contour in res[im.key]:
                        section.contour0 = contour
                else:
                    im.sections = res[im.key]
                im.status = f"已拆分 {len(im.sections)} 张脑片"
            self.project.save(); self._refresh_images()
            self._draw_sections(); self._refresh_section_table()
            n = sum(len(v) for v in res.values())
            self.statusBar().showMessage(f"共识别出 {n} 张脑片。", 8000)

        self.run_task(job, done, label="正在识别脑片…")

    def _draw_sections(self):
        if "脑片" in self.viewer.layers:
            self.viewer.layers.remove("脑片")
        if self.image is None or self.slide is None:
            return
        rects, names, identities = [], [], []
        for s in self.image.sections:
            if not self._section_in_scope(self.image, s):
                continue
            identities.append(s.name)
            x, y, w, h = s.bbox0
            rects.append(np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], float))
            names.append(s.name + ("" if s.enabled else "（不分析）"))
        lw = max(self.slide.level_shapes[0]) / 300
        layer = self.viewer.add_shapes(rects, shape_type="rectangle", name="脑片", edge_color="yellow",
                               face_color="transparent", edge_width=lw,
                               features={"name": names, "section_key": identities},
                               text={"string": "{name}", "color": "yellow", "size": 14, "anchor": "upper_left"})
        layer.feature_defaults["section_key"] = ""

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
            self.sec_table.setRowHidden(i, not self._section_in_scope(self.image, s))
        self.sec_table.blockSignals(False)

    def _section_item_changed(self, item):
        r, c = item.row(), item.column()
        s = self.image.sections[r]
        if c == 1:
            s.enabled = item.checkState() == Qt.Checked
        elif c == 0 and item.text().strip():
            new_name = item.text().strip()
            if any(other is not s and other.name == new_name for other in self.image.sections):
                self._refresh_section_table()
                self.info("同一图片内的脑片名称须唯一。")
                return
            names = (self._processing_scope or {}).get(self.image.key)
            if names is not None:
                self._processing_scope[self.image.key] = tuple(new_name if n == s.name else n for n in names)
            s.name = new_name
        self.project.save(); self._draw_sections(); self._refresh_scope()

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
        old = {s.name: s for s in self.image.sections}
        edited = {}
        added = []
        partial = (self._processing_scope or {}).get(self.image.key) is not None
        identities = layer.features["section_key"]
        used = set(old)
        for i, poly in enumerate(layer.data):
            ys, xs = poly[:, -2], poly[:, -1]
            x0, y0 = max(0, xs.min()), max(0, ys.min())
            bbox = (int(x0), int(y0), int(xs.max() - x0), int(ys.max() - y0))
            if bbox[2] < 50 or bbox[3] < 50:
                continue
            prev = old.get(identities.iloc[i])
            if partial and (prev is None or not self._section_in_scope(self.image, prev)):
                self.info("新增脑片框需要在处理范围中勾选整张图片。")
                return
            if prev is not None and tuple(prev.bbox0) == bbox:
                s = prev
            else:
                s = Section(id=i + 1, bbox0=bbox, contour0=contour_in_bbox(self.slide, bbox, self.project.tissue_channels()),
                            name=prev.name if prev else "", enabled=prev.enabled if prev else True)
            if prev:
                edited[prev.name] = s
            else:
                added.append(s)
        new = [edited.get(sec.name, sec) for sec in self.image.sections
               if sec.name in edited or not self._section_in_scope(self.image, sec)] + added
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
        self.display_layout.addWidget(hint("设置应用到顶部所选的图片和脑片，以当前查看对象预览。\n"
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
        if self._processing_scope is None and self.project and self.project.images:
            choose = QPushButton("选择要调整的图片或脑片…")
            choose.clicked.connect(self.choose_processing_scope)
            self.display_rows.addWidget(choose)
            return
        if not self.slide:
            return
        vmax = 255 if self.slide._arr(0).dtype == np.uint8 else 65535
        for c in range(len(self.slide.channels)):
            d = self.project.channel_display(c, self.image, self._focus_section())
            box = QGroupBox(self.ch_name(c)); f = QFormLayout(box)
            vis = QCheckBox("显示"); vis.setChecked(d.visible)
            lo = QSlider(Qt.Horizontal); lo.setRange(0, vmax); lo.setValue(int(d.low))
            hi = QSlider(Qt.Horizontal); hi.setRange(0, vmax); hi.setValue(int(d.high))
            gm = QDoubleSpinBox(); gm.setRange(0.2, 3.0); gm.setSingleStep(0.1); gm.setValue(d.gamma)
            auto = QPushButton("自动")
            auto.setToolTip("按当前查看脑片（或整图）估计亮度范围，将同一范围应用到所选对象。")
            lo_lab, hi_lab = QLabel(str(int(d.low))), QLabel(str(int(d.high)))
            r1 = QHBoxLayout(); r1.addWidget(lo, 1); r1.addWidget(lo_lab)
            r2 = QHBoxLayout(); r2.addWidget(hi, 1); r2.addWidget(hi_lab)
            f.addRow(vis, auto); f.addRow("最暗", r1); f.addRow("最亮", r2); f.addRow("伽马", gm)
            f.addRow("显示颜色", self._color_row(c))

            def apply(field, value, c=c, lo=lo, hi=hi, lo_lab=lo_lab, hi_lab=hi_lab):
                changes = {field: value}
                if field in ("low", "high"):
                    # The two sliders describe one interval on all selected targets.
                    changes = {"low": float(lo.value()), "high": float(max(hi.value(), lo.value() + 1))}
                    lo_lab.setText(str(int(changes["low"]))); hi_lab.setText(str(int(changes["high"])))
                self.project.update_display(self._processing_scope, c, **changes)
                d = self.project.channel_display(c, self.image, self._focus_section())
                layer = self.image_layers[c]
                layer.visible = d.visible and c not in self.project.unused_channels
                layer.contrast_limits = (d.low, d.high); layer.gamma = d.gamma
                self._save_later()

            def do_auto(_=None, c=c):
                self._auto_contrast(c)
                self._sync_display_panel()

            vis.toggled.connect(lambda v, a=apply: a("visible", v))
            lo.valueChanged.connect(lambda v, a=apply: a("low", v))
            hi.valueChanged.connect(lambda v, a=apply: a("high", v))
            gm.valueChanged.connect(lambda v, a=apply: a("gamma", v))
            auto.clicked.connect(do_auto)
            self.display_rows.addWidget(box)

    COLOR_DEFAULT = "默认（按染色）"
    COLOR_CUSTOM = "自定义…"

    def _color_row(self, c):
        """“③ 亮度 / 对比度”里每个通道的显示颜色（v0.8.3）：常用颜色 + 调色板。
        看图、质控图、截图制图、配准图导出都用这个颜色；不影响细胞识别。"""
        from qtpy.QtWidgets import QColorDialog
        from qtpy.QtGui import QColor
        from ..core.project import COLOR_PRESETS
        wd = QWidget(); row = QHBoxLayout(wd); row.setContentsMargins(0, 0, 0, 0)
        sw = QLabel(); sw.setFixedSize(22, 22)
        cb = QComboBox(); cb.addItems([self.COLOR_DEFAULT] + list(COLOR_PRESETS) + [self.COLOR_CUSTOM])
        cb.setToolTip("这个通道在看图窗口、质控图、截图制图和配准图里用什么颜色显示。\n"
                      "新项目默认沿用显微镜软件里的颜色（例如 Far Red 显示为黄色）；只改显示，不影响识别和统计。")
        def current_rgb():
            rgb = self._display_color(c)
            if rgb is None:
                cm = COLORMAP.get(self.project.channel_names[c] if c < len(self.project.channel_names) else "", "")
                rgb = {"blue": (0, 0, 255), "green": (0, 255, 0), "red": (255, 0, 0), "magenta": (255, 0, 255)}.get(cm, (200, 200, 200))
            return rgb
        def show():
            r, g, b = current_rgb()
            sw.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid #888; border-radius: 3px;")
            rgb = self._display_color(c)
            name = next((k for k, v in COLOR_PRESETS.items() if rgb is not None and tuple(v) == tuple(rgb)), None)
            cb.blockSignals(True)
            cb.setCurrentText(self.COLOR_DEFAULT if rgb is None else (name or self.COLOR_CUSTOM))
            cb.blockSignals(False)
        def pick(text):
            if text == self.COLOR_CUSTOM:
                r, g, b = current_rgb()
                col = QColorDialog.getColor(QColor(r, g, b), self, f"选择“{self.ch_name(c)}”的显示颜色")
                if not col.isValid():
                    show(); return
                rgb = (col.red(), col.green(), col.blue())
            elif text == self.COLOR_DEFAULT:
                rgb = None
            else:
                rgb = COLOR_PRESETS[text]
            self.set_channel_color(c, rgb)
            show()
        cb.currentTextChanged.connect(pick)
        row.addWidget(sw); row.addWidget(cb, 1)
        show()
        return wd

    def set_channel_color(self, c, rgb):
        self.project.update_display(self._processing_scope, c, color=list(rgb) if rgb is not None else None)
        self._refresh_channel_color(c)
        self.project.save()

    def _refresh_channel_color(self, c):
        if c < len(getattr(self, "image_layers", [])):
            try:
                self.image_layers[c].colormap = self.ch_cmap(c)
            except Exception:  # noqa: BLE001
                pass
        for layer in list(self.viewer.layers):
            nm = self.project.channel_names[c] if c < len(self.project.channel_names) else None
            if nm and layer.name in (f"细胞-{nm}", f"预览-{nm}"):
                try:
                    if hasattr(layer, "border_color"):
                        layer.border_color = self.ch_point_color(c)
                    else:
                        layer.edge_color = self.ch_point_color(c)
                except Exception:  # noqa: BLE001
                    pass
            elif nm and layer.name == f"预处理-{nm}":
                layer.colormap = self.ch_cmap(c)

    def _save_later(self):
        if not hasattr(self, "_save_timer"):
            self._save_timer = QTimer(self); self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(lambda: self.project and self.project.save())
        self._save_timer.start(500)

    # ================================================================ ④ 细胞识别
    def _panel_detection(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("Cellpose 4（Cellpose-SAM）逐通道识别细胞，可按通道设置掩膜阈值。\n"
                         "建议流程：先把看图窗口放大到一块有代表性的区域，点“预览当前视野”，"
                         "对照图片检查圈出的细胞是否合适，再调整参数；满意后点“识别当前图片”或“批量识别”。"))
        from .device_panel import DetectionDeviceBox
        self.device_box = DetectionDeviceBox(self.run_task)
        L.addWidget(self.device_box)
        tb = QGroupBox("切片厚度"); tl = QVBoxLayout(tb)
        self.thick_combo = QComboBox()
        self.thick_combo.addItem("薄切片（≤ 20 µm）", "thin")
        self.thick_combo.addItem("厚切片（30–50 µm，如 40 µm 漂片）", "thick")
        bt = QPushButton("使用这个厚度的推荐参数"); bt.clicked.connect(self.apply_thickness)
        tl.addWidget(self.thick_combo); tl.addWidget(bt)
        tl.addWidget(QLabel("厚切片推荐：识别前减背景 25 µm（去掉焦平面外的雾）并去卷积（细胞边界更清楚），"
                            "去掉比周围亮不到 1.5 倍的模糊细胞，共标距离放宽到 8 µm。"
                            "z-stack 图片请在“① 项目与图片”中选择合成方式。"))
        tl.itemAt(2).widget().setWordWrap(True)
        L.addWidget(tb)
        L.addWidget(self._unmix_box())
        self.det_rows = QVBoxLayout(); L.addLayout(self.det_rows)
        b0 = big_button("预览当前视野的减背景/去卷积效果"); b0.clicked.connect(lambda: self._require_scope() and self.preview_enhance())
        L.addWidget(b0)
        b1 = big_button("预览当前视野的细胞识别"); b1.clicked.connect(lambda: self._require_scope() and self.preview_detection())
        b2 = big_button("选择处理范围并识别…"); b2.clicked.connect(lambda: self.choose_processing_scope() and self.run_detection(True))
        L.addWidget(b1)
        L.addWidget(b2)
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
        b = QPushButton("保存手动校正"); b.clicked.connect(lambda: self._require_scope() and self.save_manual()); ml.addWidget(b)
        self.man_info = QLabel(""); self.man_info.setWordWrap(True); ml.addWidget(self.man_info)
        L.addWidget(mb)
        self._accuracy_widgets(L)

        tb2 = QGroupBox("保存人工校正的标注块"); tl2 = QVBoxLayout(tb2)
        lb = QLabel("将视野移到目标区域，点击“取一块并自动标注”。用画笔或橡皮修正轮廓，"
                    "按 M 开始下一个细胞，再点“保存这块”。标注块按通道保存在项目的 training 文件夹中。")
        lb.setWordWrap(True); tl2.addWidget(lb)
        r = QHBoxLayout(); r.addWidget(QLabel("通道：")); self.ft_ch = QComboBox(); r.addWidget(self.ft_ch, 1); tl2.addLayout(r)
        self.ft_ch.currentIndexChanged.connect(lambda _: self._ft_refresh())
        r = QHBoxLayout()
        self.ft_new_btn = QPushButton("取一块并自动标注")
        self.ft_new_btn.clicked.connect(self.ft_new_patch); r.addWidget(self.ft_new_btn)
        self.ft_save_btn = QPushButton("保存这块")
        self.ft_save_btn.clicked.connect(self.ft_save_patch); r.addWidget(self.ft_save_btn)
        tl2.addLayout(r)
        self.ft_info = QLabel(""); self.ft_info.setWordWrap(True); tl2.addWidget(self.ft_info)
        L.addWidget(tb2)
        self.det_summary = QLabel(""); self.det_summary.setWordWrap(True)
        L.addWidget(self.det_summary)
        L.addStretch(1)
        return w

    # ---------------------------------------------------------------- 识别前去除通道串色（v0.8.2）
    UNMIX_OFF_WARN = "可能把串过来的信号当成阳性细胞，共标数会偏高。"

    def _unmix_box(self):
        box = QGroupBox("通道串色"); v = QVBoxLayout(box)
        self.um_on = QCheckBox("识别前去除通道串色（推荐）")
        self.um_on.setToolTip("短波长染料的荧光会有一部分“漏”进长波长通道（例如 488 漏进 647），\n"
                              "让那个通道里出现假的阳性细胞、共标数偏高。软件从图片本身自动估算漏进去的比例并扣除，\n"
                              "不需要单染对照。识别、细胞亮度和脑区平均荧光强度都用去除串色后的图。")
        self.um_on.toggled.connect(self._unmix_toggled)
        v.addWidget(self.um_on)
        self.um_warn = QLabel("⚠ " + self.UNMIX_OFF_WARN); self.um_warn.setWordWrap(True)
        self.um_warn.setStyleSheet("color: #d08000;")
        v.addWidget(self.um_warn)
        self.um_info = QLabel(""); self.um_info.setWordWrap(True); self.um_info.setStyleSheet("color: gray;")
        v.addWidget(self.um_info)
        self.um_sliders = QFormLayout(); v.addLayout(self.um_sliders)
        self.um_est = QPushButton("重新估算串色比例（用当前图片）"); self.um_est.clicked.connect(self._unmix_estimate)
        v.addWidget(self.um_est)
        return box

    def _sync_unmix_box(self):
        p = self.project
        while self.um_sliders.rowCount():
            self.um_sliders.removeRow(0)
        d = p.unmix or {}
        on = bool(d.get("enabled"))
        self.um_on.blockSignals(True); self.um_on.setChecked(on); self.um_on.blockSignals(False)
        self.um_warn.setVisible(not on)
        self.um_est.setEnabled(on)
        if not on:
            self.um_info.setText("")
            return
        if "coef" not in d:
            self.um_info.setText("第一次识别（或预览）时会自动估算各通道的串色比例，估算后可以在这里微调。")
            return
        names = p.channel_names
        if d.get("names") and list(d["names"]) != list(names):
            self.um_info.setText("通道名称改过了，建议点下面的按钮重新估算。")
        else:
            self.um_info.setText(f"自动估算（用“{d.get('image', '')}”）。拖动滑块可以微调；"
                                 "比例太大会把弱的真阳性细胞也减掉，太小则去不干净。改动后需要重新识别。")
        if not d.get("coef"):
            self.um_sliders.addRow(QLabel("没有发现明显的串色，识别时不需要扣除。"))
        for k, a in sorted(d.get("coef", {}).items()):
            j, i = (int(x) for x in k.split(","))
            if j >= len(names) or i >= len(names) or j in p.unused_channels or i in p.unused_channels:
                continue
            auto = d.get("auto", {}).get(k, a)
            sl = QSlider(Qt.Horizontal); sl.setRange(0, 400); sl.setValue(int(round(a * 1000)))  # 0.1% 一格，最多 40%
            lab = QLabel(f"{a * 100:.1f}%"); lab.setMinimumWidth(110)
            def moved(val, k=k, lab=lab, auto=auto):
                lab.setText(f"{val / 10:.1f}%" + ("" if abs(val / 1000 - auto) < 5e-4 else f"（自动 {auto * 100:.1f}%）"))
            def released(k=k, sl=sl):
                self.project.unmix.setdefault("coef", {})[k] = round(sl.value() / 1000, 4)
                self.project.save()
                self.statusBar().showMessage("串色比例已改。需要重新识别细胞才会生效。", 8000)
            sl.valueChanged.connect(moved); sl.sliderReleased.connect(released)
            sl.valueChanged.connect(lambda _v, sl=sl, released=released: None if sl.isSliderDown() else released())
            moved(sl.value())
            row = QHBoxLayout(); row.addWidget(sl, 1); row.addWidget(lab)
            self.um_sliders.addRow(f"{names[j]} → {names[i]}", row)

    def _unmix_toggled(self, on):
        self.project.unmix = {**(self.project.unmix or {}), "enabled": bool(on)}
        self.project.save()
        self._sync_unmix_box()
        if not on:
            self.info("已关闭“识别前去除通道串色”。\n\n" + self.UNMIX_OFF_WARN +
                      "\n\n改动后需要重新识别细胞才会生效。")
        else:
            self.statusBar().showMessage("已开启识别前去除通道串色。需要重新识别细胞才会生效。", 8000)

    def _unmix_estimate(self):
        if self.image is None:
            self.error("请先在上方选择一张图片。"); return
        project, image = self.project, self.image

        def job(task):
            from ..core.unmix import estimate_project
            task.report(0, 1, "正在估算通道串色比例…")
            return estimate_project(project, image=image)

        def done(d):
            self._sync_unmix_box()
            lines = d.get("detail") or ["没有发现明显的串色。"]
            self.info("串色比例已重新估算：\n" + "\n".join(lines) + "\n\n需要重新识别细胞才会生效。")

        self.run_task(job, done, label="正在估算通道串色比例…")

    def _sync_detection_panel(self):
        try:
            self._sync_unmix_box()
        except AttributeError:
            pass
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
            if c in self.project.unused_channels:
                continue            # “② 脑片拆分”里设为不使用的通道
            p = self.project.detection.channels[c]
            box = QGroupBox(nm); f = QFormLayout(box)
            on = QCheckBox("识别这个通道的细胞"); on.setChecked(p.detect)
            amin = QDoubleSpinBox(); amin.setRange(1, 5000); amin.setValue(p.min_area_um2); amin.setSuffix(" µm²")
            amax = QDoubleSpinBox(); amax.setRange(10, 20000); amax.setValue(p.max_area_um2); amax.setSuffix(" µm²")
            imin = QDoubleSpinBox(); imin.setRange(0, 65535); imin.setValue(p.min_intensity)
            imin.setToolTip("细胞内平均荧光强度低于此值的不计为阳性。0 表示不限制。")
            cp_thresh = QDoubleSpinBox()
            cp_thresh.setRange(-6, 6)
            cp_thresh.setSingleStep(0.5)
            cp_thresh.setValue(p.cellprob_threshold)
            cp_thresh.setToolTip("Cellpose 的细胞像素得分阈值，默认 0。降低阈值可纳入较弱的像素。")
            f.addRow(on)
            f.addRow("Cellpose 掩膜阈值", cp_thresh)
            f.addRow("最小面积", amin); f.addRow("最大面积", amax)
            skip_dark = QCheckBox("跳过极暗分块"); skip_dark.setChecked(p.skip_dark_tiles)
            skip_dark.setToolTip("对照弱阳性漏检时可取消勾选，然后重新识别。")
            f.addRow(skip_dark)
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
                           "0 表示不使用。调整后请点“选择处理范围并识别…”。")
            f.addRow("局部对比度下限", mct)
            bgr = QDoubleSpinBox(); bgr.setRange(0, 500); bgr.setSingleStep(5); bgr.setValue(p.bg_radius_um)
            bgr.setSuffix(" µm"); bgr.setSpecialValueText("不减背景")
            bgr.setToolTip("识别前去掉大范围的雾状背景。半径要比细胞大，一般 20–50 µm。只影响识别，强度测量仍用原图。")
            f.addRow("识别前减背景", bgr)
            dcv = QCheckBox("识别前去卷积"); dcv.setChecked(p.deconv)
            dcv.setToolTip("对模糊的细胞边界做去卷积。可在同一视野预览效果。\n"
                           "厚切片推荐勾选（厚切片预设默认勾选），薄切片一般不需要（默认不勾选）。")
            dcb = QDoubleSpinBox(); dcb.setRange(0.1, 3.0); dcb.setSingleStep(0.1); dcb.setValue(p.deconv_blur_um); dcb.setSuffix(" µm")
            dcb.setToolTip("图像的模糊程度。太大会让细胞变成空心圆环，样图建议 0.4 µm。")
            dci = QDoubleSpinBox(); dci.setDecimals(0); dci.setRange(1, 50); dci.setValue(p.deconv_iter)
            f.addRow(dcv); f.addRow("去卷积模糊程度", dcb); f.addRow("去卷积迭代次数", dci)

            def apply(_=None, c=c, on=on, amin=amin, amax=amax, imin=imin, gm=gm, bgf=bgf,
                      bgr=bgr, dcv=dcv, dcb=dcb, dci=dci, mct=mct, skip_dark=skip_dark, cp_thresh=cp_thresh):
                p = self.project.detection.channels[c]
                p.skip_dark_tiles = skip_dark.isChecked()
                p.min_contrast = mct.value()
                p.gamma = gm.value(); p.bg_fold = bgf.value()
                p.bg_radius_um = bgr.value(); p.deconv = dcv.isChecked()
                p.deconv_blur_um = dcb.value(); p.deconv_iter = int(dci.value())
                p.detect = on.isChecked()
                p.cellprob_threshold = cp_thresh.value()
                p.min_area_um2 = amin.value(); p.max_area_um2 = amax.value(); p.min_intensity = imin.value()
                self._save_later()

            for wdg in (cp_thresh, amin, amax, imin, gm, bgf, bgr, dcb, dci, mct):
                wdg.valueChanged.connect(apply)
            skip_dark.toggled.connect(apply)
            on.toggled.connect(apply)
            dcv.toggled.connect(apply)
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
        if w <= 0 or h <= 0:
            self.info("请先把视野移到图片范围内。")
            return
        field_box = (x, y, w, h)
        sections = []
        for part in self.image.sections:
            sx, sy, sw, sh = part.bbox0
            if self._section_in_scope(self.image, part) and part.enabled and x < sx+sw and y < sy+sh and sx < x+w and sy < y+h:
                sections.append(part)
        if self.image.sections and not sections:
            self.info("请将视野移到已启用的脑片上。")
            return
        if not sections:
            sections = [Section(id=0, bbox0=field_box, contour0=[], name="预览")]
        settings = self.project.detection

        project, image = self.project, self.image

        def job(task):
            s = image.open_slide(z_mode=project.z_mode)
            try:
                from ..core.unmix import prepare as _prepare_unmix
                if (project.unmix or {}).get("enabled") and "coef" not in project.unmix:
                    task.report(0, 1, "第一次识别：正在估算通道串色比例…")
                _prepare_unmix(project, s, image)
                from ..core.field_review import effective_settings
                frames = []
                for part in sections:
                    preview_settings = effective_settings(settings, image, part)
                    active = {c: p for c, p in preview_settings.channels.items() if p.detect}
                    task.report(0, 1, f"正在读取 {part.name} 的归一化范围…")
                    norms = det.preview_norms(s, part, active, preview_settings.unmix,
                                              sl.preview_norm_cache, task.is_cancelled)
                    frame = det.detect_section(s, part, preview_settings, field_box=field_box,
                                               norms=norms,
                                               progress=lambda d, t: task.report(d, t, "正在预览识别…"),
                                               cancel=task.is_cancelled)
                    frame.attrs.clear()
                    frames.append(frame)
                return pd.concat(frames, ignore_index=True)
            finally:
                s.close()

        def done(df):
            if self.project is not project or self.slide is not sl:
                return
            self._sync_unmix_box()
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
                                           edge_color=self.ch_point_color(c), face_color="transparent",
                                           edge_width=max(w, h) / 800)
            scope = "按脑片参数预览" if image.sections else "未拆分脑片，按当前视野归一化预览"
            self.det_summary.setText(scope + "：" + "，".join(msg) + "。")

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
        while lv + 1 < sl.n_levels and max(w, h) / sl.downsample(lv) > 2048:
            lv += 1
        x, y = max(0, x), max(0, y)
        w, h = min(w, W0 - x), min(h, H0 - y)
        if w <= 0 or h <= 0:
            self.info("当前视野不在图片范围内，请先把图片移到视野中间再预览。")
            return
        from ..core.field_review import effective_settings
        from ..core.manual import section_of
        name = section_of(self.image, x+w/2, y+h/2)
        sec = next((s for s in self.image.sections if s.name == name), None)
        if (self._processing_scope or {}).get(self.image.key) is not None:
            if sec is None or not self._section_in_scope(self.image, sec):
                self.info("请将预览视野移到所选脑片上。")
                return
            sx, sy, sw, sh = sec.bbox0
            right, bottom = min(x + w, sx + sw), min(y + h, sy + sh)
            x, y = max(sx, x), max(sy, y)
            w, h = right - x, bottom - y
        settings = effective_settings(self.project.detection, self.image, sec) if sec else self.project.detection
        chans = [c for c, p in settings.channels.items() if p.deconv or p.bg_radius_um > 0]
        if not chans:
            self.info("还没有任何通道勾选“去卷积”或设置“减背景”。")
            return
        params = {c: settings.channels[c] for c in chans}
        project, image = self.project, self.image
        ds = sl.downsample(lv)
        pix = sl.pixel_size_um * ds

        def job(task):
            s = image.open_slide(z_mode=project.z_mode)
            try:
                a = s.read(lv, (x, y, w, h), chans)
            finally:
                s.close()
            out = []
            for k, c in enumerate(chans):
                task.report(k, len(chans), f"正在处理 {self.ch_name(c)}…")
                p = params[c]
                out.append(enhance(a[k], pix, p.deconv, p.deconv_blur_um, p.deconv_iter, p.bg_radius_um))
            return out

        def done(res):
            if self.project is not project or self.slide is not sl:
                return
            for nm in [l.name for l in self.viewer.layers if l.name.startswith("预处理-")]:
                self.viewer.layers.remove(nm)
            for c, im in zip(chans, res):
                nm = self.ch_name(c)
                lo, hi = np.percentile(im, [1, 99.7])
                self.viewer.add_image(im, name=f"预处理-{nm}", colormap=self.ch_cmap(c),
                                      blending="additive", scale=(ds, ds), translate=(y + ds / 2, x + ds / 2),
                                      contrast_limits=(float(lo), float(max(hi, lo + 1))))
                if c < len(self.image_layers):
                    self.image_layers[c].visible = False
            self.statusBar().showMessage("已显示预处理后的图像（“预处理-”图层）。在左侧图层列表中点眼睛图标可与原图对比。", 15000)

        self.run_task(job, done, label="正在预览预处理效果…")

    def choose_detection_sections(self):
        sections = [s for s in self.image.sections if s.enabled] if self.image else []
        if not sections:
            self.info("请先在「② 脑片拆分」中添加脑片，并勾选需要分析的脑片。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("选择要识别的脑片")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("勾选本次要识别的脑片，可多选。"))
        items = QListWidget()
        layout.addWidget(items)
        from ..core.pipeline import drop_excluded
        from ..core.project import section_result_states
        cells = drop_excluded(self.cells_for(self.image), self.image)
        states = section_result_states(self.image, cells)
        carried = set(cells.attrs.get("carried_forward_sections", [])) if cells is not None else set()
        for section in sections:
            status = states[section.name]
            if cells is not None and status != "未分析":
                saved_models = cells.attrs.get("section_models", {}).get(section.name, {})
                if not saved_models or any(key != cellpose4.MODEL_KEY for key in saved_models.values()):
                    status += "，历史结果（可用 Cellpose 4 重跑）"
            if section.name in carried:
                status += "，沿用上次筛选结果"
            item = QListWidgetItem(f"{section.name}　{status}", items)
            item.setData(Qt.UserRole, section.name)
            item.setCheckState(Qt.Unchecked)
        all_button = QPushButton("全选")
        def select_all():
            for i in range(items.count()):
                items.item(i).setCheckState(Qt.Checked)
        all_button.clicked.connect(select_all)
        layout.addWidget(all_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        start = buttons.button(QDialogButtonBox.Ok)
        start.setText("开始识别")
        start.setEnabled(False)
        def selected():
            return [items.item(i).data(Qt.UserRole) for i in range(items.count())
                    if items.item(i).checkState() == Qt.Checked]
        items.itemChanged.connect(lambda _: start.setEnabled(bool(selected())))
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        fit_window(dlg, 360, 420)
        if dlg.exec_() == QDialog.Accepted:
            self.run_detection(False, selected())

    def run_detection(self, all_images: bool, section_names=None):
        targets = self._target_images(all_images)
        selection = self._processing_scope or {}
        names_by_key = {}
        for im in targets:
            names = section_names if section_names is not None else selection.get(im.key)
            names_by_key[im.key] = None if names is None else [s.name for s in im.sections if s.enabled and s.name in names]
        targets = [im for im in targets if names_by_key[im.key] is None or names_by_key[im.key]]
        if not targets:
            self.info("所选脑片均未启用分析。可在“② 脑片拆分”中勾选“分析”。")
            return
        if not all_images and not self.image.sections:
            self.info("当前图片还没有拆分脑片，请先到“② 脑片拆分”。")
            return
        project = self.project

        def job(task):
            from ..core.pipeline import detect_image
            from ..core.log import friendly, log
            import traceback
            summary, errors, reused = {}, [], 0
            for i, im in enumerate(targets):
                try:
                    r = detect_image(project, im, report=task.report, cancel=task.is_cancelled,
                                     progress_range=(i * 100, i * 100 + 100), total=len(targets) * 100,
                                     section_names=names_by_key[im.key])
                    reused += r["reused"]
                    summary[im.key] = project.load_cells(im)
                except InterruptedError:
                    project.save()
                    raise
                except Exception as e:  # noqa: BLE001  一张图出错不影响其他图片
                    log.error("识别 %s (%s) 出错\n%s", im.display_name, im.path, traceback.format_exc())
                    im.error = friendly(e); im.status = "出错"
                    errors.append(f"{im.display_name}：{im.error}")
            project.save()
            return summary, errors, reused

        def done(out):
            res, errors, reused = out
            self.project.save(); self._refresh_images(); self._sync_unmix_box()
            for k_, df in res.items():
                self._cells_cache[k_] = df
            self._draw_cells()
            lines = []
            if section_names:
                lines.append("本次选择：" + "、".join(section_names))
            if reused:
                lines.append(f"（有 {reused} 张脑片的识别参数没有变化，直接使用了上次的结果）")
            if errors:
                self.error("以下图片识别出错，已跳过，其他图片正常完成：\n" + "\n".join(errors))
            for k_, df in res.items():
                if df is None:
                    continue
                if names_by_key[k_] is not None:
                    df = df[df["section"].isin(names_by_key[k_])]
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
        if th == "thick":
            self.info("已使用厚切片推荐参数。\n\n"
                      "注意：所有通道已默认勾选“识别前去卷积”。\n"
                      "• 去卷积让厚切片中模糊的细胞边界变清楚，粘在一起的细胞更容易分开；\n"
                      "• 只用于找细胞，细胞的亮度、阳性判断和脑区平均荧光强度仍按原图测量；\n"
                      "• 开启去卷积会增加处理时间。\n\n"
                      "不需要时可以在下面各通道里取消勾选。建议先放大到一块区域，"
                      "点“预览当前视野的减背景/去卷积效果”和“预览当前视野的细胞识别”检查效果，再识别全部脑片。")
        else:
            self.info("已使用薄切片推荐参数（不减背景、不去卷积）。\n"
                      "建议先放大到一块区域点“预览当前视野的细胞识别”检查效果，再识别全部脑片。")

    # ---------------- 手动校正
    def _man_layer(self, create=True):
        nm = self.man_ch.currentText()
        if not nm or self.slide is None:
            return None
        name = f"细胞-{nm}"
        if name not in self.viewer.layers:
            if not create:
                return None
            size = 10 / self.slide.pixel_size_um
            layer = self.viewer.add_points(np.zeros((0, 2)), name=name, size=size, face_color="transparent",
                                           border_color=self.ch_point_color(self._ch_index(nm)), border_width=0.15,
                                           features={"cid": np.zeros(0, int)})
            layer.feature_defaults = {"cid": -1}
        return self.viewer.layers[name]

    def _man_mode(self, mode):
        if self.slide is None:
            return
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
        from ..core.pipeline import refresh_coloc
        if self.slide is None:
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
            names = (self._processing_scope or {}).get(self.image.key)
            if names is not None:
                orig = orig[orig["section"].isin(names)]
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
                if not sec or (names is not None and sec not in names):
                    n_out += 1; continue
                means, cont = measure_added(self.slide, x, y, px)
                man["added"].append({"channel": c, "x": float(x), "y": float(y), "section": sec,
                                     "means": means, "contour": cont})
                n_add += 1
        save_manual(self.project, self.image, man)
        project = self.project
        selection = {self.image.key: (self._processing_scope or {}).get(self.image.key)}

        def done(_):
            self._cells_cache.clear(); self._draw_cells(); self._refresh_coloc_table(); self._man_refresh_info()
            msg = f"已保存：补加 {n_add} 个，删除 {n_del} 个。"
            if n_out:
                msg += f"\n有 {n_out} 个补加的点不在任何脑片方框内，已忽略。"
            self.statusBar().showMessage(msg.replace("\n", " "), 10000)

        self.run_task(lambda task: refresh_coloc(project, task.report, selection=selection, cancel=task.is_cancelled), done, label="正在更新细胞和共标…")

    # ---------------- 人工标注块
    def _ft_refresh(self):
        if self.project is None or not self.ft_ch.currentText():
            return
        from ..core.finetune import list_patches
        n = len(list_patches(self.project, self.ft_ch.currentIndex()))
        self.ft_info.setText(f"“{self.ft_ch.currentText()}”已保存 {n} 个标注块。")

    def ft_new_patch(self):
        if self.slide is None:
            self.info("请先在顶部选择一张图片。"); return
        from ..core.finetune import make_patch
        c = self.ft_ch.currentIndex()
        cam = self.viewer.camera
        center = (cam.center[-1], cam.center[-2])
        project, image, view_slide = self.project, self.image, self.slide

        def job(task):
            task.report(0, 1, "正在自动识别这一块…")
            sl = image.open_slide(z_mode=project.z_mode)
            try:
                return make_patch(project, sl, image, center, c, cancel=task.is_cancelled)
            finally:
                sl.close()

        def done(res):
            if self.project is not project or self.slide is not view_slide:
                return
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
        if self.image is None or self.slide is None:
            return
        df = self.cells_for(self.image)
        if df is None or df.empty:
            return
        if not {"channel", "section", "x", "y"}.issubset(df.columns):
            self.statusBar().showMessage("已有细胞结果缺少坐标或通道信息，可在细胞识别步骤重新处理。", 10000)
            return
        chosen = (self._processing_scope or {}).get(self.image.key)
        if chosen is not None:
            df = df[df["section"].isin(chosen)]
        size = 10 / self.slide.pixel_size_um
        for c, nm in enumerate(self.project.channel_names):
            g = df[df["channel"] == c]
            if g.empty:
                continue
            layer = self.viewer.add_points(g[["y", "x"]].to_numpy(), name=f"细胞-{nm}", size=size,
                                           face_color="transparent", border_color=self.ch_point_color(c),
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
        b = big_button("重新计算所选脑片的共标"); b.clicked.connect(lambda: self._require_scope() and self.recompute_coloc())
        L.addWidget(b)
        self.coloc_table = QTableWidget(); self.coloc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.coloc_table.setMinimumHeight(220)
        L.addWidget(self.coloc_table, 1)
        return w

    def recompute_coloc(self):
        self.project.coloc.max_dist_um = self.coloc_dist.value()
        self.project.coloc.use_contour = self.coloc_contour.isChecked()
        self.project.coloc.nucleus_by_intensity = self.coloc_nuc_int.isChecked()
        self.project.coloc.require_nucleus = self.coloc_req_nuc.isChecked()
        self.project.save()
        project = self.project
        selection = self._processing_scope

        def job(task):
            from ..core.pipeline import refresh_coloc
            return refresh_coloc(project, task.report, selection=selection, cancel=task.is_cancelled)

        def done(_):
            self._cells_cache.clear(); self._refresh_coloc_table()

        self.run_task(job, done, label="正在计算共标…")

    def _refresh_coloc_table(self):
        if not self.project:
            return
        self.coloc_dist.setValue(self.project.coloc.max_dist_um)
        self.coloc_contour.setChecked(self.project.coloc.use_contour)
        self.coloc_nuc_int.setChecked(self.project.coloc.nucleus_by_intensity)
        self.coloc_req_nuc.setChecked(self.project.coloc.require_nucleus)
        frames = []
        for im in self._view_project().images:
            df = self.cells_for(im)
            if df is None or df.empty or "merged_id" not in df.columns:
                continue
            enabled = {s.name for s in im.sections if s.enabled}
            from ..core.pipeline import drop_excluded
            cells = merged_cells(drop_excluded(df[df["section"].isin(enabled)], im), len(self.project.channel_names))
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
        L.addWidget(hint("将所选图片和脑片的统计结果导出为 Excel，其中包括：\n"
                         "• 按脑片汇总：每张脑片各通道阳性细胞数和共标数\n"
                         "• 按脑区×脑片：每张脑片每个脑区的细胞数、面积和密度，默认左右合并（需先配准）\n"
                         "• 分析参数：方便以后复现\n"
                         "如果在“② 脑片拆分”中画了自定义区域，会多一张“按自定义区域”表。"))
        ob = QGroupBox("导出选项"); of = QVBoxLayout(ob)
        from .region_scope import RegionScopeBox
        rb = QGroupBox("“按脑区×脑片”表里列出哪些脑区"); rl = QVBoxLayout(rb)
        self.exp_scope = RegionScopeBox(); rl.addWidget(self.exp_scope)
        self.exp_scope.changed.connect(self._save_export_scope)
        rl.addWidget(QLabel("半脑导出方式"))
        self.exp_hemisphere = QComboBox()
        self.exp_hemisphere.addItems(list(EXPORT_HEMISPHERES))
        rl.addWidget(self.exp_hemisphere)
        rl.addWidget(hint("一个父脑区的行包括它下面所有子区的细胞和面积。若同时列出父区和子区，请分别解读这两行。"))
        self.exp_masks = QCheckBox("同时保存细胞 mask（1/3 分辨率）"); self.exp_masks.setChecked(True)
        self.exp_qc = QCheckBox("同时生成每张脑片的质控图（PNG，方便快速检查）"); self.exp_qc.setChecked(True)
        self.exp_intensity = QCheckBox("统计每个脑区的平均荧光强度（较慢）")
        self.exp_intensity.setToolTip("按原始分辨率读一遍原图，算每个脑区（和自定义区域）各通道的平均灰度值。\n"
                                      "图片越大越慢（每张图片多几十秒）；只看细胞数和密度时不用勾选。\n"
                                      "算过一次后会保存下来，配准和排除区域没改时再次导出不用重新算。\n"
                                      "组间统计、行为相关、功能网络不需要这一项。\n"
                                      "“④ 细胞识别”里开启了去除通道串色时，按去除串色后的图计算，与细胞识别口径一致。")
        for wdg in (self.exp_masks, self.exp_qc, self.exp_intensity):
            of.addWidget(wdg)
        of.addWidget(hint("勾选“平均荧光强度”会增加导出时间（需要读原图），但可以比较各脑区的整体荧光亮度。"))
        L.addWidget(rb)
        L.addWidget(ob)
        b = big_button("导出 Excel…"); b.clicked.connect(lambda: self._require_scope() and self.export())
        L.addWidget(b)
        b2 = big_button("打开项目文件夹"); b2.clicked.connect(lambda: self._open_folder(self.project.folder))
        L.addWidget(b2)
        L.addStretch(1)
        return w

    def _sync_export_panel(self):
        p = self.project
        self.exp_scope.blockSignals(True)
        try:
            self.exp_scope.set_values(p.export_region_set, p.export_max_depth, p.export_skip_layers, p.export_region_ids)
        finally:
            self.exp_scope.blockSignals(False)
        self.exp_intensity.setChecked(p.export_intensity)
        self.exp_hemisphere.setCurrentText(p.export_hemisphere)

    def _save_export_scope(self):
        if self.project is None or getattr(self, "_export_task", None) is not None:
            return
        p = self.project
        p.export_region_set = self.exp_scope.mode()
        p.export_region_ids = self.exp_scope.region_ids()
        p.export_skip_layers = self.exp_scope.skip_layers()
        p.export_max_depth = 0
        p.save()

    def export(self):
        from ..core.region_sets import CUSTOM
        if self.task is not None and self.task.isRunning():
            self.info("上一个任务还在运行，请等待完成或点击“取消”。")
            return
        if self.exp_scope.mode() == CUSTOM and not self.exp_scope.region_ids():
            self.info("请在“编辑已选脑区”中勾选至少一个脑区，或切换到其他导出范围。")
            return
        default = str(self.project.folder / "细胞统计结果.xlsx")
        f, _ = QFileDialog.getSaveFileName(self, "保存 Excel", default, "Excel 文件 (*.xlsx)")
        if not f:
            return
        project = self.project
        project.export_intensity = self.exp_intensity.isChecked()
        project.export_hemisphere = self.exp_hemisphere.currentText()
        project.save()
        if getattr(self, '_processing_scope', None) is not None:
            project = self._view_project()
        masks, qc = self.exp_masks.isChecked(), self.exp_qc.isChecked()

        def done(p):
            extra = []
            if masks:
                extra.append(f"• 细胞 mask：“{companion_dir(project, p, '细胞mask').name}”文件夹（原图 1/3 分辨率）")
            if qc:
                extra.append(f"• 质控图：“{companion_dir(project, p, '质控图').name}”文件夹，每张脑片一张，可以快速翻看识别和配准是否正常")
            self.info(f"已导出：\n{p}\n\n同一文件夹中还有：\n" + "\n".join(extra) if extra else f"已导出：\n{p}")

        self.run_task(lambda task: export_excel(project, f, masks=masks, qc=qc, progress=task.report),
                      done, label="正在导出 Excel…", lock_export=True)

    def _target_images(self, all_images):
        candidates = self.project.images if all_images else [self.image]
        return [im for im in candidates if im is not None and
                (self._processing_scope is None or im.key in self._processing_scope)]

    def _open_folder(self, path):
        path = str(path)
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
