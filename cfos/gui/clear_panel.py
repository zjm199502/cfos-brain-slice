"""每个步骤页底部的“清除本步结果”（v1.0.2）。"""
from qtpy.QtWidgets import QBoxLayout, QCheckBox, QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ..core import clear_results as CR

# 步骤页序号 → 清除哪一步
PAGE_STEPS = [CR.ALL, CR.SECTIONS, CR.DISPLAY, CR.DETECT, CR.COLOC, CR.REGISTRATION, CR.GROUPS, CR.EXPORT, CR.FIGURE]

BUTTON_TEXT = {
    CR.ALL: "清除所选范围的全部分析结果…",
    CR.SECTIONS: "清除所选范围的脑片拆分结果…",
    CR.DISPLAY: "所选范围恢复为项目亮度设置…",
    CR.DETECT: "清除所选范围的细胞识别结果…",
    CR.COLOC: "清除所选范围的共标结果…",
    CR.REGISTRATION: "清除所选范围的配准结果…",
    CR.GROUPS: "清除所选范围的分组…",
    CR.EXPORT: "清除所选范围的导出记录…",
    CR.FIGURE: "清除所选范围的截图框和制图记录…",
}

RERUN_TEXT = dict(zip(PAGE_STEPS[1:], (text for _step, text in CR.rerun_behavior())))
RERUN_TEXT[CR.ALL] = "清除后保留图片，脑片轮廓、识别、共标、配准、分组和人工修正全部删除，可以从“② 脑片拆分”重新开始。"


class ClearResultsMixin:
    def _add_clear_box(self, widget, page):
        layout = widget.layout()
        if not isinstance(layout, QBoxLayout):
            return
        step = PAGE_STEPS[page]
        box = QGroupBox("重新分析"); v = QVBoxLayout(box)
        tip = QLabel("重新分析时：" + RERUN_TEXT[step]); tip.setWordWrap(True)
        tip.setStyleSheet("color:#8b949e;")
        v.addWidget(tip)
        b = QPushButton(BUTTON_TEXT[step])
        b.setStyleSheet("color:#ff7b72;")
        b.setToolTip("先列出将删除的内容和需要重做的后续步骤，确认后才删除。范围与本页“选择处理范围”一致。")
        b.clicked.connect(lambda _=False, step=step: self.clear_step_results(step))
        v.addWidget(b)
        layout.addWidget(box)
        if not hasattr(self, "clear_buttons"):
            self.clear_buttons = {}
        self.clear_buttons[step] = b

    def clear_step_results(self, step, *, confirm=None):
        """confirm：测试用，传入 (report) -> (是否确认, 是否勾选附加项)。"""
        if not self.project:
            self.info("请先打开项目。")
            return None
        if self.task is not None and self.task.isRunning():
            self.info("请在当前任务完成后再清除结果。")
            return None
        if not self._require_scope():
            return None
        selection = self._processing_scope
        preview = CR.clear(self.project, step, selection, dry_run=True)
        extra_label = None
        if step == CR.DETECT and preview.manual_records:
            extra_label = f"同时清除人工修正的细胞（{preview.manual_records} 条补加 / 删除）"
        elif step == CR.GROUPS and preview.output_folders:
            extra_label = f"同时删除项目文件夹里以往的 {len(preview.output_folders)} 个统计结果文件夹"
        if preview.empty and extra_label is None:
            self.info(f"所选范围内没有{CR.STEP_TITLES[step]}可以清除。")
            return preview
        if confirm is not None:
            ok, extra = confirm(preview)
        else:
            ok, extra = self._confirm_clear(step, preview, extra_label)
        if not ok:
            return None
        self.project.wait_saved()  # 后台保存写完再改结果
        report = CR.clear(self.project, step, selection, dry_run=False,
                          include_manual=extra and step == CR.DETECT,
                          include_outputs=extra and step == CR.GROUPS)
        self._after_clear()
        if report.empty:
            self.statusBar().showMessage("没有需要清除的内容。", 6000)
        else:
            self.statusBar().showMessage(f"已清除{CR.STEP_TITLES[step]}。", 8000)
        return report

    def _confirm_clear(self, step, report, extra_label):
        box, go, check = self._clear_dialog(step, report, extra_label)
        box.exec()
        return box.clickedButton() is go, bool(check and check.isChecked())

    def _clear_dialog(self, step, report, extra_label):
        n = f"{report.images} 张图片" + (f"、{report.sections} 张脑片" if report.sections else "")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("清除结果")
        box.setText(f"清除所选范围（{n}）的{CR.STEP_TITLES[step]}？删除后不能撤销。")
        box.setInformativeText(CR.describe(report))
        check = None
        if extra_label:
            check = QCheckBox(extra_label)
            box.setCheckBox(check)
        go = box.addButton("清除", QMessageBox.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)        # 回车 = 取消，避免误删
        return box, go, check

    def _after_clear(self):
        """清掉界面上的缓存，让各页按磁盘上的新状态重新显示。"""
        self._cells_cache.clear()
        self._coloc_summaries.clear()
        self._coloc_generation += 1
        if hasattr(self, "_reg_cache"):
            self._reg_cache.clear()
        self._q_levels = None
        self._preview_grid = None
        self._refresh_images()
        self._refresh_scope()
        if self.slide is not None and self.image is not None:
            self._draw_sections()
            self._draw_cells()
            if hasattr(self, "_draw_figure_rois"):
                self._draw_figure_rois()
        self._on_step(self.stack.currentIndex())
