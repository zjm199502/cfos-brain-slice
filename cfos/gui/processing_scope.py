"""Select project images/scenes and named sections for the workflow."""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                            QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)
from .layout import fit_window


class ScopeDialog(QDialog):
    def __init__(self, images, previous, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择处理的图片和脑片")
        fit_window(self, 720, 490)
        layout = QVBoxLayout(self)
        help_text = QLabel("勾选图片可处理整图；展开后可单选或多选脑片。相同文件名可按来源路径区分。")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["图片 / 脑片", "原图位置"])
        layout.addWidget(self.tree)
        for im in images:
            root = QTreeWidgetItem(self.tree, [im.display_name, im.path])
            root.setToolTip(0, im.path)
            root.setToolTip(1, im.path)
            root.setData(0, Qt.UserRole, im.key)
            root.setFlags(root.flags() | Qt.ItemIsUserCheckable)
            whole = im.key in previous and previous[im.key] is None
            root.setCheckState(0, Qt.Checked if whole else Qt.Unchecked)
            for s in im.sections:
                item = QTreeWidgetItem(root, [s.name if s.enabled else s.name + "（未启用分析）"])
                item.setData(0, Qt.UserRole, s.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if whole or s.name in (previous.get(im.key) or ()) else Qt.Unchecked)
            if not whole and any(root.child(j).checkState(0) == Qt.Checked for j in range(root.childCount())):
                root.setCheckState(0, Qt.PartiallyChecked)
            root.setExpanded(len(images) < 8 or im.key in previous)
        self.tree.resizeColumnToContents(0)
        actions = QHBoxLayout()
        for label, state in (("全选", Qt.Checked), ("清空", Qt.Unchecked)):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, st=state: self._set_all(st))
            actions.addWidget(btn)
        actions.addStretch()
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        self.ok.setText("应用范围")
        self.ok.setEnabled(bool(self.selection()))
        self.tree.itemChanged.connect(self._check_item)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _check_item(self, item, column):
        if column != 0:
            return
        self.tree.blockSignals(True)
        parent = item.parent()
        if parent is None:
            state = item.checkState(0)
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, state)
        else:
            # Individually ticking every existing section still means those named sections.
            # Whole-image selection can also re-split the image or include future sections.
            selected = any(parent.child(j).checkState(0) == Qt.Checked for j in range(parent.childCount()))
            parent.setCheckState(0, Qt.PartiallyChecked if selected else Qt.Unchecked)
        self.tree.blockSignals(False)
        self.ok.setEnabled(bool(self.selection()))

    def _set_all(self, state):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            root.setCheckState(0, state)
            for j in range(root.childCount()):
                root.child(j).setCheckState(0, state)
        self.tree.blockSignals(False)
        self.ok.setEnabled(bool(self.selection()))

    def selection(self):
        chosen = {}
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            key = root.data(0, Qt.UserRole)
            if root.checkState(0) == Qt.Checked:
                chosen[key] = None
            elif root.checkState(0) == Qt.PartiallyChecked:
                chosen[key] = tuple(root.child(j).data(0, Qt.UserRole) for j in range(root.childCount())
                                    if root.child(j).checkState(0) == Qt.Checked)
        return chosen


class ProcessingScopeMixin:
    def _scope_bar(self):
        row = QHBoxLayout()
        self._processing_scope = None
        self.scope_button = QPushButton("选择处理范围…")
        self.scope_button.clicked.connect(self.choose_processing_scope)
        row.addWidget(self.scope_button)
        self.scope_summary = QLabel("尚未选择")
        row.addWidget(self.scope_summary)
        self.focus_combo = QComboBox()
        self.focus_combo.setMinimumContentsLength(6)
        self.focus_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.focus_combo.setToolTip("查看范围中的图片或脑片")
        self.focus_combo.activated.connect(self._focus_target)
        row.addWidget(self.focus_combo, 1)
        return row

    def _view_project(self):
        from ..core.scope import scoped_project
        return self.project if self._processing_scope is None else scoped_project(self.project, self._processing_scope)

    def choose_processing_scope(self, *_):
        if self.task is not None and self.task.isRunning():
            self.info("请在当前任务完成后更改处理范围。")
            return False
        if not self.project or not self.project.images:
            self.info("请先添加图片。")
            return False
        dlg = ScopeDialog(self.project.images, self._processing_scope or {}, self)
        if dlg.exec() != QDialog.Accepted:
            return False
        self._apply_scope(dlg.selection())
        return True

    def _apply_scope(self, selection):
        self._preview_grid = None
        self._processing_scope = selection
        self._refresh_scope()
        self._focus_target(self.focus_combo.currentIndex())
        self._on_step(self.stack.currentIndex())
        if self.slide:
            self._draw_sections()
            self._draw_polygons()
            self._draw_figure_rois()
            self._refresh_fig_list()
            self._draw_cells()

    def _require_scope(self):
        if self._processing_scope is None or not self.focus_combo.count():
            return self.choose_processing_scope()
        return True

    def _refresh_scope(self):
        scope = self._processing_scope
        if scope is not None:
            scope = {im.key: (None if scope[im.key] is None else
                              tuple(s.name for s in im.sections if s.name in scope[im.key]))
                     for im in self.project.images if im.key in scope} if self.project else {}
            self._processing_scope = scope = {key: names for key, names in scope.items() if names is None or names} or None
        combo = self.focus_combo
        prev = combo.currentData()
        combo.clear()
        busy = self.task is not None and self.task.isRunning()
        self.scope_button.setEnabled(bool(self.project and self.project.images) and not busy)
        for idx, im in enumerate(self.project.images if self.project else []):
            allowed = scope is None or im.key in scope
            row = self.image_combo.model().item(idx)
            if row is not None:
                row.setEnabled(allowed)
            if not allowed or scope is None:
                continue
            if scope[im.key] is None:
                combo.addItem(im.display_name, (im.key, None))
            for s in im.sections:
                if scope[im.key] is None or s.name in scope[im.key]:
                    combo.addItem(f"{im.display_name} · {s.name}", (im.key, s.name))
        for i in range(combo.count()):
            if combo.itemData(i) == prev:
                combo.setCurrentIndex(i)
                break
        if scope is None:
            self.scope_summary.setText("尚未选择")
        else:
            whole = sum(names is None for names in scope.values())
            named = sum(len(names) for names in scope.values() if names is not None)
            parts = ([f"{whole} 张整图"] if whole else []) + ([f"{named} 张脑片"] if named else [])
            self.scope_summary.setText("、".join(parts) or "尚未选择")

    def _section_in_scope(self, image, section):
        scope = self._processing_scope
        return scope is None or (image.key in scope and
                                 (scope[image.key] is None or section.name in scope[image.key]))

    def _focus_section(self):
        target = self.focus_combo.currentData()
        if target and self.image and target[0] == self.image.key:
            return next((s for s in self.image.sections if s.name == target[1]), None)
        return None

    def _focus_target(self, index):
        target = self.focus_combo.itemData(index) if index >= 0 else None
        if not target or not self.project:
            return
        for i, im in enumerate(self.project.images):
            if im.key == target[0]:
                self.image_combo.setCurrentIndex(i)
                if target[1] is not None and self.slide:
                    for j, section in enumerate(im.sections):
                        if section.name == target[1]:
                            if self.stack.currentIndex() == 5:
                                self.reg_sec_combo.setCurrentIndex(j)
                            self._zoom_section(j, 0)
                            break
                if self.slide:
                    self._apply_focused_display()
                    if self.stack.currentIndex() == 2:
                        self._sync_display_panel()
                break
