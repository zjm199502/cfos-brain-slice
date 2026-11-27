"""组间统计和 Excel 导出共用的脑区选择框与图谱树。"""
from __future__ import annotations

from .layout import fit_window
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                            QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                            QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import region_sets as RS
from ..core.atlas_choice import CCF, use_atlas, structure_id
from functools import wraps


def _scope_atlas(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        project = getattr(self.window(), "project", None)
        with use_atlas(getattr(project, "registration_atlas", CCF)):
            return method(self, *args, **kwargs)
    return run


class RegionScopeBox(QWidget):
    changed = Signal()

    def __init__(self, with_layers: bool = True, parent=None):
        super().__init__(parent)
        self._region_ids = []
        self._current_mode = RS.ALL
        L = QVBoxLayout(self); L.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for mode in (RS.ALL, RS.SUMMARY_ONLY, RS.CUSTOM):
            label = "自选脑区…" if mode == RS.CUSTOM else RS.MODE_LABEL[mode]
            self.combo.addItem(label, mode)
            self.combo.setItemData(self.combo.count() - 1, RS.MODE_HELP[mode], Qt.ToolTipRole)
        L.addWidget(self.combo)
        self.layers = QCheckBox("不列出皮层各层（layer 1、2/3、4、5、6a、6b）")
        self.layers.setToolTip("皮层每个区又分 6 层（如 MOp1、MOp2/3…）。不关心分层时勾选，表格会短很多。")
        self.layers.setVisible(with_layers); self._with_layers = with_layers
        L.addWidget(self.layers)
        self.desc = QLabel(); self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:#c9d1d9; background:#262c36; padding:6px; border-radius:5px; font-size:12px;")
        L.addWidget(self.desc)
        self.btn = QPushButton("查看会列出哪些脑区…")
        self.btn.clicked.connect(self.show_preview)
        L.addWidget(self.btn)
        self.combo.currentIndexChanged.connect(self._mode_changed)
        self.layers.toggled.connect(self._sync)
        self._sync()

    # ---- 读 / 写
    def mode(self) -> str:
        return self.combo.currentData()

    def skip_layers(self) -> bool:
        return self._with_layers and self.layers.isEnabled() and self.layers.isChecked()

    def region_ids(self) -> list[int]:
        return self._region_ids.copy()

    @_scope_atlas
    def set_values(self, mode: str, depth: int = 0, skip_layers: bool = False, region_ids=()):
        mode = RS.resolve_mode(mode, depth)
        warning = ""
        if mode in (RS.MAJOR_ONLY, RS.MAJOR_SUMMARY, RS.DEPTH):
            # 旧预设在新版图谱树中呈现为相同的一组 Allen ID。
            try:
                acronyms = RS.selected(mode, depth, skip_layers)
                atlas = RS.info()
                region_ids = atlas.loc[atlas["acronym"].isin(acronyms), "id"].tolist()
            except (OSError, ValueError, KeyError) as exc:
                region_ids = []
                warning = f"无法读取旧项目的脑区清单，请检查图谱或重新选择脑区。\n{exc}"
            mode = RS.CUSTOM
        self._region_ids = sorted(set(region_ids or ()))
        for w in (self.combo, self.layers):
            w.blockSignals(True)
        self.combo.setCurrentIndex(self.combo.findData(mode))
        self.layers.setChecked(bool(skip_layers))
        for w in (self.combo, self.layers):
            w.blockSignals(False)
        self._sync()
        if warning:
            self.desc.setText(warning)

    # ---- 界面
    def _mode_changed(self):
        if self.mode() == RS.CUSTOM and not self._edit_custom():
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(self.combo.findData(self._current_mode))
            self.combo.blockSignals(False)
            return
        self._sync()

    @_scope_atlas
    def _sync(self, *_):
        m = self.mode()
        self._current_mode = m
        self.layers.setEnabled(m == RS.ALL)
        self.btn.setText("编辑已选脑区…" if m == RS.CUSTOM else "查看会列出哪些脑区…")
        if m == RS.CUSTOM:
            n = len(self._region_ids)
            self.desc.setText(f"已选 {n} 个脑区。父区行包含全部下属脑区的统计。"
                              if n else "请至少选择一个脑区，或切换到其他脑区范围。")
        else:
            try:
                count = f"共 {len(RS.selected(m, skip_layers=self.skip_layers()))} 个脑区。"
            except Exception:  # noqa: BLE001  图谱文件缺失时不影响界面
                count = ""
            self.desc.setText(f"{count}{RS.MODE_HELP[m]}")
        self.changed.emit()

    def _edit_custom(self):
        try:
            dialog = RegionPreviewDialog(RS.CUSTOM, parent=self.window(), region_ids=self._region_ids)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "无法读取脑区清单", str(exc))
            return False
        if dialog.exec() != QDialog.Accepted:
            return False
        self._region_ids = sorted(dialog.selected_ids())
        return True

    def show_preview(self):
        if self.mode() == RS.CUSTOM:
            if self._edit_custom():
                self._sync()
        else:
            RegionPreviewDialog(self.mode(), skip_layers=self.skip_layers(), parent=self.window()).exec()


class RegionPreviewDialog(QDialog):
    """预览预设范围或编辑自选范围；父区与子区的勾选相互独立。"""

    def __init__(self, mode, depth=0, skip_layers=False, parent=None, *, region_ids=()):
        super().__init__(parent)
        with use_atlas(getattr(getattr(parent, "project", None), "registration_atlas", CCF)):
            self._populate(mode, depth, skip_layers, region_ids)

    def _populate(self, mode, depth, skip_layers, region_ids):
        custom = mode == RS.CUSTOM
        self.setWindowTitle("选择脑区" if custom else "会列出哪些脑区")
        fit_window(self, 900, 660)
        L = QVBoxLayout(self)
        t = RS.info()
        sel = set(region_ids) if custom else RS.selected(mode, depth, skip_layers)
        head = QLabel("勾选至少一个需要单独列出的脑区。父区的统计包含全部下属脑区；同时勾选父区和子区时，两行的统计会重叠。"
                      if custom else f"当前选法：{RS.MODE_LABEL[RS.resolve_mode(mode, depth)]}\n"
                      f"打 ✔ 的 {len(sel)} 个脑区会出现在结果里。")
        head.setWordWrap(True); L.addWidget(head)
        tree_layout = L
        if custom:
            self.tabs = QTabWidget()
            L.addWidget(self.tabs, 1)
            browse = QWidget()
            tree_layout = QVBoxLayout(browse)
            self.tabs.addTab(browse, "图谱树")
        row = QHBoxLayout()
        self.search = QLineEdit(); self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText(f"搜索中文名、缩写、英文名或 {structure_id()}")
        row.addWidget(self.search, 1)
        self.only = None
        if not custom:
            self.only = QCheckBox("只看会列出的")
            row.addWidget(self.only)
        tree_layout.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["脑区缩写", "中文名", "英文名", structure_id(), "脑区类别", "图谱层级"])
        self.tree.setUniformRowHeights(True)
        tree_layout.addWidget(self.tree, 1)
        bold = QFont(); bold.setBold(True)
        grey = QBrush(QColor("#7d8590"))
        items = {}
        self._by_id = {}
        for r in t.itertuples(index=False):
            rid = int(r.id)
            on = rid in sel if custom else r.acronym in sel
            prefix = "" if custom else ("✔ " if on else "    ")
            it = QTreeWidgetItem([prefix + r.acronym, r.zh, r.name, str(rid), r.脑区类别, f"第 {r.depth} 级"])
            it.setData(0, Qt.UserRole, rid)
            if custom:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            else:
                it.setData(0, Qt.UserRole + 1, on)
                for c in range(it.columnCount()):
                    if on:
                        it.setFont(c, bold)
                    else:
                        it.setForeground(c, grey)
            parent_item = items.get(r.parent)
            if parent_item is None:
                self.tree.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            items[r.acronym] = it
            self._by_id[rid] = it
        if custom:
            unknown = sorted(sel.difference(self._by_id))
            if unknown:
                ids = "、".join(map(str, unknown))
                head.setText(head.text() + f"\n已有选择中的{structure_id()} {ids} 在当前图谱中找不到，应用后将移除。")
            self.tree.expandToDepth(1)
        for it in items.values():
            if (it.checkState(0) == Qt.Checked if custom else it.data(0, Qt.UserRole + 1)):
                p = it.parent()
                while p is not None:
                    p.setExpanded(True); p = p.parent()
        for c, w in enumerate((200, 150, 240, 90, 110, 70)):
            self.tree.setColumnWidth(c, w)
        if custom:
            self._add_selection_tab(L)
        else:
            b = QPushButton("关闭"); b.clicked.connect(self.accept); L.addWidget(b)
        self.search.textChanged.connect(self._filter)
        if self.only is not None:
            self.only.toggled.connect(self._filter)

    def selected_ids(self):
        return [rid for rid, it in self._by_id.items() if it.checkState(0) == Qt.Checked]

    def _add_selection_tab(self, layout):
        page = QWidget()
        column = QVBoxLayout(page)
        self.selection = QListWidget()
        self.selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        column.addWidget(self.selection, 1)
        row = QHBoxLayout()
        self.remove_button = QPushButton("移除选中项")
        self.remove_button.clicked.connect(lambda: self._uncheck_items(self.selection.selectedItems()))
        self.selection.itemSelectionChanged.connect(lambda: self.remove_button.setEnabled(bool(self.selection.selectedItems())))
        row.addWidget(self.remove_button)
        self.clear_button = QPushButton("清空已选")
        self.clear_button.clicked.connect(lambda: self._uncheck_items([self.selection.item(i) for i in range(self.selection.count())]))
        row.addWidget(self.clear_button); row.addStretch(1)
        column.addLayout(row)
        self.tabs.addTab(page, "已选")
        actions = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.apply_button = actions.button(QDialogButtonBox.Ok)
        self.apply_button.setText("应用")
        actions.button(QDialogButtonBox.Cancel).setText("取消")
        actions.accepted.connect(self.accept); actions.rejected.connect(self.reject)
        layout.addWidget(actions)
        self.tree.itemChanged.connect(self._update_selection)
        self._update_selection()

    def _update_selection(self):
        self.selection.clear()
        for rid in self.selected_ids():
            it = self._by_id[rid]
            label = " · ".join(text for text in (it.text(0), it.text(1), it.text(2), str(rid)) if text)
            entry = QListWidgetItem(label, self.selection)
            entry.setData(Qt.UserRole, rid)
            entry.setToolTip(label)
        count = self.selection.count()
        self.tabs.setTabText(1, f"已选（{count}）")
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(bool(count))
        self.apply_button.setEnabled(bool(count))
        self.apply_button.setToolTip("" if count else "请先在图谱树中勾选至少一个脑区。")

    def _uncheck_items(self, entries):
        self.tree.blockSignals(True)
        for entry in entries:
            self._by_id[entry.data(Qt.UserRole)].setCheckState(0, Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_selection()

    def _filter(self, *_):
        key = self.search.text().strip().lower()
        only = self.only is not None and self.only.isChecked()

        def visit(it) -> bool:
            kids = [visit(it.child(i)) for i in range(it.childCount())]
            matches = (it.text(3) == key if key.isdecimal() else any(key in it.text(c).lower() for c in range(3)))
            hit = (not key or matches) and (not only or it.data(0, Qt.UserRole + 1))
            show = hit or any(kids)
            it.setHidden(not show)
            if (key or only) and any(kids):
                it.setExpanded(True)
            return show
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
