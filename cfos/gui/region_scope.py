"""“脑区范围”选择框：导出和组间统计共用（v0.8.5）。

用几种看得懂的选法（12 个大脑区 / 约 300 个常用脑区 / 全部…）代替“第几级”，
下面直接写明这种选法包含什么，并可以打开一个树形窗口查看具体会列出哪些脑区。"""
from __future__ import annotations

from .layout import fit_window
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                            QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
                            QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import region_sets as RS


class RegionScopeBox(QWidget):
    changed = Signal()

    def __init__(self, with_layers: bool = True, parent=None, *, allow_custom: bool = False):
        super().__init__(parent)
        self._region_ids = []
        self._current_mode = RS.ALL
        L = QVBoxLayout(self); L.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for m, label, tip in RS.MODES:
            self.combo.addItem(label, m)
            self.combo.setItemData(self.combo.count() - 1, tip, Qt.ToolTipRole)
        if allow_custom:
            self.combo.addItem("自选脑区…", RS.CUSTOM)
        L.addWidget(self.combo)
        row = QHBoxLayout()
        self.depth_label = QLabel("最多列到图谱的")
        self.depth = QSpinBox(); self.depth.setRange(1, 10); self.depth.setValue(6)
        self.depth.setPrefix("第 "); self.depth.setSuffix(" 级")
        self.depth.setToolTip("第 2 级：大脑 / 脑干 / 小脑；第 5 级：新皮层、海马结构及部分核团；\n"
                              "第 6 级：运动皮层 MO、躯体感觉皮层 SS 等；第 7–8 级：更细的亚区和皮层分层。")
        row.addWidget(self.depth_label); row.addWidget(self.depth); row.addStretch(1)
        L.addLayout(row)
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
        self.depth.valueChanged.connect(self._sync)
        self.layers.toggled.connect(self._sync)
        self._sync()

    # ---- 读 / 写
    def mode(self) -> str:
        return self.combo.currentData()

    def max_depth(self) -> int:
        return int(self.depth.value()) if self.mode() == RS.DEPTH else 0

    def skip_layers(self) -> bool:
        return self._with_layers and self.layers.isEnabled() and self.layers.isChecked()

    def region_ids(self) -> list[int]:
        return self._region_ids.copy()

    def set_values(self, mode: str, depth: int = 0, skip_layers: bool = False, region_ids=()):
        mode = RS.resolve_mode(mode, depth)
        self._region_ids = sorted(set(region_ids or ()))
        for w in (self.combo, self.depth, self.layers):
            w.blockSignals(True)
        self.combo.setCurrentIndex(max(0, self.combo.findData(mode)))
        if depth and depth > 0:
            self.depth.setValue(int(depth))
        self.layers.setChecked(bool(skip_layers))
        for w in (self.combo, self.depth, self.layers):
            w.blockSignals(False)
        self._sync()

    # ---- 界面
    def _mode_changed(self):
        if self.mode() == RS.CUSTOM and not self._edit_custom():
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(self.combo.findData(self._current_mode))
            self.combo.blockSignals(False)
            return
        self._sync()

    def _sync(self, *_):
        m = self.mode()
        self._current_mode = m
        by_depth = m == RS.DEPTH
        self.depth_label.setVisible(by_depth); self.depth.setVisible(by_depth)
        self.layers.setEnabled(m in (RS.ALL, RS.DEPTH))    # 大脑区 / 常用脑区本来就不含皮层分层
        self.btn.setText("编辑已选脑区…" if m == RS.CUSTOM else "查看会列出哪些脑区…")
        if m == RS.CUSTOM:
            n = len(self._region_ids)
            self.desc.setText(f"已选 {n} 个脑区。父区行包含全部下属脑区的统计。"
                              if n else "请至少选择一个脑区，或切换到其他导出范围。")
        else:
            try:
                count = f"共 {len(RS.selected(m, self.max_depth(), self.skip_layers()))} 个脑区。"
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
            RegionPreviewDialog(self.mode(), self.max_depth(), self.skip_layers(), self.window()).exec()


class RegionPreviewDialog(QDialog):
    """预览预设范围或编辑自选范围；父区与子区的勾选相互独立。"""

    def __init__(self, mode, depth=0, skip_layers=False, parent=None, *, region_ids=()):
        super().__init__(parent)
        custom = mode == RS.CUSTOM
        self.setWindowTitle("选择导出的脑区" if custom else "会列出哪些脑区")
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
        self.search.setPlaceholderText("搜索中文名、缩写、英文名或 Allen ID")
        row.addWidget(self.search, 1)
        self.only = None
        if not custom:
            self.only = QCheckBox("只看会列出的")
            row.addWidget(self.only)
        tree_layout.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["脑区缩写", "中文名", "英文名", "Allen ID", "脑区类别", "图谱层级"])
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
                head.setText(head.text() + f"\n已有选择中的 Allen ID {ids} 在当前图谱中找不到，应用后将移除。")
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
