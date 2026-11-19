"""“脑区范围”选择框：导出和组间统计共用（v0.8.5）。

用几种看得懂的选法（12 个大脑区 / 约 300 个常用脑区 / 全部…）代替“第几级”，
下面直接写明这种选法包含什么，并可以打开一个树形窗口查看具体会列出哪些脑区。"""
from __future__ import annotations

from .layout import fit_window
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import region_sets as RS


class RegionScopeBox(QWidget):
    changed = Signal()

    def __init__(self, with_layers: bool = True, parent=None):
        super().__init__(parent)
        L = QVBoxLayout(self); L.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for m, label, tip in RS.MODES:
            self.combo.addItem(label, m)
            self.combo.setItemData(self.combo.count() - 1, tip, Qt.ToolTipRole)
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
        self.combo.currentIndexChanged.connect(self._sync)
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

    def set_values(self, mode: str, depth: int = 0, skip_layers: bool = False):
        mode = RS.resolve_mode(mode, depth)
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
    def _sync(self, *_):
        m = self.mode()
        by_depth = m == RS.DEPTH
        self.depth_label.setVisible(by_depth); self.depth.setVisible(by_depth)
        self.layers.setEnabled(m in (RS.ALL, RS.DEPTH))    # 大脑区 / 常用脑区本来就不含皮层分层
        try:
            n = len(RS.selected(m, self.max_depth(), self.skip_layers()))
            count = f"共 {n} 个脑区。"
        except Exception:  # noqa: BLE001  图谱文件缺失时不影响界面
            count = ""
        self.desc.setText(f"{count}{RS.MODE_HELP[m]}")
        self.changed.emit()

    def show_preview(self):
        RegionPreviewDialog(self.mode(), self.max_depth(), self.skip_layers(), self.window()).exec()


class RegionPreviewDialog(QDialog):
    """按图谱的树形结构显示全部脑区：会列出的打 ✔ 加粗，不列出的变灰。"""

    def __init__(self, mode, depth=0, skip_layers=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("会列出哪些脑区")
        fit_window(self, 900, 660)
        L = QVBoxLayout(self)
        t = RS.info()
        sel = RS.selected(mode, depth, skip_layers)
        head = QLabel(f"当前选法：{RS.MODE_LABEL[RS.resolve_mode(mode, depth)]}\n"
                      f"打 ✔ 的 {len(sel)} 个脑区会出现在结果里，灰色的不列出（它们的细胞不会丢，已经计入所属的上级或常用脑区）。")
        head.setWordWrap(True); L.addWidget(head)
        row = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText("搜索缩写、英文名或中文名，例如 BLA、amygdala、杏仁核")
        self.only = QCheckBox("只看会列出的"); row.addWidget(self.search, 1); row.addWidget(self.only)
        L.addLayout(row)
        self.tree = QTreeWidget(); self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["脑区缩写", "中文名", "英文名", "脑区类别", "图谱层级"])
        L.addWidget(self.tree, 1)
        bold = QFont(); bold.setBold(True)
        grey = QBrush(QColor("#7d8590"))
        items = {}
        for r in t.itertuples(index=False):
            on = r.acronym in sel
            it = QTreeWidgetItem([("✔ " if on else "    ") + r.acronym, r.zh, r.name, r.脑区类别, f"第 {r.depth} 级"])
            it.setData(0, Qt.UserRole, on)
            for c in range(5):
                if on:
                    it.setFont(c, bold)
                else:
                    it.setForeground(c, grey)
            parent = items.get(r.parent)
            (parent.addChild(it) if parent is not None else self.tree.addTopLevelItem(it))
            items[r.acronym] = it
        self._items = items
        # 展开到能看见所有会列出的脑区
        for it in items.values():
            if it.data(0, Qt.UserRole):
                p = it.parent()
                while p is not None:
                    p.setExpanded(True); p = p.parent()
        for c, w in enumerate((210, 180, 280, 100, 70)):
            self.tree.setColumnWidth(c, w)
        b = QPushButton("关闭"); b.clicked.connect(self.accept); L.addWidget(b)
        self.search.textChanged.connect(self._filter); self.only.toggled.connect(self._filter)

    def _filter(self, *_):
        key = self.search.text().strip().lower()
        only = self.only.isChecked()

        def visit(it) -> bool:
            kids = [visit(it.child(i)) for i in range(it.childCount())]
            hit = (not key or any(key in it.text(c).lower() for c in range(3))) and (not only or it.data(0, Qt.UserRole))
            show = hit or any(kids)
            it.setHidden(not show)
            if (key or only) and any(kids):
                it.setExpanded(True)
            return show
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
