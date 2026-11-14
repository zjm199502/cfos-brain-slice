"""自动质量检查的界面部分：脑片表格里的“质量”列、检查按钮和结果窗口。"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView, QDialog, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

LEVEL_COLORS = {"可疑": QColor(248, 203, 173), "注意": QColor(255, 242, 204), "正常": QColor(226, 239, 218)}
QUALITY_HELP = ("自动检查每张脑片：配准是否贴合、有没有模糊（没对上焦）、有没有过曝（亮度饱和）、"
                "组织有没有撕裂或缺损，以及 DAPI 细胞密度是否异常。\n"
                "结果分为 正常 / 注意 / 可疑；“可疑”的脑片建议优先人工查看（导出的 Excel 中也会标红）。"
                "第一次检查每张脑片约需 5–10 秒，之后有缓存。")


class QualityMixin:
    def _quality_widgets(self, layout):
        b = QPushButton("检查所有脑片的质量（模糊、过曝、缺损、配准）")
        b.setToolTip(QUALITY_HELP)
        b.clicked.connect(self.run_quality)
        layout.addWidget(b)

    def _quality_cell(self, im, s):
        from ..core.quality import cached_levels
        if getattr(self, "_q_levels", None) is None:
            self._q_levels = cached_levels(self.project)
        lvl, why = self._q_levels.get((im.key, s.name), ("—", "还没有检查，或检查后改动过"))
        it = QTableWidgetItem(lvl)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setToolTip(why or "没有发现问题")
        if lvl in LEVEL_COLORS:
            it.setBackground(LEVEL_COLORS[lvl])
            it.setForeground(QColor(0, 0, 0))
        return it

    def run_quality(self):
        if self.project is None or not self.project.images:
            return
        project = self.project
        from ..core.quality import quality_table

        def done(df):
            self._q_levels = None
            self._refresh_section_table()
            self._show_quality(df)

        self.run_task(lambda task: quality_table(project, progress=task.report), done, label="正在检查脑片质量…")

    def _show_quality(self, df):
        if df is None or df.empty:
            self.info("没有可检查的脑片。")
            return
        df = df.sort_values(["质量分", "图片", "脑片"]).reset_index(drop=True)
        n_bad = int((df["质量"] == "可疑").sum()); n_warn = int((df["质量"] == "注意").sum())
        dlg = QDialog(self); dlg.setWindowTitle("脑片质量检查"); dlg.resize(820, 520)
        L = QVBoxLayout(dlg)
        lb = QLabel(f"共 {len(df)} 张脑片：可疑 {n_bad} 张，注意 {n_warn} 张，正常 {len(df) - n_bad - n_warn} 张。"
                    "按质量分从低到高排列；双击一行可以跳到这张脑片。\n"
                    "处理建议：配准不准、阳性细胞落在白质或脑室里 → 到“⑥ 脑图谱配准”调整；模糊、过曝 → 考虑重新扫描或不分析这张；"
                    "组织缺损 → 在“② 脑片拆分”画排除区域，或取消“分析”勾选。")
        lb.setWordWrap(True); L.addWidget(lb)
        cols = ["图片", "脑片", "质量", "质量分", "需要检查的问题"]
        t = QTableWidget(len(df), len(cols)); t.setHorizontalHeaderLabels(cols)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers); t.setSelectionBehavior(QAbstractItemView.SelectRows)
        h = t.horizontalHeader()
        for j in range(len(cols) - 1):
            h.setSectionResizeMode(j, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(cols) - 1, QHeaderView.Stretch)
        for i, r in df.iterrows():
            for j, c in enumerate(cols):
                it = QTableWidgetItem(str(r[c]) if r[c] == r[c] else "")
                if r["质量"] in LEVEL_COLORS:
                    it.setBackground(LEVEL_COLORS[r["质量"]]); it.setForeground(QColor(0, 0, 0))
                if c == "需要检查的问题":
                    detail = [f"{k}：{r[k]}" for k in df.columns[5:] if r[k] == r[k] and r[k] is not None]
                    it.setToolTip("\n".join(detail))
                t.setItem(i, j, it)
        t.cellDoubleClicked.connect(lambda i, _c: self._goto_section(df.at[i, "图片"], df.at[i, "脑片"]))
        L.addWidget(t, 1)
        b = QPushButton("关闭"); b.clicked.connect(dlg.accept); L.addWidget(b)
        dlg.show()
        self._q_dialog = dlg

    def _goto_section(self, key, name):
        keys = [im.key for im in self.project.images]
        if key not in keys:
            return
        i = keys.index(key)
        if self.image is not self.project.images[i]:
            self.image_combo.setCurrentIndex(i)
        names = [s.name for s in self.image.sections]
        if name in names:
            self._zoom_section(names.index(name), 0)
