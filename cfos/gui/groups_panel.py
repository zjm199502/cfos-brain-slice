"""⑦ 分组与统计：给每张脑片填写动物编号和组别，按动物汇总后做组间比较并画图。"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.stats import StatsParams, metric_options, run_group_stats


def big_button_local(text):
    from .main_window import big_button
    return big_button(text)


class GroupsMixin:
    def _panel_groups(self):
        from .main_window import hint, big_button
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(hint("为每张脑片填写“动物编号”和“组别”（例如 对照 / 电击）。同一只动物的多张脑片会先合并"
                         "（细胞数相加、面积相加），再按动物做组间比较，避免把脑片当作独立样本。\n"
                         "可以在表格里直接双击修改；也可以选中多行后点“把选中的设为…”，或导出到 Excel 填好再导入。"))
        self.st_test_tip = "Welch t 检验：常用，允许两组方差不同。\nMann-Whitney U：非参数，动物数少或数据明显偏态时用。"
        self.grp_table = QTableWidget(0, 4)
        self.grp_table.setHorizontalHeaderLabels(["图片", "脑片", "动物编号", "组别"])
        self.grp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grp_table.itemChanged.connect(self._grp_item_changed)
        L.addWidget(self.grp_table, 1)
        r1 = QHBoxLayout()
        b = QPushButton("选中的设为同一组…"); b.clicked.connect(lambda: self._grp_set_selected("group")); r1.addWidget(b)
        b = QPushButton("选中的设为同一动物…"); b.clicked.connect(lambda: self._grp_set_selected("animal")); r1.addWidget(b)
        L.addLayout(r1)
        r2 = QHBoxLayout()
        b = QPushButton("动物编号=图片名"); b.clicked.connect(self._grp_guess); r2.addWidget(b)
        b.setToolTip("还没填动物编号的脑片，用图片名作为动物编号（一张玻片 = 一只动物时适用）。")
        b = QPushButton("导出分组表…"); b.clicked.connect(self._grp_export); r2.addWidget(b)
        b = QPushButton("导入分组表…"); b.clicked.connect(self._grp_import); r2.addWidget(b)
        L.addLayout(r2)

        box = QGroupBox("组间比较"); f = QFormLayout(box)
        self.st_metric = QComboBox()
        self.st_test = QComboBox()
        self.st_test.addItem("Welch t 检验", "welch")
        self.st_test.addItem("Mann-Whitney U（非参数）", "mannwhitney")
        self.st_test.setToolTip(self.st_test_tip)
        self.st_ctrl = QComboBox()
        self.st_hemi = QComboBox(); self.st_hemi.addItems(["左右合计", "左", "右"])
        self.st_area = QDoubleSpinBox(); self.st_area.setRange(0, 10); self.st_area.setDecimals(3)
        self.st_area.setSingleStep(0.01); self.st_area.setSuffix(" mm²")
        self.st_area.setToolTip("某只动物在某个脑区的总面积小于这个值时，这只动物不纳入该脑区的比较（面积太小密度不稳定）。")
        self.st_depth = QDoubleSpinBox(); self.st_depth.setDecimals(0); self.st_depth.setRange(0, 12)
        self.st_depth.setSpecialValueText("全部脑区"); self.st_depth.setPrefix("第 "); self.st_depth.setSuffix(" 级以内")
        self.st_depth.setToolTip("脑区越细，比较次数越多，多重比较校正越严格。只关心大脑区时可以设 6–8。")
        f.addRow("比较的指标", self.st_metric); f.addRow("统计方法", self.st_test); f.addRow("对照组", self.st_ctrl)
        f.addRow("半脑", self.st_hemi); f.addRow("最小面积", self.st_area); f.addRow("脑区范围", self.st_depth)
        L.addWidget(box)
        b = big_button("计算组间统计并画图"); b.clicked.connect(lambda: self.run_stats()); L.addWidget(b)
        self.st_summary = QLabel(""); self.st_summary.setWordWrap(True); L.addWidget(self.st_summary)
        self.st_preview = QLabel(); self.st_preview.setAlignment(Qt.AlignCenter); L.addWidget(self.st_preview)
        self.st_open = QPushButton("打开统计结果文件夹"); self.st_open.hide(); L.addWidget(self.st_open)
        self.st_open.clicked.connect(lambda: self._open_folder(self._st_out))
        self._network_box(L)
        return w

    # ------------------------------------------------------------ 功能网络
    def _network_box(self, L):
        from qtpy.QtWidgets import QCheckBox, QSpinBox
        box = QGroupBox("脑区功能网络分析（共激活网络）"); f = QFormLayout(box)
        lb = QLabel("在每个组内，以动物为样本计算脑区之间 c-Fos 的相关，强相关的脑区连成网络，比较两组网络的结构"
                    "（枢纽脑区、模块、网络密度等）。指标、半脑、最小面积沿用上面“组间比较”的设置。\n"
                    "需要每组至少 4 只动物，建议 ≥ 8 只；动物少时结果不稳定，请重点看跨阈值一致、留一法稳定的结论。")
        lb.setWordWrap(True); f.addRow(lb)
        self.net_depth = QSpinBox(); self.net_depth.setRange(4, 10); self.net_depth.setValue(7)
        self.net_depth.setToolTip("数字越大脑区分得越细（节点越多）。皮层分层不作为单独节点。\n"
                                  "一般 6–8；节点数最好不要远多于动物数的 10 倍。")
        self.net_method = QComboBox()
        self.net_method.addItem("Pearson 相关", "pearson"); self.net_method.addItem("Spearman 秩相关（对离群值更稳）", "spearman")
        self.net_thr = QDoubleSpinBox(); self.net_thr.setRange(0.3, 0.95); self.net_thr.setSingleStep(0.05); self.net_thr.setValue(0.7)
        self.net_thr.setToolTip("画网络图和判断枢纽脑区用的阈值。另外还会在 0.5–0.9 的多个阈值下计算，看结论是否一致。")
        self.net_pos = QCheckBox("只用正相关建网（常用做法）"); self.net_pos.setChecked(True)
        self.net_perm = QSpinBox(); self.net_perm.setRange(100, 10000); self.net_perm.setSingleStep(500); self.net_perm.setValue(1000)
        self.net_perm.setToolTip("组间比较的置换次数。越多 p 值越精确，也越慢（1000 次一般 1 分钟以内）。")
        f.addRow("脑区粒度：第几级", self.net_depth); f.addRow("相关方法", self.net_method)
        f.addRow("主阈值 r ≥", self.net_thr); f.addRow(self.net_pos); f.addRow("置换次数", self.net_perm)
        L.addWidget(box)
        b = big_button_local("计算功能网络并画图"); b.clicked.connect(lambda: self.run_network()); L.addWidget(b)
        self.net_summary = QLabel(""); self.net_summary.setWordWrap(True); L.addWidget(self.net_summary)
        self.net_preview = QLabel(); self.net_preview.setAlignment(Qt.AlignCenter); L.addWidget(self.net_preview)
        self.net_open = QPushButton("打开网络分析结果文件夹"); self.net_open.hide(); L.addWidget(self.net_open)
        self.net_open.clicked.connect(lambda: self._open_folder(self._net_out))

    def _sync_network_box(self):
        d = self.project.network or {}
        self.net_depth.setValue(int(d.get("depth", 7)))
        self.net_method.setCurrentIndex(max(0, self.net_method.findData(d.get("method", "pearson"))))
        self.net_thr.setValue(float(d.get("main_threshold", 0.7)))
        self.net_pos.setChecked(bool(d.get("positive_only", True)))
        self.net_perm.setValue(int(d.get("n_perm", 1000)))

    def run_network(self, region_sec=None):
        if not isinstance(region_sec, pd.DataFrame):  # 按钮信号会传入 checked=False
            region_sec = None
        from ..core.network import NetParams, run_network_analysis
        sp = self._stats_params()
        np_ = NetParams(metric=sp.metric, hemisphere=sp.hemisphere, min_area_mm2=sp.min_area_mm2, control=sp.control,
                        depth=self.net_depth.value(), method=self.net_method.currentData(),
                        main_threshold=round(self.net_thr.value(), 2), positive_only=self.net_pos.isChecked(),
                        n_perm=self.net_perm.value())
        self.project.network = np_.to_dict(); self.project.save()
        project = self.project
        out = project.folder / f"功能网络_{_dt.datetime.now():%Y%m%d_%H%M}"

        def done(res):
            self._net_out, summary, figs = res
            self.net_summary.setText(f"结果已保存到“{Path(self._net_out).name}”文件夹（网络分析.xlsx 和图片）。\n{summary}")
            self.net_open.show()
            if figs:
                self.net_preview.setPixmap(QPixmap(str(figs[0])).scaledToWidth(360, Qt.SmoothTransformation))

        self.run_task(lambda task: run_network_analysis(project, out, np_, region_sec=region_sec, progress=task.report),
                      done, label="正在计算功能网络…")

    # ------------------------------------------------------------ 表格
    def _sync_groups_panel(self):
        p = self.project
        if p is None:
            return
        rows = p.sample_rows()
        t = self.grp_table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, col in enumerate(["图片", "脑片", "动物", "组别"]):
                it = QTableWidgetItem(r[col])
                if j < 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setData(Qt.UserRole, r["key"])
                t.setItem(i, j, it)
        t.blockSignals(False)
        # 统计参数
        sp = StatsParams(**{k: v for k, v in p.stats.items() if k in StatsParams.__dataclass_fields__})
        self.st_metric.blockSignals(True); self.st_metric.clear()
        self.st_metric.addItems(metric_options(p.channel_names, grading=p.grading))
        if sp.metric:
            self.st_metric.setCurrentText(sp.metric)
        self.st_metric.blockSignals(False)
        self.st_test.setCurrentIndex(max(0, self.st_test.findData(sp.test)))
        groups = sorted({r["组别"] for r in rows if r["组别"]})
        self.st_ctrl.clear(); self.st_ctrl.addItems(groups)
        if sp.control in groups:
            self.st_ctrl.setCurrentText(sp.control)
        self.st_hemi.setCurrentText(sp.hemisphere)
        self.st_area.setValue(sp.min_area_mm2); self.st_depth.setValue(sp.max_depth)
        self._sync_network_box()

    def _grp_item_changed(self, item):
        key = item.data(Qt.UserRole)
        if item.column() == 2:
            self.project.set_sample(key, animal=item.text())
        elif item.column() == 3:
            self.project.set_sample(key, group=item.text())
            self._refresh_ctrl_groups()
        self._save_later()

    def _refresh_ctrl_groups(self):
        cur = self.st_ctrl.currentText()
        groups = sorted({r["组别"] for r in self.project.sample_rows() if r["组别"]})
        self.st_ctrl.clear(); self.st_ctrl.addItems(groups)
        if cur in groups:
            self.st_ctrl.setCurrentText(cur)

    def _grp_set_selected(self, what):
        rows = sorted({i.row() for i in self.grp_table.selectedIndexes()})
        if not rows:
            self.info("请先在表格中选中要修改的脑片（按住 Ctrl 或 Shift 可以多选）。")
            return
        label = "组别" if what == "group" else "动物编号"
        text, ok = QInputDialog.getText(self, "批量设置", f"把选中的 {len(rows)} 张脑片的{label}设为：")
        if not ok:
            return
        for r in rows:
            key = self.grp_table.item(r, 0).data(Qt.UserRole)
            self.project.set_sample(key, **{what: text})
        self.project.save(); self._sync_groups_panel()

    def _grp_guess(self):
        n = self.project.guess_animals(); self.project.save(); self._sync_groups_panel()
        self.statusBar().showMessage(f"已为 {n} 张脑片填写动物编号（= 图片名）。一张玻片上有多只动物时请手动修改。", 10000)

    def _grp_export(self):
        f, _ = QFileDialog.getSaveFileName(self, "导出分组表", str(self.project.folder / "分组表.xlsx"), "Excel 文件 (*.xlsx)")
        if not f:
            return
        df = pd.DataFrame(self.project.sample_rows()).drop(columns=["key"]).rename(columns={"动物": "动物编号"})
        try:
            df.to_excel(f, index=False)
        except PermissionError:
            self.error("文件正在被 Excel 打开，请先关闭它。")
            return
        self.info(f"已导出：{f}\n在 Excel 中填写“动物编号”和“组别”两列，保存后点“导入分组表”。不要修改“图片”和“脑片”两列。")

    def _grp_import(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入分组表", str(self.project.folder), "表格 (*.xlsx *.xls *.csv)")
        if not f:
            return
        try:
            df = pd.read_csv(f, dtype=str) if f.lower().endswith(".csv") else pd.read_excel(f, dtype=str)
        except Exception as e:  # noqa: BLE001
            self.error(f"文件无法读取：{e}")
            return
        need = {"图片", "脑片"}
        if not need <= set(df.columns):
            self.error("分组表必须包含“图片”“脑片”两列（建议先用“导出分组表”生成）。")
            return
        df = df.fillna("")
        keys = {r["key"] for r in self.project.sample_rows()}
        n, miss = 0, []
        for _, r in df.iterrows():
            k = f"{r['图片']}/{r['脑片']}"
            if k not in keys:
                miss.append(k); continue
            self.project.set_sample(k, animal=str(r.get("动物编号", r.get("动物", ""))), group=str(r.get("组别", "")))
            n += 1
        self.project.save(); self._sync_groups_panel()
        msg = f"已导入 {n} 张脑片的分组。"
        if miss:
            msg += f"\n以下 {len(miss)} 行在项目中找不到，已忽略：\n" + "\n".join(miss[:10])
        self.info(msg)

    # ------------------------------------------------------------ 统计
    def _stats_params(self) -> StatsParams:
        sp = StatsParams(metric=self.st_metric.currentText(), test=self.st_test.currentData(),
                         control=self.st_ctrl.currentText(), hemisphere=self.st_hemi.currentText(),
                         min_area_mm2=self.st_area.value(), max_depth=int(self.st_depth.value()))
        self.project.stats = sp.to_dict(); self.project.save()
        return sp

    def run_stats(self, region_sec=None):
        if not isinstance(region_sec, pd.DataFrame):  # 按钮信号会传入 checked=False
            region_sec = None
        sp = self._stats_params()
        project = self.project
        out = project.folder / f"组间统计_{_dt.datetime.now():%Y%m%d_%H%M}"

        def done(res):
            self._st_out, summary, figs = res
            self.st_summary.setText(f"结果已保存到“{Path(self._st_out).name}”文件夹（组间统计.xlsx 和图片）。\n{summary}")
            self.st_open.show()
            if figs:
                pm = QPixmap(str(figs[0]))
                self.st_preview.setPixmap(pm.scaledToWidth(360, Qt.SmoothTransformation))

        self.run_task(lambda task: run_group_stats(project, out, sp, region_sec=region_sec, progress=task.report),
                      done, label="正在计算组间统计…")
