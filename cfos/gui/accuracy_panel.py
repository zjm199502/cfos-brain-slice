"""④ 细胞识别 里的“检查识别准确率”：人工点数与软件识别对照。"""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from ..core import accuracy as A

BOX_LAYER = "准确率检查框"
PTS_LAYER = "人工计数"
RES_LAYER = "准确率结果"


class AccuracyMixin:
    def _accuracy_widgets(self, layout):
        g = QGroupBox("检查识别准确率（人工点数对照）"); gl = QVBoxLayout(g)
        lb = QLabel("用来回答“软件数得准不准”：①选通道，点“放一个检查框”（放在当前视野中心）；"
                    "②放大，在框里把这个通道的每个真实细胞都点一下（只点中心在框内的）；"
                    "③点“计算准确率”。绿色 = 双方都找到，红色 = 软件漏掉，黄色 = 软件多找。"
                    "建议在不同脑区、不同亮度处各查 3–5 个框，结果会汇总写进 Excel 的“分析参数”。")
        lb.setWordWrap(True); gl.addWidget(lb)
        r = QHBoxLayout(); r.addWidget(QLabel("通道："))
        self.acc_ch = QComboBox(); r.addWidget(self.acc_ch, 1)
        r.addWidget(QLabel("框边长"))
        self.acc_size = QDoubleSpinBox(); self.acc_size.setRange(50, 2000); self.acc_size.setValue(250)
        self.acc_size.setDecimals(0); self.acc_size.setSuffix(" µm"); r.addWidget(self.acc_size)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("放一个检查框"); b.clicked.connect(self.acc_new_box); r.addWidget(b)
        b = QPushButton("计算准确率"); b.clicked.connect(self.acc_compute); r.addWidget(b)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("查看所有检查记录"); b.clicked.connect(self.acc_show_all); r.addWidget(b)
        b = QPushButton("删除最后一条记录"); b.clicked.connect(self.acc_delete_last); r.addWidget(b)
        gl.addLayout(r)
        self.acc_info = QLabel(""); self.acc_info.setWordWrap(True); gl.addWidget(self.acc_info)
        layout.addWidget(g)

    def _acc_sync(self):
        if self.project is None:
            return
        cur = self.acc_ch.currentText()
        self.acc_ch.blockSignals(True); self.acc_ch.clear(); self.acc_ch.addItems(self.project.channel_names)
        if cur:
            self.acc_ch.setCurrentText(cur)
        else:
            nm = [n for n in self.project.channel_names if n != "DAPI"]
            if nm:
                self.acc_ch.setCurrentText(nm[0])
        self.acc_ch.blockSignals(False)
        self._acc_summary_text()

    def _acc_summary_text(self):
        s = A.summary(self.project) if self.project else None
        if s is None or s.empty:
            self.acc_info.setText("还没有检查记录。")
            return
        lines = [f"{r['通道']}：{int(r['检查方框数'])} 个框，人工 {r['人工计数']} / 软件 {r['软件识别']}，"
                 f"查准率 {r['查准率']:.0%}，查全率 {r['查全率']:.0%}，F1 {r['F1']:.2f}" for _, r in s.iterrows()]
        self.acc_info.setText("累计：\n" + "\n".join(lines))

    def acc_new_box(self):
        if self.slide is None:
            return
        if self.cells_for(self.image) is None:
            self.info("这张图片还没有识别细胞，请先识别。")
            return
        for nm in (BOX_LAYER, PTS_LAYER, RES_LAYER):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        px = self.slide.pixel_size_um
        side = self.acc_size.value() / px
        cy, cx = self.viewer.camera.center[-2], self.viewer.camera.center[-1]
        y0, x0 = cy - side / 2, cx - side / 2
        rect = np.array([[y0, x0], [y0, x0 + side], [y0 + side, x0 + side], [y0 + side, x0]], float)
        self.viewer.add_shapes([rect], shape_type="rectangle", name=BOX_LAYER, edge_color="#2ee6d6",
                               face_color="transparent", edge_width=max(2.0, 3 / px))
        layer = self.viewer.add_points(np.zeros((0, 2)), name=PTS_LAYER, size=8 / px, face_color="#ff9f0a",
                                       border_color="white", border_width=0.1)
        # 软件识别的细胞先隐藏，免得影响人工判断
        for l in self.viewer.layers:
            if l.name.startswith("细胞-"):
                l.visible = False
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self.viewer.camera.zoom = 0.8 * min(self.qt_viewer.height(), self.qt_viewer.width()) / side
        self.statusBar().showMessage(f"在青色框里点出每个 {self.acc_ch.currentText()} 细胞（软件的结果已暂时隐藏）。"
                                     "点错了：切到“选择 / 拖动点”，选中后按删除。点完按“计算准确率”。", 20000)

    def _acc_box0(self):
        if BOX_LAYER not in self.viewer.layers or not len(self.viewer.layers[BOX_LAYER].data):
            return None
        r = self.viewer.layers[BOX_LAYER].data[0]
        ys, xs = r[:, -2], r[:, -1]
        return float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())

    def acc_compute(self):
        box = self._acc_box0()
        if box is None or PTS_LAYER not in self.viewer.layers:
            self.info("请先点“放一个检查框”，并在框里点出细胞。")
            return
        pts = np.asarray(self.viewer.layers[PTS_LAYER].data)
        if len(pts) == 0:
            self.info("还没有在框里点细胞。")
            return
        c = self.acc_ch.currentIndex()
        df = self.cells_for(self.image)
        res = A.check(df, c, box, pts[:, ::-1][:, :2] if pts.shape[1] == 2 else pts[:, [-1, -2]],
                      self.slide.pixel_size_um)
        from ..core.manual import section_of
        sec = section_of(self.image, box[0] + box[2] / 2, box[1] + box[3] / 2)
        A.record(self.project, self.image, sec, c, box, res, self.project.detection.channels[c].model
                 if c in self.project.detection.channels else "")
        self.project.save()
        self._acc_show_result(res)
        self._acc_summary_text()
        p, r, f = res["precision"], res["recall"], res["f1"]
        self.info(f"这个框：人工 {res['n_manual']} 个，软件 {res['n_auto']} 个，配对成功 {res['tp']} 个；\n"
                  f"漏检 {res['fn']} 个（红），误检 {res['fp']} 个（黄）。\n"
                  f"查准率 {p:.0%}（软件找到的里有多少是真的）\n查全率 {r:.0%}（真的细胞里软件找到多少）\nF1 {f:.2f}\n\n"
                  "一般 F1 ≥ 0.85 算好；查全率低 → 降低置信度阈值或用自己的标注微调模型；"
                  "查准率低 → 提高阈值或调高面积下限 / 对比度下限。")

    def _acc_show_result(self, res):
        if RES_LAYER in self.viewer.layers:
            self.viewer.layers.remove(RES_LAYER)
        px = self.slide.pixel_size_um
        miss = np.asarray(res["missed_xy"], float).reshape(-1, 2)
        false = np.asarray(res["false_xy"], float).reshape(-1, 2)
        man = np.asarray(res["manual_xy"], float).reshape(-1, 2)
        missed_set = {tuple(np.round(m, 3)) for m in miss}
        ok = np.array([m for m in man if tuple(np.round(m, 3)) not in missed_set]).reshape(-1, 2)
        pts = np.vstack([ok, miss, false])
        cols = ["#30d158"] * len(ok) + ["#ff3b30"] * len(miss) + ["#ffd60a"] * len(false)
        if len(pts):
            self.viewer.add_points(pts[:, ::-1], name=RES_LAYER, size=14 / px, face_color="transparent",
                                   border_color=cols, border_width=0.2)
        if PTS_LAYER in self.viewer.layers:
            self.viewer.layers[PTS_LAYER].visible = False

    def acc_show_all(self):
        import pandas as pd
        from qtpy.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
        if not self.project or not self.project.accuracy:
            self.info("还没有检查记录。")
            return
        df = pd.DataFrame(self.project.accuracy).drop(columns=["人工点", "方框"], errors="ignore")
        dlg = QDialog(self); dlg.setWindowTitle("识别准确率检查记录"); dlg.resize(900, 400)
        L = QVBoxLayout(dlg)
        t = QTableWidget(len(df), len(df.columns)); t.setHorizontalHeaderLabels(list(df.columns))
        for i, (_, r) in enumerate(df.iterrows()):
            for j, c in enumerate(df.columns):
                v = r[c]
                t.setItem(i, j, QTableWidgetItem(f"{v:.1%}" if c in ("查准率", "查全率") and v == v and v is not None
                                                 else ("" if v is None else str(v))))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        L.addWidget(t, 1)
        b = QPushButton("关闭"); b.clicked.connect(dlg.accept); L.addWidget(b)
        dlg.show(); self._acc_dlg = dlg

    def acc_delete_last(self):
        if self.project and self.project.accuracy:
            self.project.accuracy.pop()
            self.project.save()
            self._acc_summary_text()
