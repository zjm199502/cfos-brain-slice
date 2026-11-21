"""④ 细胞识别 里的“检查识别准确率”：人工点数与软件识别对照。"""
from __future__ import annotations

import numpy as np
from .layout import fit_window
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QMessageBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from ..core import accuracy as A
from ..core.cellpose4 import MODEL_KEY as CELLPOSE_MODEL

BOX_LAYER = "准确率检查框"
PTS_LAYER = "人工计数"
RES_LAYER = "准确率结果"


def _score(value, fmt):
    return format(value, fmt) if np.isfinite(value) else "—"


def _model_name(key):
    if key is None:
        return "模型未记录"
    if key == CELLPOSE_MODEL:
        return "Cellpose 4（cpsam_v2）"
    return key or "StarDist（荧光细胞核）"


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
        self.acc_field = QComboBox()
        self.acc_field.addItems(["常规", "弱阳性", "密集核", "强背景", "阴性组织", "分块接缝"])
        r.addWidget(QLabel("视野类型")); r.addWidget(self.acc_field)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("放一个检查框"); b.clicked.connect(self.acc_new_box); r.addWidget(b)
        b = QPushButton("计算准确率"); b.clicked.connect(self.acc_compute); r.addWidget(b)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("查看所有检查记录"); b.clicked.connect(self.acc_show_all); r.addWidget(b)
        b = QPushButton("删除最后一条记录"); b.clicked.connect(self.acc_delete_last); r.addWidget(b)
        gl.addLayout(r)
        gl.addWidget(QLabel("按脑片盲评：放框并点出阳性中心，再锁定人工点比较两种参数。"))
        row = QHBoxLayout()
        row.addWidget(QLabel("候选局部对比度"))
        self.review_contrast = QDoubleSpinBox()
        self.review_contrast.setRange(0, 10); self.review_contrast.setValue(1.3)
        self.review_contrast.setSingleStep(.1); row.addWidget(self.review_contrast)
        self.review_deconv = QCheckBox("候选去卷积"); row.addWidget(self.review_deconv)
        gl.addLayout(row)
        b = QPushButton("锁定人工点，比较当前与候选参数"); b.clicked.connect(self.compare_review); gl.addWidget(b)
        b = QPushButton("本图盲评记录与脑片参数"); b.clicked.connect(self.review_history); gl.addWidget(b)
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
        lines = [f"{r['通道']} · {r['视野类型']} · {r['模型']} · {r['参数编号']}：{int(r['检查方框数'])} 个框，人工 {r['人工计数']} / 软件 {r['软件识别']}，"
                 f"查准率 {_score(r['查准率'], '.0%')}，查全率 {_score(r['查全率'], '.0%')}，F1 {_score(r['F1'], '.2f')}" for _, r in s.iterrows()]
        self.acc_info.setText("累计：\n" + "\n".join(lines))

    def acc_new_box(self):
        if self.slide is None:
            return
        self._blind_context = (id(self.project), id(self.image), self.acc_ch.currentIndex())
        self._blind_locked = False
        for nm in (BOX_LAYER, PTS_LAYER, RES_LAYER):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        if self._layout_mode == "single":
            self.pane_button.setChecked(True)
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
            if l.name.startswith(("细胞-", "预览", "盲评-", "预处理-")):
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
        c = self.acc_ch.currentIndex()
        df = self.cells_for(self.image)
        if df is None:
            self.info("尚无整图结果。可直接使用下方的盲评比较。"); return
        from ..core.manual import section_of
        from ..core.project import processed_section_names
        sec = section_of(self.image, box[0] + box[2] / 2, box[1] + box[3] / 2)
        if sec not in processed_section_names(self.image, df):
            self.info("检查框所在脑片尚无识别结果。可使用下方的盲评比较。"); return
        models = df.attrs.get("section_models", {}).get(sec)
        if models is not None and c not in models:
            self.info("这张脑片上次识别时未启用此通道。可使用下方的盲评比较。"); return
        model = _model_name(models.get(c) if models is not None else None)
        self._blind_locked = True
        res = A.check(df[df["section"] == sec], c, box, pts[:, [-1, -2]], self.slide.pixel_size_um)
        res["miss_details"] = A.miss_details(self.project, self.image, c, res["missed_xy"], self.slide.pixel_size_um)
        res["field_type"] = self.acc_field.currentText()
        res["field_area_mm2"] = box[2] * box[3] * (self.slide.pixel_size_um / 1000) ** 2
        A.record(self.project, self.image, sec, c, box, res, model, cells=df)
        self.project.save()
        self._acc_show_result(res)
        self._acc_summary_text()
        p, r, f = res["precision"], res["recall"], res["f1"]
        from collections import Counter
        counts = Counter(item["线索"] for item in res["miss_details"])
        clues = "\n".join(f"{key}：{n} 个" for key, n in counts.items())
        self.info(f"这个框：人工 {res['n_manual']} 个，软件 {res['n_auto']} 个，配对成功 {res['tp']} 个；\n"
                  f"漏检 {res['fn']} 个（红），误检 {res['fp']} 个（黄）。\n"
                  f"查准率 {_score(p, '.0%')}，查全率 {_score(r, '.0%')}，F1 {_score(f, '.2f')}\n{clues}")

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
        dlg = QDialog(self); dlg.setWindowTitle("识别准确率检查记录"); fit_window(dlg, 900, 400)
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

    def compare_review(self):
        from dataclasses import asdict
        from ..core.field_review import compare, effective_settings, pooled_reviews, section_key
        from ..core.slide_io import open_slide
        from ..core import unmix
        from matplotlib.path import Path
        from matplotlib.transforms import Bbox
        box = self._acc_box0()
        project, image, c = self.project, self.image, self.acc_ch.currentIndex()
        context = getattr(self, "_blind_context", None)
        if (not box or PTS_LAYER not in self.viewer.layers or not context or
                context[0] != id(project) or context[1] != id(image) or context[2] != c
                or getattr(self, "_blind_locked", False)):
            self.info("请选好通道并重新放一个盲评框。"); return
        if self.task is not None and self.task.isRunning():
            self.info("请等待当前任务完成。"); return
        x, y, w, h = box
        def contains(sec):
            sx, sy, sw, sh = sec.bbox0
            return sec.enabled and sx <= x and sy <= y and x+w <= sx+sw and y+h <= sy+sh
        sections = [sec for sec in image.sections if contains(sec)]
        if len(sections) != 1 or w <= 0 or h <= 0:
            self.info("请先拆分脑片，并把整个检查框放在一张已启用脑片内。"); return
        sec = sections[0]
        rect = Bbox.from_bounds(*box)
        if any(Path(poly).intersects_bbox(rect) for poly in image.exclusions if len(poly)>=3):
            self.info("检查框与排除区域重叠，请移动检查框。"); return
        pts = np.asarray(self.viewer.layers[PTS_LAYER].data)[:, [-1,-2]]
        pts = pts[(pts[:,0]>=x)&(pts[:,0]<x+w)&(pts[:,1]>=y)&(pts[:,1]<y+h)]
        if len(sec.contour0)>2 and not Path(sec.contour0).contains_points(
                [[x,y],[x+w,y],[x,y+h],[x+w,y+h],[x+w/2,y+h/2]]).all():
            self.info("请将检查框移到组织内部，或缩小方框。"); return
        if not project.detection.channels[c].detect:
            self.info("所选通道未启用细胞识别。"); return
        question = ("框内没有人工点。确认视野可判读且没有阳性细胞？" if not len(pts)
                    else f"锁定框内 {len(pts)} 个人工点并揭示对照结果？")
        if QMessageBox.question(self, "脑片盲评", question) != QMessageBox.Yes:
            return
        self._blind_locked = True
        self.viewer.layers[PTS_LAYER].editable = False
        self.viewer.layers[BOX_LAYER].editable = False
        candidate = dict(min_contrast=self.review_contrast.value(), deconv=self.review_deconv.isChecked())
        field_type = self.acc_field.currentText()
        px = self.slide.pixel_size_um
        norm_cache = self.slide.preview_norm_cache
        def job(task):
            slide = open_slide(image.path, z_mode=project.z_mode)
            try:
                unmix.prepare(project, slide, image)
                from copy import deepcopy
                settings = deepcopy(effective_settings(project.detection, image, sec))
                return compare(slide, sec, settings, c, box, pts, candidate, task.is_cancelled, norm_cache), settings
            finally:
                slide.close()
        def done(payload):
            if self.project is not project or self.image is not image or self.acc_ch.currentIndex() != c:
                return
            (results, lost, gained, confirmed), settings = payload
            record = {"脑片":sec.name, "通道":project.channel_names[c], "视野":list(box), "类型":field_type,
                      "人工点":pts.tolist(), "面积mm2":w*h*(px/1000)**2, "当前":asdict(settings.channels[c]),
                      "候选":{**asdict(settings.channels[c]), **candidate},
                      "新增":len(gained), "新增获确认":confirmed, "仅当前":len(lost)}
            lines = [f"{sec.name} · {project.channel_names[c]} · {_model_name(settings.channels[c].model)} · 人工 {len(pts)} 个"]
            for label, (score, frame) in zip(("当前","候选"), results):
                record[label+"计数"] = {k:score[k] for k in ("tp","fp","fn")}
                lines.append(f"{label}：TP {score['tp']}，FP {score['fp']}，FN {score['fn']}，F1 {_score(score['f1'], '.2f')}")
            lines.append(f"新增 {len(gained)} 个，人工确认 {confirmed} 个；仅当前设置检出 {len(lost)} 个。")
            for label, data, ids, color in (("仅当前", results[0][1], lost, "cyan"),("仅候选", results[1][1], gained,"orange")):
                name = "盲评-"+label
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
                polygons = [np.asarray(data.iloc[i].contour)[:,::-1] for i in ids if len(data.iloc[i].contour)>2]
                if polygons:
                    self.viewer.add_shapes(polygons, shape_type="polygon", name=name, edge_color=color,
                                           face_color="transparent", edge_width=max(1.,w/400))
            if not len(pts):
                area = record["面积mm2"]
                lines.append(f"阴性区误检/mm²：当前 {results[0][0]['fp']/area:.1f}，候选 {results[1][0]['fp']/area:.1f}")
            record["采用候选"] = False
            from ..core.detection import raw_signature
            record["识别条件"] = [raw_signature(settings), project.z_mode, float(px)]
            record["位置标识"] = section_key(sec)
            image.field_reviews.append(record)
            lines.append(pooled_reviews(image.field_reviews, record))
            project.save()
            prompt = QMessageBox(self)
            prompt.setWindowTitle("脑片盲评结果"); prompt.setText("\n".join(lines))
            apply_btn = prompt.addButton(f"应用到 {sec.name} 的{project.channel_names[c]}并重新识别", QMessageBox.AcceptRole)
            keep = prompt.addButton("保留当前设置", QMessageBox.RejectRole)
            prompt.setDefaultButton(keep); prompt.exec_()
            if prompt.clickedButton() is apply_btn:
                image.review_settings.setdefault(section_key(sec), {})[str(c)] = dict(candidate)
                record["采用候选"] = True
                image.status = "盲评参数已更新，待重新识别"
                project.save()
                self.run_detection(False, [sec.name])
        self.run_task(job, done, label="正在比较当前与候选参数…")

    def review_history(self):
        from ..core.field_review import section_key
        if self.image is None:
            return
        im = self.image
        dlg = QMessageBox(self); dlg.setWindowTitle("本图盲评记录与脑片参数")
        lines = []
        for sec in im.sections:
            for ch, values in im.review_settings.get(section_key(sec), {}).items():
                lines.append(f"{sec.name} · {self.ch_name(int(ch))}：局部对比度 {values['min_contrast']:g}，"
                             f"去卷积{'开' if values['deconv'] else '关'}")
        dlg.setText("\n".join(lines) or "各脑片使用项目参数。")
        history = []
        for r in im.field_reviews:
            a,b = r["当前计数"], r["候选计数"]
            history.append(f"{r['脑片']} · {r['通道']} · {_model_name(r['当前'].get('model'))} · {r['类型']} · 框 {r['视野']}\n"
                           f"当前 TP/FP/FN {a['tp']}/{a['fp']}/{a['fn']}；候选 {b['tp']}/{b['fp']}/{b['fn']}；"
                           f"新增确认 {r['新增获确认']}/{r['新增']}")
        if history:
            dlg.setDetailedText("\n".join(history))
        dlg.addButton("关闭",QMessageBox.RejectRole)
        reset = dlg.addButton("恢复本图所有脑片的项目参数…",QMessageBox.ActionRole) if lines else None
        dlg.exec_()
        if reset is not None and dlg.clickedButton() is reset and QMessageBox.question(
                self,"恢复项目参数","恢复本图所有脑片的识别参数并重新识别？") == QMessageBox.Yes:
            im.review_settings.clear(); im.status = "参数已恢复，待重新识别"
            self.project.save(); self.run_detection(False)
