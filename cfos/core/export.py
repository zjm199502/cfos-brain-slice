"""导出 Excel：汇总表、细胞明细和分析参数。"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from .coloc import colocalize, merged_cells, summarize, nucleus_channel
from .project import Project


def build_tables(project: Project) -> dict[str, pd.DataFrame]:
    names = project.channel_names
    per_section, per_image, details = [], [], []
    region_img, region_sec, roi_rows = [], [], []
    for im in project.images:
        df = project.load_cells(im)
        if df is None or df.empty:
            continue
        enabled = {s.name for s in im.sections if s.enabled}
        df = df[df["section"].isin(enabled)]
        from .pipeline import drop_excluded
        df = drop_excluded(df, im)
        if "combo" not in df.columns:
            from .slide_io import open_slide
            sl = open_slide(im.path)
            df = colocalize(df, sl.pixel_size_um, project.coloc, nucleus_channel(names))
            sl.close()
        cells = merged_cells(df, len(names))
        from .grading import add_grades
        cells = add_grades(cells, names, project.grading)
        cells.insert(0, "图片", im.key)
        s = summarize(cells, names, group_cols=("图片", "section"), require_nucleus=project.coloc.require_nucleus)
        # 动物、组别、前后位置（按动物和 Bregma 排序，方便看连续切片）
        from . import registration as _R
        secmap = {x.name: x for x in im.sections}
        def _ap(n):
            r = _R.Registration.from_dict(secmap[n].registration) if n in secmap else None
            return round(_R.adjusted_plane(r).ap_mm(), 2) if r is not None and r.done else None
        info = lambda n, k: (project.samples.get(f"{im.key}/{n}") or {}).get(k, "")
        s.insert(2, "动物", s["section"].map(lambda n: info(n, "animal")))
        s.insert(3, "组别", s["section"].map(lambda n: info(n, "group")))
        s.insert(4, "约 Bregma (mm)", s["section"].map(_ap))
        from .manual import load_manual, correction_counts
        cc = correction_counts(load_manual(project, im))
        if len(cc):
            for c, nm in enumerate(names):
                sub = cc[cc["channel"] == c].set_index("section")
                add = s["section"].map(sub["补加"]).fillna(0).astype(int)
                rem = s["section"].map(sub["删除"]).fillna(0).astype(int)
                auto = df[df["channel"] == c].groupby("section").size()
                n_auto = s["section"].map(auto).fillna(0) - add + rem
                s[f"{nm} 人工补加"] = add
                s[f"{nm} 人工删除"] = rem
                s[f"{nm} 人工改动比例 %"] = (100 * (add + rem) / n_auto.replace(0, np.nan)).round(2)
        per_section.append(s)
        from .slide_io import open_slide as _open
        _sl = _open(im.path); px = _sl.pixel_size_um
        reg_secs = [s_ for s_ in im.sections if s_.enabled and s_.registration.get("plane")]
        if reg_secs:
            from .regions import assign_regions, region_summary
            from .atlas import load_atlas
            cells, areas = assign_regions(cells, _sl, reg_secs, project.results_dir, im.key, im.exclusions,
                                          intensity=True)
            rcells = cells[cells["section"].isin({s_.name for s_ in reg_secs})]
            region_sec.append(region_summary(rcells, areas, names, ("图片", "section"), project.coloc.require_nucleus))
            st = load_atlas().structures.set_index("index")
            cells["脑区缩写"] = cells["region_index"].map(st["acronym"]).fillna("（未配准/图谱外）")
            cells["半脑"] = cells["hemisphere"].map({1: "左", 2: "右"}).fillna("")
            cells = cells.drop(columns=["region_index", "hemisphere"])
        if im.rois:
            from .pipeline import inside_any
            for r in im.rois:
                inn = inside_any(cells["x"].to_numpy(float), cells["y"].to_numpy(float), [r["polygon"]])
                sub = cells[inn].copy()
                sub["区域"] = r["name"]
                t = summarize(sub, names, group_cols=("图片", "区域"), require_nucleus=project.coloc.require_nucleus) \
                    if len(sub) else pd.DataFrame({"图片": [im.key], "区域": [r["name"]]})
                area = _roi_area_mm2(r["polygon"], im.exclusions, px)
                t["面积 (mm²)"] = round(area, 4)
                for nm, v in zip(names, roi_mean_intensity(_sl, r["polygon"], im.exclusions)):
                    t[f"{nm} 平均荧光强度"] = round(v, 1) if v == v else None
                for nm in names:
                    if f"{nm}+" in t.columns and area > 0:
                        t[f"{nm}+ 密度 (个/mm²)"] = (t[f"{nm}+"] / area).round(1)
                roi_rows.append(t)
        _sl.close()
        cells["x"] = (cells["x"] * px).round(1)
        cells["y"] = (cells["y"] * px).round(1)
        det = cells.rename(columns={"section": "脑片", "merged_id": "细胞编号", "x": "X (µm)", "y": "Y (µm)"})
        for c, nm in enumerate(names):
            det = det.rename(columns={f"pos_ch{c}": f"{nm}阳性"})
        details.append(det)
    out = {}
    if per_section:
        # 只保留三张表：按脑片汇总、按脑区×脑片、分析参数（分析参数在 export_excel 里加）
        t = pd.concat(per_section, ignore_index=True).rename(columns={"section": "脑片"})
        if t["动物"].astype(bool).any():
            t = t.sort_values(["组别", "动物", "约 Bregma (mm)"], ascending=[True, True, False], na_position="last")
        out["按脑片汇总"] = t.reset_index(drop=True)
        if region_sec:
            t = pd.concat(region_sec, ignore_index=True).rename(columns={"section": "脑片"})
            if project.export_skip_layers:
                t = t[~t["脑区名称"].astype(str).str.contains(r"[Ll]ayer|/Layer", regex=True)]
            if project.export_max_depth > 0:
                t = t[t["层级"] <= project.export_max_depth]
            out["按脑区×脑片"] = t
        if roi_rows:  # 只有画了自定义区域时才有这张表
            out["按自定义区域"] = pd.concat(roi_rows, ignore_index=True)
    return out


def roi_mean_intensity(slide, polygon, exclusions, target_um: float = 0.0) -> list[float]:
    """自定义区域内（去掉排除区域）每个通道的平均荧光强度，原始灰度值。"""
    from skimage.draw import polygon as skpoly
    a = np.asarray(polygon, float)
    n_ch = len(slide.channels)
    if len(a) < 3:
        return [float("nan")] * n_ch
    lv = slide.level_for_pixel_size(target_um)
    ds = slide.downsample(lv)
    x0, y0 = a[:, 0].min(), a[:, 1].min()
    bb = (int(x0), int(y0), int(np.ceil(a[:, 0].max() - x0)) + 1, int(np.ceil(a[:, 1].max() - y0)) + 1)
    sums, n = np.zeros(n_ch), 0
    ox, oy = int(bb[0] / ds) * ds, int(bb[1] / ds) * ds
    for c in range(n_ch):
        img = slide.read(lv, bb, [c])[0]
        if c == 0:
            H, W = img.shape
            m = np.zeros((H, W), bool)
            rr, cc = skpoly((a[:, 1] - oy) / ds, (a[:, 0] - ox) / ds, (H, W)); m[rr, cc] = True
            for e in exclusions or []:
                e = np.asarray(e, float)
                if len(e) >= 3:
                    rr, cc = skpoly((e[:, 1] - oy) / ds, (e[:, 0] - ox) / ds, (H, W)); m[rr, cc] = False
            n = int(m.sum())
        if n:
            sums[c] = float(img[m].astype(np.float64).sum())
    return [s / n if n else float("nan") for s in sums]


def _roi_area_mm2(polygon, exclusions, px_um) -> float:
    """自定义区域面积（减去与排除区域重叠的部分），在 10 µm 网格上估算。"""
    import numpy as np
    from .pipeline import inside_any
    a = np.asarray(polygon, float)
    step = 10.0 / px_um
    xs = np.arange(a[:, 0].min(), a[:, 0].max() + step, step)
    ys = np.arange(a[:, 1].min(), a[:, 1].max() + step, step)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    inn = inside_any(gx, gy, [polygon])
    if exclusions:
        inn &= ~inside_any(gx, gy, exclusions)
    return float(inn.sum()) * (10.0 / 1000.0) ** 2


def _version() -> str:
    from .. import __version__
    return __version__


def _net_text() -> str:
    """StarDist 通用模型（2D_versatile_fluo）+ 实际运行方式。"""
    from . import accel
    dev = accel._status.get("device") or ""
    if "ONNX" not in dev:
        dev = "ONNX Runtime（CPU）" if accel.net_backend() == "onnx" and accel._ort() is not None else "TensorFlow（CPU）"
    return f"StarDist（{dev}）"


def _registration_rows(project: Project) -> list:
    """每张脑片的配准方法汇总（论文方法部分用）。"""
    from .registration import Registration
    n = {"lin": 0, "tex": 0, "cont": 0, "old": 0}
    fixes, fixed_secs, total = 0, 0, 0
    for im in project.images:
        for s in im.sections:
            r = Registration.from_dict(s.registration)
            if not r.done:
                continue
            total += 1
            if not r.nonlinear:
                n["lin"] += 1
            elif r.nl_method == "轮廓 + DAPI 纹理":
                n["tex"] += 1
            elif r.nl_method == "只用轮廓":
                n["cont"] += 1
            else:
                n["old"] += 1
            if r.control_points:
                fixes += len(r.control_points); fixed_secs += 1
    if not total:
        return []
    parts = [f"{k} {v} 张" for k, v in (("DeepSlice 定位 + 非线性贴合（轮廓 + DAPI 纹理）", n["tex"]),
                                       ("DeepSlice 定位 + 非线性贴合（只用轮廓）", n["cont"]),
                                       ("DeepSlice 定位 + 非线性贴合（只用轮廓，v0.7 之前配准）", n["old"]),
                                       ("DeepSlice 定位，未做非线性贴合", n["lin"])) if v]
    return [("图谱", "Allen 小鼠脑图谱 CCFv3（25 µm）"),
            (f"配准方法（共 {total} 张脑片）", "；".join(parts)),
            ("手动修正", f"{fixed_secs} 张脑片，共 {fixes} 个修正点" if fixes else "无")]


def params_table(project: Project) -> pd.DataFrame:
    rows = [("导出时间", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("软件版本", f"cFos 脑片分析 v{_version()}"),
            ("细胞识别神经网络", _net_text()),
            *_registration_rows(project),
            ("识别分辨率 (µm/像素)", project.detection.pixel_um),
            ("共标距离阈值 (µm)", project.coloc.max_dist_um),
            ("共标：中心落入轮廓也算", "是" if project.coloc.use_contour else "否"),
            ("轮廓内 DAPI 信号足够强也算有核", "是" if project.coloc.nucleus_by_intensity else "否"),
            ("只统计有核（DAPI+）的细胞", "是" if project.coloc.require_nucleus else "否"),
            ("切片厚度预设", "厚切片（30–50 µm）" if project.thickness == "thick" else "薄切片（≤ 20 µm）"),
            ("z-stack 合成方式", {"max": "最大投影", "edf": "景深合成", "mean": "平均投影"}.get(project.z_mode, project.z_mode)),
            ("排除区域数", sum(len(im.exclusions) for im in project.images))]
    for nm, cuts in (project.grading or {}).items():
        rows.append((f"{nm} 强度分级（倍背景）", f"弱 < {cuts[0]} ≤ 中 < {cuts[1]} ≤ 强"))
    for c, nm in enumerate(project.channel_names):
        p = project.detection.channels.get(c)
        if p is None:
            continue
        rows += [(f"{nm} 是否识别", "是" if p.detect else "否"),
                 (f"{nm} 识别模型", f"微调模型“{p.model}”" if p.model else "StarDist 通用荧光细胞核模型（2D_versatile_fluo）"),
                 (f"{nm} 置信度阈值", p.prob_thresh),
                 (f"{nm} 面积范围 (µm²)", f"{p.min_area_um2} – {p.max_area_um2}"),
                 (f"{nm} 最低平均强度", p.min_intensity),
                 (f"{nm} 识别前伽马增强", p.gamma),
                 (f"{nm} 自动阳性阈值（倍背景）", p.bg_fold if p.bg_fold else "不使用"),
                 (f"{nm} 局部对比度下限（倍周围）", p.min_contrast if p.min_contrast else "不使用"),
                 (f"{nm} 识别前减背景半径 (µm)", p.bg_radius_um if p.bg_radius_um else "不减背景"),
                 (f"{nm} 识别前去卷积", f"是（模糊 {p.deconv_blur_um} µm，{p.deconv_iter} 次）" if p.deconv else "否")]
    from . import quality as Q
    rows.append(("质量检查标准（注意 / 可疑）",
                 f"配准重合度 < {Q.DICE_WARN} / {Q.DICE_BAD}；模糊面积 ≥ {Q.BLUR_WARN:g}% / {Q.BLUR_BAD:g}%；"
                 f"过曝像素 ≥ {Q.SAT_WARN:g}% / {Q.SAT_BAD:g}%；组织缺损 ≥ {Q.DAMAGE_WARN:g}% / {Q.DAMAGE_BAD:g}%；"
                 f"整体清晰度 < 中位数的 {Q.SHARP_WARN:g} / {Q.SHARP_BAD:g} 倍；"
                 f"DAPI 密度 < {Q.DENS_LOW:g} 或 > {Q.DENS_HIGH:g} 倍中位数为可疑"))
    try:
        from .accuracy import summary, MATCH_UM
        acc = summary(project)
        for _, a in acc.iterrows():
            rows.append((f"{a['通道']} 识别准确率（人工点数对照，{int(a['检查方框数'])} 个方框）",
                         f"人工 {a['人工计数']} 个，软件 {a['软件识别']} 个，配对 {a['配对成功']} 个；"
                         f"查准率 {a['查准率']:.1%}，查全率 {a['查全率']:.1%}，F1 {a['F1']:.3f}（配对距离 ≤ {MATCH_UM:g} µm）"))
    except Exception:  # noqa: BLE001
        pass
    rows += [("图片", im.path) for im in project.images]
    return pd.DataFrame(rows, columns=["参数", "值"])


def add_quality(table: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """把质量检查结果并入“按脑片汇总”：等级和问题放在前面，具体指标放在最后。"""
    if q is None or q.empty:
        return table
    q = q.rename(columns={"脑片": "_脑片"})
    t = table.merge(q, left_on=["图片", "脑片"], right_on=["图片", "_脑片"], how="left").drop(columns=["_脑片"])
    front = ["质量", "质量分", "需要检查的问题"]
    cols = [c for c in t.columns if c not in front]
    k = cols.index("约 Bregma (mm)") + 1 if "约 Bregma (mm)" in cols else 2
    metrics = [c for c in q.columns if c not in front + ["图片", "_脑片"]]
    cols = [c for c in cols if c not in metrics]
    return t[cols[:k] + front + cols[k:] + metrics]


def export_excel(project: Project, path: str | Path, masks: bool = True, progress=None, qc: bool = True,
                 quality: bool = True) -> Path:
    path = Path(path)
    if progress:
        progress(0, 1, "正在统计…")
    tables = build_tables(project)
    if not tables:
        raise ValueError("还没有任何细胞识别结果，请先完成“细胞识别”。")
    if quality:
        from .quality import quality_table
        import logging
        try:
            tables["按脑片汇总"] = add_quality(tables["按脑片汇总"], quality_table(project, progress=progress))
        except Exception:  # noqa: BLE001  质量检查出错不影响导出
            logging.getLogger("cfos").exception("质量检查失败")
    tables["分析参数"] = params_table(project)
    try:
        _write_xlsx(tables, path)
    except PermissionError:
        raise PermissionError(13, "Excel 文件正在被打开，无法覆盖。请在 Excel 中关闭它，或换一个文件名", str(path))
    if masks:
        from .masks import export_masks
        export_masks(project, path.parent / f"细胞mask_{path.stem}", progress=progress)
    if qc:
        from .qc import export_qc
        export_qc(project, path.parent / f"质控图_{path.stem}", progress=progress)
    return path


def _write_xlsx(tables, path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for j, col in enumerate(df.columns, 1):
                ws.cell(1, j).font = Font(bold=True)
                ws.cell(1, j).fill = PatternFill("solid", fgColor="DDEBF7")
                ws.cell(1, j).alignment = Alignment(horizontal="center", wrap_text=True)
                w = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
                ws.column_dimensions[get_column_letter(j)].width = min(max(w, 8), 60)
            ws.freeze_panes = "A2"
            if "质量" in df.columns:  # 可疑 = 浅红，注意 = 浅黄
                fills = {"可疑": PatternFill("solid", fgColor="F8CBAD"), "注意": PatternFill("solid", fgColor="FFF2CC")}
                for i, v in enumerate(df["质量"], 2):
                    if v in fills:
                        for j in range(1, len(df.columns) + 1):
                            ws.cell(i, j).fill = fills[v]
