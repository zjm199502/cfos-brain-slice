"""导出 Excel：脑片和脑区的汇总表及分析参数。"""
from __future__ import annotations

from .atlas_choice import STAM, project_atlas, structure_id
from .atlas_choice import has_registration
import datetime as _dt
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from .coloc import colocalize, merged_cells, summarize, nucleus_channel
from .project import EXPORT_HEMISPHERES, Project, processed_section_names, section_result_states
from .. import APP_NAME


_RESULT_TILE_COLUMN = "结果使用的核心图块（不含重叠）"


def _tile_size_text(edge, percent, source_pixels):
    if percent is None:
        unit = "原图像素" if source_pixels else "识别层像素"
        return f"{edge} × {edge} {unit}"
    result = f"{edge} × {edge} 原图像素"
    if percent != 100:
        effective = max(1, round(edge / (100 / percent)))
        result += f"（识别层 {effective} × {effective} 像素）"
    return result


def _recorded_tile_size(section_settings, channel_names):
    """Read the saved inference settings; old results can lack this information."""
    settings = section_settings.get("inference_settings") if isinstance(section_settings, dict) else None
    if isinstance(settings, str):
        try:
            settings = json.loads(settings)
        except ValueError:
            settings = None
    if not isinstance(settings, dict) or not isinstance(settings.get("tile"), int) or settings["tile"] <= 0:
        return "历史结果未记录"
    channels = settings.get("ch")
    if not isinstance(channels, dict):
        return "历史结果未记录"
    groups = {}
    for key, channel in channels.items():
        if not str(key).isdigit() or not isinstance(channel, dict):
            continue
        index = int(key)
        name = channel_names[index] if index < len(channel_names) else f"通道 {index + 1}"
        percent = channel.get("resolution_percent", settings.get("resolution_percent"))
        if percent is not None and not (isinstance(percent, (int, float)) and 0 < percent <= 100):
            return "历史结果未记录"
        size = _tile_size_text(settings["tile"], percent, settings.get("tile_in_source_pixels", False))
        groups.setdefault(size, []).append(name)
    return "；".join(f"{'、'.join(names)} {size}" for size, names in groups.items()) or "历史结果未记录"


@project_atlas
def build_tables(project: Project, intensity: bool | None = None, *, for_excel=False, all_regions=False) -> dict[str, pd.DataFrame]:
    """intensity：是否统计脑区 / 自定义区域的平均荧光强度（要按原始分辨率读原图，较慢）。
    None = 按项目设置（“⑧ 导出 Excel”里的勾选框）。组间统计等内部分析可用 all_regions 读取完整脑区表。"""
    from . import region_sets as RS
    custom = not all_regions and project.export_region_set == RS.CUSTOM
    selected_ids = getattr(project, "export_region_ids", [])
    if custom:
        RS.custom_regions(selected_ids)
    if intensity is None:
        intensity = bool(getattr(project, "export_intensity", False))
    names = project.channel_names
    from .unmix import model_from_settings
    from .segmentation import recorded_model_name
    um = model_from_settings(project.unmix, names) if intensity else None   # v0.8.3：强度也用去除串色后的图
    per_section, region_sec, roi_rows = [], [], []
    roi_counts = []
    section_info = []
    from . import registration
    from .pipeline import drop_excluded
    for im in project.images:
        df = drop_excluded(project.load_cells(im), im)
        enabled = processed_section_names(im, df)
        if for_excel and im.rois and df is not None:
            from .roi_counts import count_rois, saved_coverage
            channels = [(c, name) for c, name in enumerate(names) if c not in project.unused_channels]
            counts = count_rois(df, im.rois, saved_coverage(im, df), channels)
            counts.insert(0, "图片", im.key)
            roi_counts.append(counts)
        states = section_result_states(im, df) if for_excel else {}
        carried = set(df.attrs.get("carried_forward_sections", [])) if df is not None else set()
        for sec in im.sections:
            if not sec.enabled or (not for_excel and sec.name not in enabled):
                continue
            sample = project.samples.get(f"{im.key}/{sec.name}") or {}
            reg = registration.Registration.from_dict(sec.registration)
            row = {"图片": im.key, "脑片": sec.name, "动物": sample.get("animal", ""),
                   "组别": sample.get("group", ""),
                   "约 Bregma (mm)": round(registration.adjusted_plane(reg).ap_mm(), 2) if reg and reg.done else None}
            if for_excel:
                row["识别状态"] = states[sec.name]
                row["筛选说明"] = "沿用上次结果" if sec.name in carried else ""
                model_map = df.attrs.get("section_models", {}).get(sec.name, {}) if df is not None else {}
                used = set(model_map.values())
                row["结果使用的识别模型"] = "、".join(sorted({recorded_model_name(name) for name in used}))
                if not used and sec.name in enabled:
                    row["结果使用的识别模型"] = "历史结果（模型未记录）"
                devices = df.attrs.get("section_devices", {}).get(sec.name, {}) if df is not None else {}
                row["结果使用的推理设备"] = "、".join(sorted({device for values in devices.values() for device in values}))
                settings = df.attrs.get("section_settings", {}).get(sec.name) if sec.name in enabled else None
                row[_RESULT_TILE_COLUMN] = _recorded_tile_size(settings, names) if sec.name in enabled else ""
            section_info.append(row)
        if not enabled:
            continue
        df = df[df["section"].isin(enabled)]
        if len(df) and "combo" not in df.columns:
            with closing(im.open_slide()) as slide:
                df = colocalize(df, slide.pixel_size_um, project.coloc, nucleus_channel(names))
        cells = merged_cells(df, len(names))
        cells.insert(0, "图片", im.key)
        s = summarize(cells, names, group_cols=("图片", "section"), require_nucleus=project.coloc.require_nucleus)
        from .coloc import add_nucleus_ratios
        covered = pd.DataFrame([(im.key, sec.name) for sec in im.sections if sec.name in enabled],
                               columns=["图片", "section"])
        s = covered.merge(s, on=["图片", "section"], how="left")
        for col in s.columns.difference(["图片", "section"]):
            if "比例" not in col:
                s[col] = s[col].fillna(0).astype(int)
        s = add_nucleus_ratios(s, names)     # v0.8.3：X+ 占 DAPI+ 的比例 %（推荐主指标）
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
        reg_secs = [s_ for s_ in im.sections if s_.name in enabled and has_registration(s_.registration)]
        if not reg_secs and not im.rois:
            continue
        with closing(im.open_slide()) as slide:
            px = slide.pixel_size_um
            if reg_secs:
                from .regions import assign_regions, region_summary
                cells, areas = assign_regions(cells, slide, reg_secs, project.results_dir, im.key, im.exclusions,
                                              intensity=intensity, unmix_model=um)
                areas["图片"] = im.key
                rcells = cells[cells["section"].isin({s_.name for s_ in reg_secs})]
                region_sec.append(region_summary(rcells, areas, names, ("图片", "section"), project.coloc.require_nucleus,
                                                 include_allen_id=custom))
            if im.rois:
                from .pipeline import inside_any
                for r in im.rois:
                    boxes = getattr(project, "scope_boxes", {}).get(im.key)
                    area = _roi_area_mm2(r["polygon"], im.exclusions, px, boxes=boxes)
                    inn = inside_any(cells["x"].to_numpy(float), cells["y"].to_numpy(float), [r["polygon"]])
                    if boxes is not None:
                        inn &= _in_boxes(cells["x"].to_numpy(float), cells["y"].to_numpy(float), boxes)
                        if not area and not inn.any():
                            continue
                    sub = cells[inn].copy()
                    sub["区域"] = r["name"]
                    t = summarize(sub, names, group_cols=("图片", "区域"), require_nucleus=project.coloc.require_nucleus) \
                        if len(sub) else pd.DataFrame({"图片": [im.key], "区域": [r["name"]]})
                    t["面积 (mm²)"] = round(area, 4)
                    if intensity:
                        for nm, v in zip(names, roi_mean_intensity(slide, r["polygon"], im.exclusions, unmix_model=um,
                                                                   boxes=boxes)):
                            t[f"{nm} 平均荧光强度"] = round(v, 1) if v == v else None
                    for nm in names:
                        if f"{nm}+" in t.columns and area > 0:
                            t[f"{nm}+ 密度 (个/mm²)"] = (t[f"{nm}+"] / area).round(1)
                    from .coloc import add_nucleus_ratios
                    roi_rows.append(add_nucleus_ratios(t, names))
    out = {}
    if roi_counts:
        out["ROI细胞计数"] = pd.concat(roi_counts, ignore_index=True)
    if per_section:
        # 只保留三张表：按脑片汇总、按脑区×脑片、分析参数（分析参数在 export_excel 里加）
        t = pd.concat(per_section, ignore_index=True).rename(columns={"section": "脑片"})
        sections = pd.DataFrame(section_info)
        t = sections.merge(t, on=["图片", "脑片"], how="left")
        if t["动物"].astype(bool).any():
            t = t.sort_values(["组别", "动物", "约 Bregma (mm)"], ascending=[True, True, False], na_position="last")
        out["按脑片汇总"] = t.reset_index(drop=True)
        if region_sec:
            t = pd.concat(region_sec, ignore_index=True).rename(columns={"section": "脑片"})
            if not all_regions:
                t = RS.filter_table(t, project.export_region_set, project.export_max_depth, project.export_skip_layers,
                                    region_ids=selected_ids)
            out["按脑区×脑片"] = t
        elif custom and for_excel:
            out["按脑区×脑片"] = pd.DataFrame(columns=["图片", "脑片", structure_id(), "脑区缩写", "脑区名称", "层级",
                                                   "半脑", "面积 (mm²)", *(f"{name}+" for name in names)])
        if for_excel and "按脑区×脑片" in out:
            keys = ["图片", "脑片"]
            regions = out["按脑区×脑片"].set_index(keys)
            samples = sections[keys + ["组别", "动物"]].drop_duplicates().set_index(keys)
            labels = samples.reindex(regions.index)
            out["按脑区×脑片"] = pd.concat([labels, regions], axis=1).reset_index()
        if roi_rows:  # 只有画了自定义区域时才有这张表
            out["按自定义区域"] = pd.concat(roi_rows, ignore_index=True)
    return _drop_unused(out, project)


def _drop_unused(tables: dict, project: Project) -> dict:
    """“② 脑片拆分”里设为“不使用”的通道，导出表里不列（它们的名字是“未用通道N”，不会和在用的通道重名）。"""
    bad = [project.channel_names[c] for c in getattr(project, "unused_channels", []) if c < len(project.channel_names)]
    if not bad:
        return tables
    def unused(col):
        col = str(col)
        return any(b in col for b in bad)
    return {k: v.loc[:, [c for c in v.columns if not unused(c)]] for k, v in tables.items()}


def _in_boxes(x, y, boxes):
    inside = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
    for left, top, width, height in boxes:
        inside |= (x >= left) & (x < left + width) & (y >= top) & (y < top + height)
    return inside


def roi_mean_intensity(slide, polygon, exclusions, target_um: float = 0.0, unmix_model=None, *, boxes=None) -> list[float]:
    """自定义区域内（去掉排除区域）每个通道的平均荧光强度，原始灰度值。
    开启了串色校正时（unmix_model），先逐像素去除串色再求平均（v0.8.3）。"""
    from .regions import _usable_unmix
    um = _usable_unmix(slide, unmix_model)
    from skimage.draw import polygon as skpoly
    a = np.asarray(polygon, float)
    n_ch = len(slide.channels)
    if len(a) < 3:
        return [float("nan")] * n_ch
    lv = slide.level_for_pixel_size(target_um)
    ds = slide.downsample(lv)
    x0, y0 = a[:, 0].min(), a[:, 1].min()
    bb = (int(x0), int(y0), int(np.ceil(a[:, 0].max() - x0)) + 1, int(np.ceil(a[:, 1].max() - y0)) + 1)
    if boxes is not None:
        cuts = [(max(bb[0], x), max(bb[1], y), min(bb[0]+bb[2], x+w), min(bb[1]+bb[3], y+h))
                for x, y, w, h in boxes]
        cuts = [c for c in cuts if c[0] < c[2] and c[1] < c[3]]
        if not cuts:
            return [float("nan")] * n_ch
        x0 = min(c[0] for c in cuts)
        y0 = min(c[1] for c in cuts)
        bb = (x0, y0, max(c[2] for c in cuts) - x0, max(c[3] for c in cuts) - y0)
    sums, n = np.zeros(n_ch), 0
    ox, oy = int(bb[0] / ds) * ds, int(bb[1] / ds) * ds
    full = None
    if um is not None:
        from .unmix import apply as _unmix
        full = _unmix(slide.read(lv, bb), um)
    for c in range(n_ch):
        img = full[c] if full is not None else slide.read(lv, bb, [c])[0]
        if c == 0:
            H, W = img.shape
            m = np.zeros((H, W), bool)
            rr, cc = skpoly((a[:, 1] - oy) / ds, (a[:, 0] - ox) / ds, (H, W)); m[rr, cc] = True
            if boxes is not None:
                m &= _in_boxes(ox + np.arange(W)[None, :] * ds, oy + np.arange(H)[:, None] * ds, boxes)
            for e in exclusions or []:
                e = np.asarray(e, float)
                if len(e) >= 3:
                    rr, cc = skpoly((e[:, 1] - oy) / ds, (e[:, 0] - ox) / ds, (H, W)); m[rr, cc] = False
            n = int(m.sum())
        if n:
            sums[c] = float(img[m].astype(np.float64).sum())
    return [s / n if n else float("nan") for s in sums]


def _roi_area_mm2(polygon, exclusions, px_um, *, boxes=None) -> float:
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
    if boxes is not None:
        inn &= _in_boxes(gx, gy, boxes)
    if exclusions:
        inn &= ~inside_any(gx, gy, exclusions)
    return float(inn.sum()) * (10.0 / 1000.0) ** 2


def _version() -> str:
    from .. import __version__
    return __version__


def _net_text(project: Project) -> str:
    """列出此会话的推理状态，项目可以包含由其他设备生成的旧结果。"""
    from .segmentation import NAMES
    if any(p.detect and c not in project.unused_channels for c, p in project.detection.channels.items()):
        return "、".join(sorted({NAMES[p.model] for c,p in project.detection.channels.items() if p.detect and c not in project.unused_channels}))
    return "未启用识别通道"


@project_atlas
def _registration_rows(project: Project) -> list:
    """每张脑片的配准方法汇总（论文方法部分用）。"""
    from .registration import Registration
    from collections import Counter
    methods = Counter()
    fixes, fixed_secs, total = 0, 0, 0
    ap_n, ap_um, unified = 0, [], 0
    for im in project.images:
        for s in im.sections:
            r = Registration.from_dict(s.registration)
            if not r.done:
                continue
            total += 1
            method = r.method + " 定位"
            if r.nonlinear:
                method += " + 非线性贴合" + (f"（{r.nl_method}）" if r.nl_method else "")
            methods[method] += 1
            if r.control_points or r.landmarks:
                fixes += len(r.control_points) + len(r.landmarks); fixed_secs += 1
            if r.ap_auto:
                ap_n += 1; ap_um.append(abs(r.ap_auto) * 25)
            if r.plane_raw:
                unified += 1
    if not total:
        return []
    parts = [f"{name} {count} 张" for name, count in sorted(methods.items())]
    atlas = "STAM（官方单侧分区，40 µm）" if project.registration_atlas == STAM else "Allen 小鼠脑图谱 CCFv3（25 µm）"
    return [("图谱", atlas),
            (f"配准方法（共 {total} 张脑片）", "；".join(parts)),
            ("前后位置自动校正（图谱模板与 DAPI 互信息最大）",
             f"{ap_n} 张脑片校正过，挪动 {min(ap_um):.0f}–{max(ap_um):.0f} µm" if ap_n else "无"),
            ("同一动物统一切面角度", f"{unified} 张脑片" if unified else "无"),
            ("手动修正", f"{fixed_secs} 张脑片，共 {fixes} 个标志点 / 修正点" if fixes else "无")]


@project_atlas
def params_table(project: Project, section_summary=None) -> pd.DataFrame:
    from .detection import channel_percent
    rows = [("导出时间", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("软件版本", f"{APP_NAME} v{_version()}"),
            ("细胞识别神经网络", _net_text(project)),
            *_registration_rows(project),
            ("共标距离阈值 (µm)", project.coloc.max_dist_um),
            ("共标：中心落入轮廓也算", "是" if project.coloc.use_contour else "否"),
            ("轮廓内 DAPI 信号足够强也算有核", "是" if project.coloc.nucleus_by_intensity else "否"),
            ("只统计有核（DAPI+）的细胞", "是" if project.coloc.require_nucleus else "否"),
            ("切片厚度预设", "厚切片（30–50 µm）" if project.thickness == "thick" else "薄切片（≤ 20 µm）"),
            ("z-stack 合成方式", {"max": "最大投影", "edf": "景深合成", "mean": "平均投影"}.get(project.z_mode, project.z_mode)),
            ("排除区域数", sum(len(im.exclusions) for im in project.images))]
    from .unmix import describe_settings
    rows.append(("通道串色校正", describe_settings(project.unmix, project.channel_names)))
    from .coloc import nucleus_channel, MIN_NUCLEI
    nuc = nucleus_channel(project.channel_names)
    if nuc is not None:
        nn = project.channel_names[nuc]
        rows.append(("推荐主指标", f"X+ 占 {nn}+ 的比例 %：分子为有核（{nn}+）的 X+ 细胞，分母为同一脑片 / 脑区的 {nn}+ 细胞；"
                                   f"{nn}+ 不足 {MIN_NUCLEI} 个的区域和细胞极密的脑区（齿状回、CA1–CA3、小脑皮层等）"
                                   f"在“比例提示”列标出。密度（个/mm²）作为补充"))
    if getattr(project, "export_intensity", False):
        from .unmix import model_from_settings
        m = model_from_settings(project.unmix, project.channel_names)
        on = m is not None and not np.allclose(m.matrix, np.eye(len(m.matrix)))
        rows.append(("脑区平均荧光强度", "按去除串色后的图测量（与细胞识别一致）" if on else "按原图测量"))
    from . import region_sets as RS
    region_ids = getattr(project, "export_region_ids", [])
    rows.append(("导出的脑区范围", RS.describe(project.export_region_set, project.export_max_depth,
                                              project.export_skip_layers, region_ids=region_ids)))
    rows.extend(RS.selection_details(project.export_region_set, region_ids))
    rows.append(("脑区表半脑导出方式", project.export_hemisphere))
    rows.append(("脑区类别说明", RS.CATEGORY_HELP))
    tc = project.tissue_channels()
    rows.append(("拆分脑片用的通道", "全部通道" if tc is None else "、".join(project.channel_names[c] for c in tc)))
    if project.unused_channels:
        rows.append(("不使用的通道", "、".join(f"第 {c + 1} 通道" for c in sorted(project.unused_channels))))
    for c, nm in enumerate(project.channel_names):
        p = project.detection.channels.get(c)
        if p is None or c in project.unused_channels:
            continue
        percent = channel_percent(project.detection, c)
        rows += [(f"{nm} 是否识别", "是" if p.detect else "否"),
                 (f"{nm} 识别分辨率（原图宽高）", f"{percent}%" if percent else f"旧项目 {project.detection.pixel_um} µm/像素"),
                 (f"{nm} 当前核心图块（不含重叠）",
                  _tile_size_text(project.detection.tile, percent, project.detection.tile_in_source_pixels)),
                 (f"{nm} 识别模型", p.model),
                 (f"{nm} Cellpose 掩膜阈值", p.cellprob_threshold),
                 (f"{nm} 最小面积 (µm²)", p.min_area_um2),
                 (f"{nm} 最低平均强度", p.min_intensity),
                 (f"{nm} 识别前伽马增强", p.gamma),
                 (f"{nm} 自动阳性阈值（倍背景）", p.bg_fold if p.bg_fold else "不使用"),
                 (f"{nm} 局部对比度下限（倍周围）", p.min_contrast if p.min_contrast else "不使用"),
                 (f"{nm} 识别前减背景半径 (µm)", p.bg_radius_um if p.bg_radius_um else "不减背景"),
                 (f"{nm} 识别前去卷积", f"是（模糊 {p.deconv_blur_um} µm，{p.deconv_iter} 次）" if p.deconv else "否")]
    if section_summary is not None:
        recorded = section_summary[["图片", "脑片", _RESULT_TILE_COLUMN]]
        for image, section, size in recorded.itertuples(index=False, name=None):
            if size:
                rows.append((f"{image} · {section} 结果使用的核心图块（不含重叠）", size))
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
            rows.append((f"{a['通道']} 识别准确率（{a['模型']}，{a['视野类型']}，参数 {a['参数编号']}，{int(a['检查方框数'])} 个方框）",
                         f"人工 {a['人工计数']} 个，软件 {a['软件识别']} 个，配对 {a['配对成功']} 个；"
                         f"查准率 {a['查准率']:.1%}，查全率 {a['查全率']:.1%}，F1 {a['F1']:.3f}（配对距离 ≤ {MATCH_UM:g} µm）"))
    except Exception:  # noqa: BLE001
        pass
    rows += [("图片", im.path) for im in project.images]
    for im in project.images:
        if im.scene_id is not None:
            label = "CZI 场景" if Path(im.path).suffix.lower() == ".czi" else "VSI 扫描区"
            rows.append((f"{label}（{im.key}）", im.scene_description))
            if im.czi_time_index is not None:
                rows.append((f"CZI 时间点（{im.key}）", f"第 1 个（原始 T={im.czi_time_index}，共 {im.czi_time_count} 个）"))
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


def companion_dir(project: Project, path: str | Path, prefix: str) -> Path:
    path = Path(path)
    tag = getattr(project, "scope_tag", "")
    suffix = f"_范围_{tag}" if tag else ""
    return path.parent / f"{prefix}_{path.stem}{suffix}"


@project_atlas
def export_excel(project: Project, path: str | Path, masks: bool = True, progress=None, qc: bool = True,
                 quality: bool = True) -> Path:
    path = Path(path)
    if progress:
        progress(0, 1, "正在统计…")
    tables = build_tables(project, for_excel=True)
    if not tables:
        raise ValueError("还没有任何细胞识别结果，请先完成“细胞识别”。")
    if quality:
        from .quality import quality_table
        import logging
        try:
            tables["按脑片汇总"] = add_quality(tables["按脑片汇总"], quality_table(project, progress=progress))
        except Exception:  # noqa: BLE001  质量检查出错不影响导出
            logging.getLogger("cfos").exception("质量检查失败")
    if "按脑区×脑片" in tables:   # 只在导出的 Excel 里加（组间统计等内部计算不需要）
        from .region_sets import annotate
        regions = tables["按脑区×脑片"]
        sides = EXPORT_HEMISPHERES[project.export_hemisphere]
        if sides is not None:
            regions = regions.loc[regions["半脑"].isin(sides)]
        tables["按脑区×脑片"] = annotate(regions)
    reviews = []
    for image in project.images:
        for r in image.field_reviews:
            for label in ("当前", "候选"):
                score = r[label+"计数"]
                tp, fp, fn = (score[k] for k in ("tp","fp","fn"))
                den = 2*tp + fp + fn
                reviews.append({"图片":image.key, "脑片":r["脑片"], "通道":r["通道"],
                                "视野":str(r["视野"]), "类型":r["类型"], "设置":label,
                                "局部对比度":r[label]["min_contrast"], "去卷积":r[label]["deconv"],
                                "TP":tp, "FP":fp, "FN":fn, "F1":2*tp/den if den else None,
                                "阴性区误检/mm²":fp/r["面积mm2"] if not r["人工点"] else None,
                                "新增获确认":r["新增获确认"], "新增候选":r["新增"],
                                "当时采用候选":r["采用候选"]})
    if reviews:
        tables["脑片盲评"] = pd.DataFrame(reviews)
    if project.accuracy:
        from .accuracy import summary
        tables["准确率汇总"] = summary(project)
        fields = pd.DataFrame(project.accuracy)
        tables["准确率视野"] = fields.drop(columns=["人工点", "漏检线索"], errors="ignore")
    tables["分析参数"] = params_table(project, tables["按脑片汇总"])
    from .region_sets import CUSTOM
    if project.export_region_set == CUSTOM and tables["按脑区×脑片"].empty:
        tables["分析参数"].loc[len(tables["分析参数"])] = [
            "自选脑区导出结果", "当前已分析并配准的脑片在所选半脑中没有覆盖所选脑区，脑区表为空。"]
    try:
        _write_xlsx(tables, path)
    except PermissionError:
        raise PermissionError(13, "Excel 文件正在被打开，无法覆盖。请在 Excel 中关闭它，或换一个文件名", str(path))
    from .workflow_status import record_table_output
    record_table_output(project, "excel", tables["按脑片汇总"])
    if masks:
        from .masks import export_masks
        export_masks(project, companion_dir(project, path, "细胞mask"), progress=progress)
    if qc:
        from .qc import export_qc
        export_qc(project, companion_dir(project, path, "质控图"), progress=progress)
    return path


def _write_xlsx(tables, path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            if name == "按脑区×脑片":
                ws.auto_filter.ref = ws.dimensions
            for j, col in enumerate(df.columns, 1):
                if col in ("组别", "动物", "ROI", "图片"):
                    for i, label in enumerate(df[col], 2):
                        if isinstance(label, str) and label.startswith(("=", "#")):
                            ws.cell(i, j).data_type = "s"
                if col == "面积 (mm²)":
                    for cell in ws.iter_cols(min_col=j, max_col=j, min_row=2, max_row=ws.max_row):
                        for value in cell:
                            value.number_format = "0.0000####"
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
