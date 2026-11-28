"""各步骤“清除本步结果”：先预览不改动，确认后删除结果和磁盘缓存，后续步骤的记录一并去掉。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


from cfos.core import clear_results as CR
from cfos.core import workflow_status as ws
from cfos.core.coloc_table import image_summary
from cfos.core.manual import load_manual, save_manual
from cfos.core.project import Project, ImageEntry
from cfos.core.sections import Section

WARP_A = "a" * 64 + ".npy"
WARP_B = "b" * 64 + ".npy"
REG = {"plane": [1, 0, 0, 0, 1, 0, 0, 0, 1], "warp_file": WARP_A, "fit_atlas": "CCFv3", "status": "已配准",
       "landmarks": [[0.1, 0.2, 1, 2, 3]]}


def _project(root, n_images=2, n_sections=3):
    p = Project(folder=Path(root) / "project")
    p.channel_names = ["DAPI", "Green"]
    for i in range(n_images):
        im = ImageEntry(path=str(Path(root) / f"img{i}.tif"), storage_key=f"img{i}", status="已识别细胞")
        im.sections = [Section(j, (j * 64, 0, 64, 64), [(j * 64 + 1., 1.), (j * 64 + 60., 2.), (j * 64 + 30., 60.)],
                               f"S{j}", registration=dict(REG))
                       for j in range(n_sections)]
        p.images.append(im)
    return p


def _cells(sections):
    rows = []
    for k, name in enumerate(sections):
        for c in (0, 1):
            rows.append(dict(section=name, channel=c, x=64. * k + 10, y=10., area=20., merged_id=k,
                             combo="0+1", contour=np.array([[0., 0.], [2., 0.], [2., 2.]], np.float32)))
    df = pd.DataFrame(rows)
    df.attrs["processed_sections"] = list(sections)
    df.attrs["section_models"] = {n: {"0": "m"} for n in sections}
    return df


def _populate(p):
    """每张图片：细胞结果、原始识别缓存、人工修正、配准位移场、分组、输出记录。"""
    res = p.results_dir
    (res / "registration").mkdir(parents=True, exist_ok=True)
    (res / "registration" / WARP_A).write_bytes(b"x")
    for im in p.images:
        names = [s.name for s in im.sections]
        p.save_cells(im, _cells(names))
        for s in im.sections:
            p.save_part(im, s, "sig", _cells([s.name]))
            p.samples[f"{im.key}/{s.name}"] = {"animal": im.key, "group": "g1"}
            (res / "intensity" / im.key).mkdir(parents=True, exist_ok=True)
            (res / "intensity" / im.key / f"{s.name}.pkl").write_bytes(b"x")
        save_manual(p, im, {"added": [{"section": "S0", "channel": 1, "x": 5., "y": 5.}],
                            "removed": [{"section": "S1", "channel": 1, "x": 70., "y": 10.}]})
        im.section_display = {"S0": {"0": {"low": 1}}}
        im.display_overrides = {"0": {"low": 2}}
        im.figure_rois = [{"name": "f1", "x": 10., "y": 10., "w": 5., "h": 5.},
                          {"name": "f2", "x": 70., "y": 10., "w": 5., "h": 5.}]
        ws.record_output(p, "excel", [(im, s) for s in im.sections])
        ws.record_output(p, "group", [(im, s) for s in im.sections])
    p.save()


def _history(p, image, step):
    import sqlite3
    with sqlite3.connect(p.results_dir / "workflow.sqlite") as db:
        return {r[0] for r in db.execute("SELECT section FROM outputs WHERE image=? AND step=?", (image, step))}


def _snapshot(p):
    files = sorted(str(f.relative_to(p.folder)) for f in p.folder.rglob("*") if f.is_file()
                   and f.name != "workflow.sqlite")
    return files, json.dumps([im.to_dict() for im in p.images], sort_keys=True, default=str), json.dumps(p.samples)


class ClearResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = _project(self.tmp.name)
        _populate(self.p)

    def tearDown(self):
        self.p.wait_saved()
        self.tmp.cleanup()

    def test_dry_run_changes_nothing(self):
        before = _snapshot(self.p)
        for step in CR.STEP_TITLES:
            report = CR.clear(self.p, step, {"img0": None}, dry_run=True, include_manual=True, include_outputs=True)
            self.assertIsInstance(CR.describe(report), str)
        self.assertEqual(before, _snapshot(self.p))
        self.assertEqual(_history(self.p, "img0", "excel"), {"S0", "S1", "S2"})

    def test_clear_detection_for_named_sections(self):
        p = self.p
        im = p.images[0]
        preview = CR.clear(p, CR.DETECT, {"img0": ("S0",)})
        self.assertEqual(preview.manual_records, 1)
        self.assertIn("2 个细胞", CR.describe(preview))
        report = CR.clear(p, CR.DETECT, {"img0": ("S0",)}, dry_run=False)
        self.assertFalse(report.empty)
        df = p.load_cells(im)
        self.assertEqual(set(df["section"]), {"S1", "S2"})
        self.assertEqual(df.attrs["processed_sections"], ["S1", "S2"])
        self.assertNotIn("S0", df.attrs["section_models"])
        # 原始识别缓存删掉，下次识别不会再“沿用上次结果”
        self.assertIsNone(p.load_part(im, im.sections[0], "sig"))
        self.assertIsNotNone(p.load_part(im, im.sections[1], "sig"))
        # 默认保留人工修正
        self.assertEqual(len(load_manual(p, im)["added"]), 1)
        # 项目页：S0 待识别，后续记录去掉，其他脑片和图片不动
        progress = dict(ws.section_progress(p, im))
        self.assertEqual(progress["S0"][3], "待识别")
        self.assertEqual(progress["S1"][3], "已识别（有细胞）")
        self.assertEqual(_history(p, "img0", "excel"), {"S1", "S2"})
        self.assertEqual(_history(p, "img1", "excel"), {"S0", "S1", "S2"})
        self.assertEqual(im.status, "已识别细胞（2/3 张脑片）")
        self.assertEqual(len(p.load_cells(p.images[1])), 6)

    def test_clear_detection_whole_image_with_manual(self):
        p = self.p
        im = p.images[0]
        CR.clear(p, CR.DETECT, {"img0": None}, dry_run=False, include_manual=True)
        self.assertIsNone(p.load_cells(im))
        self.assertFalse(p.cells_path(im).exists())
        self.assertFalse(p.cells_path(im).with_suffix(".progress.json").exists())
        self.assertFalse((p.results_dir / "parts" / "img0").exists())
        self.assertFalse((p.results_dir / "img0_manual.json").exists())
        self.assertFalse(p.has_cells(im))
        self.assertEqual(im.status, "已拆分 3 张脑片")
        self.assertTrue(p.has_cells(p.images[1]))

    def test_clear_coloc_keeps_detections(self):
        p = self.p
        im = p.images[0]
        CR.clear(p, CR.COLOC, {"img0": ("S1",)}, dry_run=False)
        df = p.load_cells(im)
        self.assertEqual(len(df), 6)
        self.assertTrue(df.loc[df.section == "S1", "combo"].isna().all())
        self.assertTrue(df.loc[df.section != "S1", "combo"].notna().all())
        progress = dict(ws.section_progress(p, im))
        self.assertEqual(progress["S1"][4], "待计算")
        self.assertEqual(progress["S0"][4], "已有共标结果")
        table = image_summary(p, im, p.channel_names, False)
        self.assertEqual(set(table["section"]), {"S0", "S2"})
        from cfos.core.coloc import merged_cells
        merged = merged_cells(df, 2)   # 被清除的脑片按识别到的通道，不报错
        self.assertEqual(len(merged), 3)

    def test_clear_registration_removes_unshared_warps(self):
        p = self.p
        warp = p.results_dir / "registration" / WARP_A
        CR.clear(p, CR.REGISTRATION, {"img0": None}, dry_run=False)
        self.assertTrue(all(not s.registration for s in p.images[0].sections))
        self.assertTrue(warp.exists())           # img1 仍在用同一个位移场
        self.assertFalse((p.results_dir / "intensity" / "img0" / "S0.pkl").exists())
        self.assertTrue(all(s.registration for s in p.images[1].sections))
        CR.clear(p, CR.REGISTRATION, {"img1": None}, dry_run=False)
        self.assertFalse(warp.exists())
        self.assertEqual(dict(ws.section_progress(p, p.images[1]))["S0"][5], "待配准")
        # 另一图谱的结果不动
        p.images[0].sections[0].other_registrations = {"STAM": {"warp_file": WARP_B}}
        self.assertEqual(CR.clear(p, CR.REGISTRATION, None).lines, [])

    def test_clear_groups_and_output_folders(self):
        p = self.p
        old = p.folder / "组间统计_20261128_000000_000000"
        old.mkdir()
        preview = CR.clear(p, CR.GROUPS, {"img1": ("S2",)})
        self.assertEqual(preview.output_folders, [old])
        CR.clear(p, CR.GROUPS, {"img1": ("S2",)}, dry_run=False)
        self.assertNotIn("img1/S2", p.samples)
        self.assertIn("img1/S1", p.samples)
        self.assertTrue(old.exists())
        self.assertEqual(_history(p, "img1", "group"), {"S0", "S1"})
        CR.clear(p, CR.GROUPS, None, dry_run=False, include_outputs=True)
        self.assertFalse(old.exists())
        self.assertEqual(p.samples, {})

    def test_clear_display_export_figure(self):
        p = self.p
        im = p.images[0]
        CR.clear(p, CR.DISPLAY, {"img0": ("S0",)}, dry_run=False)
        self.assertEqual(im.section_display, {})
        self.assertEqual(im.display_overrides, {"0": {"low": 2}})
        CR.clear(p, CR.DISPLAY, {"img0": None}, dry_run=False)
        self.assertEqual(im.display_overrides, {})
        CR.clear(p, CR.EXPORT, {"img0": None}, dry_run=False)
        self.assertEqual(_history(p, "img0", "excel"), set())
        self.assertEqual(_history(p, "img0", "group"), {"S0", "S1", "S2"})
        CR.clear(p, CR.FIGURE, {"img0": ("S1",)}, dry_run=False)
        self.assertEqual([r["name"] for r in im.figure_rois], ["f1"])

    def test_clear_sections_cascades(self):
        p = self.p
        im = p.images[0]
        CR.clear(p, CR.SECTIONS, {"img0": ("S1",)}, dry_run=False)
        self.assertEqual([s.name for s in im.sections], ["S0", "S2"])
        self.assertNotIn("img0/S1", p.samples)
        self.assertEqual(load_manual(p, im)["removed"], [])
        self.assertEqual(set(p.load_cells(im)["section"]), {"S0", "S2"})
        # 重新拆分出同名脑片时，不会把旧结果当成它的结果
        im.replace_sections(im.sections + [Section(9, (64, 0, 64, 64), [(65., 1.), (120., 2.), (90., 60.)], "S1")])
        self.assertEqual(set(p.load_cells(im)["section"]), {"S0", "S2"})

        CR.clear(p, CR.ALL, {"img1": None}, dry_run=False)
        im1 = p.images[1]
        self.assertEqual(im1.sections, [])
        self.assertEqual(im1.changed_outlines, {})
        self.assertEqual(im1.status, "未处理")
        self.assertIsNone(p.load_cells(im1))
        self.assertFalse((p.results_dir / "parts" / "img1").exists())
        self.assertFalse((p.results_dir / "intensity" / "img1").exists())
        self.assertFalse(any(k.startswith("img1/") for k in p.samples))
        self.assertEqual(_history(p, "img1", "excel"), set())
        # 保存后重新打开项目，状态一致
        loaded = Project.load(p.folder)
        self.assertEqual(loaded.images[1].sections, [])
        self.assertEqual([s.name for s in loaded.images[0].sections], ["S0", "S2", "S1"])

    def test_rerun_texts_cover_every_page(self):
        from cfos.gui.clear_panel import BUTTON_TEXT, PAGE_STEPS, RERUN_TEXT
        from cfos.gui.main_window import STEPS
        self.assertEqual(len(PAGE_STEPS), len(STEPS))
        for step in PAGE_STEPS:
            self.assertTrue(RERUN_TEXT[step] and BUTTON_TEXT[step] and CR.STEP_TITLES[step])


if __name__ == "__main__":
    unittest.main()
