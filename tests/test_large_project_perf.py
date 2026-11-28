"""大项目卡顿优化：保存、共标汇总缓存、处理范围分组表、项目页进度索引。"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from cfos.core import workflow_status as ws
from cfos.core.coloc_table import LRUDict, combine, image_signature, image_summary
from cfos.core.project import PROJECT_FILE, ImageEntry, Project
from cfos.core.scope import scoped_project
from cfos.core.sections import Section
from cfos.gui.processing_scope import ProcessingScopeMixin


def _project(root, n_images=3, n_sections=4):
    p = Project(folder=Path(root) / "project")
    p.channel_names = ["DAPI", "Green"]
    for i in range(n_images):
        im = ImageEntry(path=str(Path(root) / f"img{i}.tif"), storage_key=f"img{i}")
        im.sections = [Section(j, (j * 64, 0, 64, 64), [(0.5, 1.5), (60.0, 2.0), (30.0, 60.0)], f"S{j}",
                               registration={})
                       for j in range(n_sections)]
        p.images.append(im)
    return p


class SaveTests(unittest.TestCase):
    def test_compact_round_trip_and_background_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _project(tmp)
            p.samples = {"img0/S0": {"animal": "A1", "group": "ctrl"}}
            p.save()
            text = (p.folder / PROJECT_FILE).read_text(encoding="utf-8")
            self.assertNotIn("\n", text)                      # 紧凑格式
            loaded = Project.load(p.folder)
            self.assertEqual([im.to_dict() for im in loaded.images], [im.to_dict() for im in p.images])
            self.assertEqual(loaded.samples, p.samples)
            # 后台保存之后再立即保存：文件里必须是最新内容
            p.samples["img1/S0"] = {"animal": "B"}
            p.save(background=True)
            p.samples["img2/S0"] = {"animal": "C"}
            p.save()
            p.wait_saved()
            self.assertIn("img2/S0", Project.load(p.folder).samples)
            # 后台保存期间修改设置，不影响已经排队的快照
            p.save(background=True)
            p.samples.clear()
            p.wait_saved()
            self.assertIn("img2/S0", json.loads((p.folder / PROJECT_FILE).read_text("utf-8"))["samples"])

    def test_source_images_are_checked_once_per_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _project(tmp, n_images=1)
            src = Path(p.images[0].path)
            tifffile.imwrite(src, np.ones((8, 8), np.uint8))
            p.save()
            self.assertEqual(p.images[0].size, src.stat().st_size)
            calls = []
            import cfos.core.relocate as relocate
            original = relocate.file_size
            relocate.file_size = lambda path: calls.append(path) or original(path)
            try:
                p.save()
                self.assertEqual(calls, [])
                moved = Path(tmp) / "moved.tif"
                os.replace(src, moved)
                p.images[0].path = str(moved)              # 重新定位后会再检查一次
                p.save()
                self.assertEqual(calls, [str(moved)])
            finally:
                relocate.file_size = original


class ColocSummaryTests(unittest.TestCase):
    def test_signature_follows_result_file_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _project(tmp, n_images=1, n_sections=2)
            im = p.images[0]
            names = p.channel_names
            self.assertIsNone(image_signature(p, im, names, False))
            cells = pd.DataFrame([dict(section="S0", channel=c, x=10., y=10., merged_id=1, area=20.)
                                  for c in (0, 1)])
            cells.attrs["processed_sections"] = ["S0", "S1"]
            p.save_cells(im, cells)
            sig = image_signature(p, im, names, False)
            self.assertEqual(sig, image_signature(p, im, names, False))
            table = combine([image_summary(p, im, names, False)])
            self.assertEqual(list(table["脑片"]), ["S0"])
            im.exclusions = [[(0, 0), (20, 0), (20, 20), (0, 20)]]
            self.assertNotEqual(sig, image_signature(p, im, names, False))
            time.sleep(0.01)
            p.save_cells(im, pd.concat([cells, cells.assign(section="S1")], ignore_index=True))
            self.assertNotEqual(sig[1:4], image_signature(p, im, names, False)[1:4])

    def test_lru_dict(self):
        d = LRUDict(2)
        d["a"] = 1; d["b"] = 2
        _ = d["a"]
        d["c"] = 3
        self.assertEqual(set(d), {"a", "c"})
        self.assertNotIn("b", d)
        d.clear(); self.assertEqual(len(d), 0)


class ScopeTests(unittest.TestCase):
    def test_scoped_rows_match_snapshot_and_share_outlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _project(tmp)
            p.images[1].sections[2].enabled = False
            p.samples = {"img0/S1": {"animal": "A", "group": "g"}, "img1/S0": {"animal": "B"}}
            p.behavior = {"A": 3.5}
            scope = {"img0": ("S0", "S1"), "img1": None, "img2": ()}

            class Window(ProcessingScopeMixin):
                project = p
                _processing_scope = scope
            self.assertEqual(Window()._scoped_sample_rows(), scoped_project(p, scope).sample_rows())
            view = scoped_project(p, scope)
            self.assertIs(view.images[0].sections[0].contour0, p.images[0].sections[0].contour0)
            self.assertIsNot(view.images[0].sections, p.images[0].sections)


class ProgressIndexTests(unittest.TestCase):
    def test_part_box_index_avoids_reading_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _project(tmp, n_images=1, n_sections=1)
            im, s = p.images[0], p.images[0].sections[0]
            df = pd.DataFrame([dict(section="S0", channel=0, x=1., y=1., contour=[[0., 0.], [2., 0.], [2., 2.]])])
            p.save_part(im, s, "sig", df)
            path = p.part_path(im, s.name)
            self.assertTrue(ws.part_box_index(path).is_file())
            original = pd.read_pickle
            pd.read_pickle = lambda *a, **k: (_ for _ in ()).throw(AssertionError("read whole cache"))
            try:
                ws._part_box.cache_clear()
                self.assertTrue(ws._has_raw_result(p, im, s))
            finally:
                pd.read_pickle = original


if __name__ == "__main__":
    unittest.main()
