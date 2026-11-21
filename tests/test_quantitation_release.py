from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from cfos.core import coloc, regions
from cfos.core.project import Project, ImageEntry
from cfos.core.sections import Section


class Quantitation(unittest.TestCase):
    def test_excel_section_states_and_unprocessed_counts(self):
        from cfos.core.export import build_tables, export_excel
        with TemporaryDirectory() as folder:
            p = Project(Path(folder), channel_names=["Cy5"])
            sections = [Section(i, (0, 0, 32, 32), [], name)
                        for i, name in enumerate(["A", "B", "C"])]
            im = ImageEntry("sample.tif", sections=sections)
            other = ImageEntry("other.tif", sections=[Section(1, (0, 0, 32, 32), [], "D")])
            p.images = [im, other]
            df = pd.DataFrame(dict(section=["A"], merged_id=[0], combo=["0"], channel=[0], x=[1.], y=[2.]))
            df.attrs.update(processed_sections=["A", "B"], carried_forward_sections=["A"])
            p.save_cells(im, df)
            path = export_excel(p, p.folder / "states.xlsx", masks=False, qc=False, quality=False)
            summary = pd.read_excel(path, sheet_name="按脑片汇总").set_index("脑片")
            self.assertEqual(summary.loc["A", "筛选说明"], "沿用上次结果")
            self.assertEqual(summary["识别状态"].tolist(), ["已分析（有细胞）", "已分析（零细胞）", "未分析", "未分析"])
            self.assertEqual(summary.loc[["A", "B"], "Cy5+"].tolist(), [1, 0])
            self.assertTrue(summary.loc[["C", "D"], "Cy5+"].isna().all())
            self.assertEqual(build_tables(p)["按脑片汇总"]["脑片"].tolist(), ["A", "B"])

    def test_excel_hemisphere_modes(self):
        from cfos.core.export import export_excel
        with TemporaryDirectory() as folder:
            project = Project(Path(folder))
            project.save()
            config = project.folder / "cfos_project.json"
            data = json.loads(config.read_text("utf8"))
            del data["export_hemisphere"]
            config.write_text(json.dumps(data), encoding="utf8")
            project = Project.load(project.folder)
            self.assertEqual(project.export_hemisphere, "仅左右合并")
            source = pd.DataFrame({"半脑": ["左", "右", "左右合计"], "细胞数": [2, 7, 9]})
            for mode, counts in [("仅左右合并", [9]), ("仅左半脑", [2]), ("仅右半脑", [7]),
                                 ("左右分别导出", [2, 7]), ("左右分别导出并附合计", [2, 7, 9])]:
                project.export_hemisphere = mode
                project.save()
                loaded = Project.load(project.folder)
                tables = {"按脑片汇总": pd.DataFrame({"细胞数": [9]}), "按脑区×脑片": source}
                with self.subTest(mode=mode), patch("cfos.core.export.build_tables", return_value=tables):
                    file = export_excel(loaded, project.folder / "result.xlsx", masks=False, qc=False, quality=False)
                    sheets = pd.read_excel(file, sheet_name=None)
                    self.assertEqual(sheets["按脑区×脑片"]["细胞数"].tolist(), counts)
                    self.assertEqual(sheets["按脑片汇总"]["细胞数"].tolist(), [9])
                    params = sheets["分析参数"].set_index("参数")["值"]
                    self.assertEqual(params["脑区表半脑导出方式"], mode)

    def test_covered_zero_and_density(self):
        from types import SimpleNamespace
        st = pd.DataFrame([dict(index=1, acronym="root", name="root")])
        data = dict(section=["S1"], 图片=["im"], region_index=[1], hemisphere=[1], pos_ch0=[True], pos_ch1=[True])
        areas = pd.DataFrame(dict(section=["S1","S2"], 图片=["im","im"], region_index=[1,1],
                                  hemisphere=[1,2], area_mm2=[.00004,.002]))
        with patch.object(regions, "load_atlas", return_value=SimpleNamespace(structures=st)), patch.object(
                regions, "_ancestors", return_value={1:[1]}), patch.object(regions, "dense_regions", return_value=set()):
            def run(c, a):
                return regions.region_summary(c, a, ["DAPI","Red"], ("图片","section"))
            cells = pd.DataFrame(data)
            t = run(cells, areas)
            self.assertEqual(t[t.section=="S1"]["Red+ 密度 (个/mm²)"].tolist(), [25000,25000])
            self.assertEqual(t[t.section=="S2"]["Red+"].tolist(), [0,0])
            u = run(cells, areas.assign(n_px=100,sum_ch0=10.,sum_ch1=5.))
            pd.testing.assert_frame_equal(t,u[t.columns])
            self.assertTrue((run(cells.iloc[:0],areas)["Red+"]==0).all())
            cells["pos_ch0"] = False
            filtered = run(cells,areas)
            self.assertTrue((filtered["Red+"]==0).all())
            self.assertEqual(filtered.loc[filtered.section=="S1","Red 信号中无 DAPI 核的（未计入 Red+）"].tolist(),[1,1])

    def test_duplicate_files_reopen_and_export(self):
        import tifffile
        from cfos.core.export import export_excel
        with TemporaryDirectory() as root:
            root=Path(root)
            p=Project(root/"project")
            paths=[root/group/"01.tif" for group in ("A","B")]
            for i,path in enumerate(paths):
                path.parent.mkdir()
                tifffile.imwrite(path,np.ones((64,64),np.uint16))
                im,_=p.import_image(path)
                self.assertIs(p.import_image(path)[0],im)
                im.sections=[Section(1,(0,0,64,64),[],"S01")]
                p.set_sample(f"{im.key}/S01", animal=str(i),group=str(i))
                df=pd.DataFrame(dict(section=["S01"],merged_id=[0],combo=["0"],x=[10.],y=[10.],channel=[0]))
                p.save_cells(im,df if i==0 else df.iloc[:0])
            p.save()
            q=Project.load(p.folder)
            self.assertNotEqual(q.images[0].key,q.images[1].key)
            self.assertEqual([len(q.load_cells(im)) for im in q.images],[1,0])
            f=export_excel(q,root/"out.xlsx",masks=False,qc=False,quality=False)
            t=pd.read_excel(f,sheet_name="按脑片汇总")
            self.assertEqual(t["动物"].tolist(),[0,1])
            self.assertEqual(t[f"{q.channel_names[0]}+"].tolist(),[1,0])
            # 模拟旧项目共享同一个文件名前缀。
            cfg=q.folder/"cfos_project.json"
            data=json.loads(cfg.read_text("utf8"))
            for row in data["images"]:row.pop("storage_key")
            cfg.write_text(json.dumps(data),"utf8")
            legacy=Project.load(q.folder)
            self.assertTrue(all(legacy.load_cells(im) is None for im in legacy.images))
            self.assertTrue(q.cells_path(q.images[0]).exists())
    def test_cache_reuse_and_invalidation(self):
        from cfos.core import detection, pipeline
        from cfos.core.slide_io import open_slide
        import tifffile
        with TemporaryDirectory() as root:
            root = Path(root)
            path = root/"test.tif"
            tifffile.imwrite(path,np.zeros((64,64),np.uint16))
            project = Project(root/"p")
            im,_ = project.import_image(path)
            sec = Section(1,(0,0,64,64),[],"S01")
            im.sections = [sec]
            project.unmix = {"enabled":False}
            slide = open_slide(path)
            raw = pd.DataFrame(columns=["section","channel","x","y","area_um2","mean_ch0","contour"])
            try:
                with patch.object(detection.cellpose4, "model_identity", return_value={"weights": "A"}) as identity, patch.object(
                        detection,"detect_section",return_value=raw) as inference:
                    def run():
                        return pipeline.detect_image(project,im,sl=slide)
                    self.assertEqual(run()["done"],1)
                    with patch.object(project,"load_part",wraps=project.load_part) as read:
                        self.assertEqual(run()["reused"],1)
                        self.assertEqual(read.call_count,1)
                    sec.contour0 = [(2,2),(60,2),(2,60)]
                    self.assertEqual(run()["done"],1)
                    project.z_mode="mean"
                    self.assertEqual(run()["done"],1)
                    identity.return_value = {"weights": "B"}
                    self.assertEqual(run()["done"],1)
                    project.part_path(im,sec.name).write_bytes(b"broken")
                    self.assertEqual(run()["done"],1)
                    self.assertEqual(inference.call_count,5)
            finally:
                slide.close()
