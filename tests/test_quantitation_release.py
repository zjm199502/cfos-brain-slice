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
            model = root/"model"
            model.mkdir()
            (model/"model.onnx").write_bytes(b"version A")
            try:
                with patch.object(detection,"_model_dir",return_value=model), patch.object(
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
                    (model/"model.onnx").write_bytes(b"modified version B")
                    self.assertEqual(run()["done"],1)
                    project.part_path(im,sec.name).write_bytes(b"broken")
                    self.assertEqual(run()["done"],1)
                    self.assertEqual(inference.call_count,5)
            finally:
                slide.close()
