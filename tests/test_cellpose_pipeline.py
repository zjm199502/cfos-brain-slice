import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cfos.core import accuracy, cellpose4, detection as det, finetune
from cfos.core.pipeline import detect_image, recompute_image_coloc
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section


class CellposePipelineTests(unittest.TestCase):
    def test_old_project_and_template_migrate_settings_only(self):
        with TemporaryDirectory() as folder:
            project = Project(Path(folder), channel_names=["Cy5"])
            project.detection.channels = {0: det.ChannelParams()}
            project.save()
            config = project.folder / "cfos_project.json"
            legacy = {"model": "Cy5_ft_historical", "prob_thresh": 0.9, "min_area_um2": 57}
            state = json.loads(config.read_text(encoding="utf8"))
            state["detection"]["channels"]["0"] = legacy
            config.write_text(json.dumps(state), encoding="utf8")
            old_config = config.read_bytes()
            restored = Project.load(project.folder)
            self.assertEqual(config.read_bytes(), old_config)
            for source in (restored, project):
                if source is project:
                    template = source.template_dict()
                    template["detection"]["channels"]["0"] = legacy
                    path = project.folder / "template.json"
                    path.write_text(json.dumps(template), encoding="utf8")
                    source.load_template(path)
                p = source.detection.channels[0]
                self.assertEqual((p.model, p.cellprob_threshold, p.min_area_um2),
                                 (cellpose4.MODEL_KEY, 0., 57))
                self.assertNotIn("prob_thresh", p.to_dict())

    def test_native_labels_on_edge_measured_once_and_match_preview(self):
        image = np.zeros((2, 32, 64), np.uint16)
        image[0] = 41
        image[1, 10:16, 29:37] = 800
        def read(level, box):
            x, y, w, h = map(int, box)
            return image[:, y:y+h, x:x+w]
        slide = SimpleNamespace(channels=[0, 1], pixel_size_um=1., level_for_pixel_size=lambda _: 0,
                                downsample=lambda _: 1., read=read)
        section = Section(1, (0,0,64,32), [], "S")
        p = det.ChannelParams(cellprob_threshold=-0.75)
        settings = det.DetectionSettings(tile=32, overlap=8, channels={0:det.ChannelParams(detect=False),1:p})
        def predict(plane, cellprob_threshold, cancel):
            self.assertEqual(cellprob_threshold, -0.75)
            return label(plane > 0.5)[0]
        with patch.object(det, "_pool", return_value=None), patch.object(cellpose4, "predict_labels", predict):
            full = det.detect_section(slide, section, settings, norms={1:(0.,800.,1.)})
            field = det.detect_section(slide, section, settings, norms={1:(0.,800.,1.)}, field_box=(31,9,8,8))
        pd.testing.assert_frame_equal(full, field)
        self.assertEqual(len(full), 1)
        row = full.iloc[0]
        self.assertEqual((row.channel,row.x,row.y,row.area_um2,row.mean_ch0,row.mean_ch1),
                         (1,32.5,12.5,48.,41.,800.))
        self.assertGreater(len(row.contour), 3)
        self.assertEqual(full.attrs["channel_models"], {1:cellpose4.MODEL_KEY})

    def test_partial_runs_and_coloc_retain_historical_model_attribution(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/"slide.tif"
            path.touch()
            a,b = [Section(i,(i*50,0,50,50),[],name) for i,name in enumerate(["A","B"])]
            im = ImageEntry(str(path),sections=[a,b])
            project = Project(Path(folder),images=[im],channel_names=["Cy5"])
            project.detection.channels = {0:det.ChannelParams()}
            columns = ["section","channel","x","y","area_um2","mean_ch0"]
            old = pd.DataFrame([["B",0,65.,15.,36.,700.]],columns=columns)
            old["cell_id"] = old["merged_id"] = 0
            old["combo"] = "0"
            old.attrs.update(processed_sections=["B"], section_models={"B":{0:"old-ft-model"}})
            project.save_cells(im,old)
            raw = old.assign(section="A",x=15.)
            raw.attrs = {"channel_models":{0:cellpose4.MODEL_KEY}}
            raw.attrs["inference_settings"] = det.raw_signature(project.detection)
            raw.attrs["z_mode"] = "max"
            with patch.object(cellpose4, "model_identity", return_value={"version":"4.2"}), \
                    patch.object(det,"detect_section",return_value=raw) as infer:
                def run():
                    return detect_image(project, im, sl=SimpleNamespace(pixel_size_um=1.), section_names=["A"])
                self.assertEqual(run()["done"],1)
                self.assertEqual(run()["reused"],1)
                project.detection.channels[0].cellprob_threshold = .8
                self.assertEqual(run()["done"],1)
                self.assertEqual(infer.call_count,2)
            expected = {"A":{0:cellpose4.MODEL_KEY},"B":{0:"old-ft-model"}}
            self.assertEqual(project.load_cells(im).attrs["section_models"],expected)
            self.assertEqual(recompute_image_coloc(project, im, 1.).attrs["section_models"],expected)
            saved = project.load_cells(im)
            project.detection.channels[0].min_area_um2 = 999.
            box = (0,0,50,50)
            checked = accuracy.check(saved[saved.section == "A"],0,box,[],1.)
            fresh = accuracy.record(project,im,"A",0,box,checked,"Cellpose 4",cells=saved)
            historical = accuracy.record(project,im,"B",0,box,checked,"old-ft-model",cells=saved)
            self.assertEqual(json.loads(fresh["检查时参数"])["通道筛选"]["min_area_um2"],20.)
            self.assertEqual(json.loads(historical["检查时参数"])["识别参数"],"历史结果未记录")

    def test_annotation_patch_uses_cellpose_labels(self):
        with TemporaryDirectory() as folder:
            p = Project(Path(folder),channel_names=["Cy5"])
            p.detection.channels = {0:det.ChannelParams(cellprob_threshold=.6)}
            sl = SimpleNamespace(channels=[0],pixel_size_um=1.,level_for_pixel_size=lambda _:0,
                                 downsample=lambda _:1.,level_shapes=[(256,256)],
                                 read=lambda level,box,chs: np.ones((1,256,256)))
            mask = np.zeros((256,256),np.int32)
            mask[88:94,6:11] = 3
            with patch.object(cellpose4, "predict_labels",return_value=mask) as predict:
                _, labels, meta = finetune.make_patch(p,sl,ImageEntry("slide"),(128,128),0)
            np.testing.assert_array_equal(labels,mask)
            self.assertEqual(predict.call_args.kwargs["cellprob_threshold"],.6)
            self.assertEqual(meta["model"],cellpose4.MODEL_KEY)
