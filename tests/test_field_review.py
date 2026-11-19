import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from cfos.core import detection
from cfos.core.field_review import effective_settings, section_key, pooled_reviews
from cfos.core.project import Project, ImageEntry
from cfos.core.sections import Section


class FieldReviewTests(unittest.TestCase):
    def test_persistent_per_section_and_channel(self):
        with TemporaryDirectory() as folder:
            project = Project(Path(folder))
            a = Section(1,(0,0,40,40),[],"A")
            b = Section(2,(40,0,40,40),[],"B")
            image = ImageEntry("sample.tif",sections=[a,b])
            project.images.append(image)
            project.detection.channels = {i:detection.ChannelParams(deconv=True,min_contrast=1.5) for i in (0,1)}
            image.review_settings[section_key(a)] = {"1":{"deconv":False,"min_contrast":1.3}}
            project.save()
            q = Project.load(project.folder)
            im = q.images[0]
            first = effective_settings(q.detection,im,a)
            self.assertFalse(first.channels[1].deconv)
            self.assertTrue(first.channels[0].deconv)
            self.assertTrue(effective_settings(q.detection,im,b).channels[1].deconv)
            self.assertNotEqual(detection.raw_signature(first),detection.raw_signature(q.detection))

    def test_repeated_negative_field(self):
        rec = {"位置标识":"A","通道":"Cy5","当前":{},"候选":{},"识别条件":[],"视野":[0,0,100,100],
               "人工点":[],"面积mm2":0.04,"当前计数":{"tp":0,"fp":0,"fn":0},"候选计数":{"tp":0,"fp":2,"fn":0}}
        text = pooled_reviews([rec,rec],rec)
        self.assertIn("1 个不同框",text)
        self.assertIn("50.0",text)

    def test_exact_banded_normalization(self):
        rng = np.random.default_rng(21)
        raw = rng.integers(0,65535,(2,610,128),dtype=np.uint16)
        sec = Section(1,(0,0,128,610),[(4,90),(95,20),(120,480),(18,590)])
        ch = {i:detection.ChannelParams() for i in range(2)}
        def read(box):
            x,y,w,h=box
            return raw[:,y:y+h,x:x+w]
        for contour in (sec.contour0,[(130,0),(140,0),(140,5)]):
            sec.contour0=contour
            mask=detection.section_mask(sec,raw.shape[1:],1)
            values=detection._band_norms(sec,ch,read)
            for i in ch:
                expected=np.percentile(raw[i][mask] if mask.any() else raw[i],[1,99.8,50])
                np.testing.assert_allclose(values[i],expected)
