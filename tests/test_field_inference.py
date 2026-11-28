from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cfos.core import detection, field_review
from cfos.core.sections import Section


class FieldInferenceTests(unittest.TestCase):
    def test_comparison_shares_raw_candidates_only_with_same_inference_settings(self):
        slide = SimpleNamespace(pixel_size_um=1., level_for_pixel_size=lambda _: 0, downsample=lambda _: 1.)
        sec = Section(1, (0, 0, 100, 100), [], "S1")
        settings = detection.DetectionSettings(channels={0: detection.ChannelParams(min_contrast=1.5)})
        raw = pd.DataFrame(dict(section=["S1"]*3, channel=[0]*3, area_um2=[50.]*3, mean_ch0=[100.]*3,
                                contrast=[1.6, 1.4, 2.], x=[25., 40., 90.], y=[25., 40., 90.]))
        for deconv, calls in [(False, 1), (True, 2)]:
            with self.subTest(deconv=deconv), patch.object(detection, "section_norms", return_value={}), \
                    patch.object(detection, "detect_section", return_value=raw) as infer:
                scores, _, gained, _ = field_review.compare(
                    slide, sec, settings, 0, (0, 0, 70, 70), np.array([[25.,25.]]),
                    {"min_contrast": 1.3, "deconv": deconv}, lambda: False)
                self.assertEqual(infer.call_count, calls)
                self.assertEqual([len(frame) for _, frame in scores], [1, 2])
                self.assertEqual(len(gained), 1)

    def test_field_tiles_keep_section_origin_overlap_and_center_inclusion(self):
        boxes = []
        def read(level, box):
            boxes.append(box)
            return np.ones((1, int(box[3]), int(box[2])), np.uint16)
        slide = SimpleNamespace(channels=[0], pixel_size_um=1., level_for_pixel_size=lambda _: 0,
                                downsample=lambda _: 1., read=read)
        section = Section(1, (10, 40, 150, 65), [], "S")
        settings = detection.DetectionSettings(tile=32, overlap=8, channels={0: detection.ChannelParams()},
                                               equal_tiles=False)
        def measure(labels, img, ch, tile, geo, crit):
            tx, ty, _, _, _, _ = tile
            return [dict(x=geo[0]+tx+12, y=geo[1]+ty+12, channel=0)]
        with patch.object(detection, "_pool", return_value=None), patch.object(
                detection.cellpose4, "predict_labels", return_value=np.zeros((1,1), np.int32)), \
                patch.object(detection, "_measure_tile", side_effect=measure):
            all_cells = detection.detect_section(slide, section, settings, norms={0:(0, 1, 1)})
            all_boxes = set(boxes)
            boxes.clear()
            field = detection.detect_section(slide, section, settings, norms={0:(0, 1, 1)}, field_box=(54, 47, 32, 22))
        self.assertEqual(len(boxes), 2)
        self.assertTrue(set(boxes).issubset(all_boxes))
        self.assertEqual(field.x.tolist(), [54])
        self.assertEqual(field.y.tolist(), [52])
        self.assertEqual(len(all_cells), 15)

    def test_scalar_norm_cache_invalidates_on_contour_and_percentile_change(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "example.tif"
            image.touch()
            slide = SimpleNamespace(path=image, z_mode="max", pixel_size_um=.5)
            sec = Section(1, (0, 0, 128, 128), [], "S")
            c = detection.ChannelParams()
            cache = {}
            with patch.object(detection, "section_norms", return_value={0:(1, 99, 20)}) as compute:
                for section, channel in [(sec,c), (sec,replace(c,min_contrast=1.5)),
                                         (sec,replace(c,norm_high=98)),
                                         (replace(sec,contour0=[(1,1),(5,1),(5,5)]),c)]:
                    detection.preview_norms(slide, section, {0:channel}, None, cache)
                self.assertEqual(compute.call_count, 3)
