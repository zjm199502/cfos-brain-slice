"""Equal-size tiles cropped to tissue: the grid, the crop, cell ownership and the worker batch size."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.path import Path as MplPath
from skimage import measure

from cfos.core import cellpose_worker, detection
from cfos.core.project import Project
from cfos.core.sections import Section

# A concave (L-shaped) section inside a 300 × 200 px bounding box.
CONTOUR = [(5, 5), (295, 5), (295, 80), (120, 80), (120, 195), (5, 195)]


def blob_image(centres, shape=(200, 300), radius=3):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.zeros(shape, np.uint16)
    for cx, cy in centres:
        img[(xx-cx)**2 + (yy-cy)**2 <= radius**2] = 1000
    return img


class EqualGridTests(unittest.TestCase):
    def test_edges_split_extent_without_a_sliver(self):
        edges = detection._equal_edges(4300, 2048)
        sizes = np.diff(edges)
        self.assertEqual(len(sizes), 3)
        self.assertEqual((edges[0], edges[-1]), (0, 4300))
        self.assertLessEqual(sizes.max() - sizes.min(), 1)
        self.assertLessEqual(sizes.max(), 2048)
        self.assertEqual(detection._equal_edges(500, 2048), [0, 500])

    def test_cores_partition_the_section_and_fields_use_the_same_grid(self):
        section = Section(1, (10, 40, 150, 65), [], "S")
        tiles = detection.section_tiles(section, 1., 32, equal=True)
        cover = np.zeros((65, 150), int)
        for tx, ty, tw, th in tiles:
            cover[ty:ty+th, tx:tx+tw] += 1
        self.assertTrue((cover == 1).all())
        self.assertEqual(len(tiles), 5 * 3)
        # A preview field is tiled on its own: no core reaches beyond it.
        field = detection.section_tiles(section, 1., 32, field_box=(54, 47, 40, 22), equal=True)
        self.assertEqual(field, [(44, 7, 20, 22), (64, 7, 20, 22)])
        self.assertEqual(detection.section_tiles(section, 1., 32, field_box=(500, 47, 10, 10), equal=True), [])
        legacy = detection.section_tiles(section, 1., 32)
        self.assertTrue(all(tw == th == 32 for _, _, tw, th in legacy))

    def test_tiles_outside_the_contour_are_dropped(self):
        section = Section(1, (0, 0, 300, 200), CONTOUR, "S")
        tiles = detection.section_tiles(section, 1., 64, equal=True)
        poly = MplPath(np.asarray(CONTOUR + [CONTOUR[0]], float))
        for tx, ty, tw, th in tiles:
            self.assertTrue(poly.intersects_bbox(detection.Bbox.from_extents(tx-1, ty-1, tx+tw+1, ty+th+1), filled=True))
        self.assertLess(len(tiles), 5 * 4)

    def test_crop_keeps_all_tissue_of_the_core(self):
        section = Section(1, (0, 0, 300, 200), CONTOUR, "S")
        poly = MplPath(np.asarray(CONTOUR + [CONTOUR[0]], float))
        yy, xx = np.mgrid[:200, :300]
        inside = poly.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(200, 300)
        cropped_any = False
        for tx, ty, tw, th in detection.section_tiles(section, 1., 64, equal=True):
            x0, y0, x1, y1 = detection.tissue_crop(section, 1., (tx, ty, tw, th))
            self.assertTrue(tx <= x0 < x1 <= tx+tw and ty <= y0 < y1 <= ty+th)
            core = np.zeros_like(inside)
            core[ty:ty+th, tx:tx+tw] = True
            kept = np.zeros_like(inside)
            kept[y0:y1, x0:x1] = True
            self.assertFalse((inside & core & ~kept).any())
            cropped_any |= (x1-x0)*(y1-y0) < tw*th
        self.assertTrue(cropped_any)
        # A section without a contour is not cropped; a downsampled grid is also handled.
        self.assertEqual(detection.tissue_crop(Section(1, (0, 0, 300, 200), [], "S"), 1., (0, 0, 60, 50)),
                         (0, 0, 60, 50))
        x0, y0, x1, y1 = detection.tissue_crop(section, 2., (0, 0, 75, 50))
        self.assertEqual((x0, y0), (1, 1))


class OwnershipTests(unittest.TestCase):
    def run_detection(self, centres, equal):
        img = blob_image(centres)
        boxes = []
        def read(level, box, channels=None):
            x, y, w, h = (int(v) for v in box)
            boxes.append((x, y, w, h))
            return img[None, y:y+h, x:x+w]
        slide = SimpleNamespace(channels=[0], pixel_size_um=1., level_for_pixel_size=lambda _: 0,
                                downsample=lambda _: 1., read=read)
        section = Section(1, (0, 0, 300, 200), CONTOUR, "S")
        settings = detection.DetectionSettings(tile=64, overlap=8, equal_tiles=equal,
                                               channels={0: detection.ChannelParams(min_area_um2=0)})
        def predict(x, params, cancel, progress=None):
            return measure.label(x > 0.5).astype(np.int32), "CPU"
        with patch.object(detection, "_pool", return_value=None), \
                patch.object(detection.segmentation, "predict", side_effect=predict):
            df = detection.detect_section(slide, section, settings, norms={0: (0, 1000, 1)}, prefilter=False)
        return df, boxes

    def test_every_cell_is_found_once_with_equal_cropped_tiles(self):
        rng = np.random.default_rng(3)
        poly = MplPath(np.asarray(CONTOUR + [CONTOUR[0]], float))
        centres = []
        while len(centres) < 120:
            c = rng.uniform((12, 12), (288, 188))
            if poly.contains_point(c) and all(np.hypot(*(c - d)) > 9 for d in centres):
                centres.append(np.round(c))
        # Put cells right on the equal-grid and legacy seams too.
        for c in [(60, 40), (128, 40), (180, 64), (60, 128), (64, 100), (192, 50)]:
            if all(np.hypot(*(np.subtract(c, d))) > 9 for d in centres):
                centres.append(np.array(c, float))
        expected = np.array(sorted(map(tuple, centres)))
        equal, equal_boxes = self.run_detection(centres, True)
        legacy, legacy_boxes = self.run_detection(centres, False)
        for df in (equal, legacy):
            found = np.array(sorted(zip(df.x, df.y)))
            self.assertEqual(len(found), len(expected))
            np.testing.assert_allclose(found, expected, atol=1e-6)
        self.assertLess(sum(w*h for *_, w, h in equal_boxes), sum(w*h for *_, w, h in legacy_boxes))


class PreviewFieldTests(unittest.TestCase):
    def test_field_preview_reads_only_the_field_and_finds_its_cells(self):
        centres = [(60., 30.), (90., 40.), (150., 30.), (40., 150.)]
        img = blob_image(centres)
        boxes = []
        def read(level, box, channels=None):
            x, y, w, h = (int(v) for v in box)
            boxes.append((x, y, w, h))
            return img[None, y:y+h, x:x+w]
        slide = SimpleNamespace(channels=[0], pixel_size_um=1., level_for_pixel_size=lambda _: 0,
                                downsample=lambda _: 1., read=read)
        section = Section(1, (0, 0, 300, 200), CONTOUR, "S")
        settings = detection.DetectionSettings(tile=64, overlap=8, channels={0: detection.ChannelParams(min_area_um2=0)})
        def predict(x, params, cancel, progress=None):
            return measure.label(x > 0.5).astype(np.int32), "CPU"
        with patch.object(detection, "_pool", return_value=None), \
                patch.object(detection.segmentation, "predict", side_effect=predict):
            df = detection.detect_section(slide, section, settings, norms={0: (0, 1000, 1)}, prefilter=False,
                                          field_box=(50, 20, 50, 30))
        self.assertEqual(sorted(zip(df.x, df.y)), [(60., 30.), (90., 40.)])
        for x, y, w, h in boxes:
            self.assertTrue(x >= 50-8 and y >= 20-8 and x+w <= 100+8 and y+h <= 50+8)


class SettingsTests(unittest.TestCase):
    def test_signature_and_old_projects(self):
        new = detection.DetectionSettings(channels={0: detection.ChannelParams()})
        old = detection.DetectionSettings(channels={0: detection.ChannelParams()}, equal_tiles=False)
        self.assertEqual(json.loads(detection.raw_signature(new)).get("equal_tiles"), 1)
        self.assertNotIn("equal_tiles", json.loads(detection.raw_signature(old)))
        self.assertTrue(Project(folder=".").detection.equal_tiles)


class BatchSizeTests(unittest.TestCase):
    def test_gpu_batch_override_and_halving(self):
        self.assertEqual(cellpose_worker.initial_batch_size("mps", {}), 8)
        self.assertEqual(cellpose_worker.initial_batch_size("cuda", {}), 8)
        self.assertEqual(cellpose_worker.initial_batch_size("cpu", {}), 4)
        self.assertEqual(cellpose_worker.initial_batch_size("mps", {"CFOS_CELLPOSE_BATCH": "16"}), 16)
        self.assertEqual(cellpose_worker.initial_batch_size("mps", {"CFOS_CELLPOSE_BATCH": "x"}), 8)
        sizes, size = [], 8
        while (size := cellpose_worker.smaller_batch(size, "MPS backend out of memory")) is not None:
            sizes.append(size)
            if size == 1:
                self.assertIsNone(cellpose_worker.smaller_batch(1, "out of memory"))
                break
        self.assertEqual(sizes, [4, 2, 1])
        self.assertIsNone(cellpose_worker.smaller_batch(8, "not implemented for MPS"))


if __name__ == "__main__":
    unittest.main()
