from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

from cfos.core.sections import _tissue_image
from cfos.core.slide_io import open_slide


class EfiScanTests(unittest.TestCase):
    def test_namespaced_ome_channel_roles(self):
        xml = tifffile.OmeXml()
        xml.addimage(dtype=np.uint16, shape=(3, 8, 9), storedshape=(3, 1, 1, 8, 9, 1),
                     axes="CYX", Channel={"Name": ["DAPI", "FITC", "Cy5"]})
        # Olympus exports may prefix child tags even with a default namespace.
        desc = str(xml).replace('<OME ', '<OME xmlns:o="http://www.openmicroscopy.org/Schemas/OME/2016-06" ', 1)
        desc = desc.replace('<Channel ', '<o:Channel ').replace('</Channel>', '</o:Channel>')
        with TemporaryDirectory() as folder:
            path = Path(folder) / "scan.tif"
            tifffile.imwrite(path, np.zeros((3, 8, 9), np.uint16), description=desc,
                             metadata=None, photometric="minisblack")
            slide = open_slide(path)
            try:
                self.assertEqual([c.name for c in slide.channels], ["DAPI", "FITC", "Cy5"])
            finally:
                slide.close()

    def test_banded_means_at_native_resolution(self):
        h, w = 2003, 4001
        a = np.broadcast_to(np.arange(w, dtype=np.uint16), (h, w))
        class Slide:
            level_shapes = [(h, w)]
            pixel_size_um = 1
            channels = [0]
            def level_for_pixel_size(self, target): return 0
            def downsample(self, level): return 1
            def read(self, level, box, channels):
                x, y, bw, bh = box
                return a[None, y:y+bh, x:x+bw]
        expected = a[:h//8*8,:w//8*8].astype(np.float32).reshape(h//8,8,w//8,8).mean((1,3))
        for selection in ([0], None):
            with self.subTest(channels=selection):
                thumb, ds = _tissue_image(Slide(), selection)
                self.assertEqual(ds, 8)
                np.testing.assert_array_equal(thumb[0], expected)
