"""Small real ZISRAW containers keep geometry, codec and metadata tests offline."""
from contextlib import closing
from pathlib import Path
import struct
import tempfile
import unittest

import imagecodecs
import numpy as np

from cfos.core.czi import list_czi_scenes
from cfos.core.slide_io import open_slide


def write_czi(path, planes, *, metadata=None, compression=0, file_part=0):
    """planes are (scene, channel, z, absolute x, absolute y, YX ndarray[, time])."""
    def segment(sid, body):
        padding = (-len(body)) % 32
        return struct.pack("<16sqq", sid, len(body) + padding, len(body)) + body + bytes(padding)

    if metadata is None:
        channels = sorted({p[1] for p in planes})
        seq = "".join(f'<Channel Id="Channel:{c}" Name="dye{c}"><Color>#FF00FF55</Color></Channel>'
                      for c in reversed(channels))
        metadata = (f'<ImageDocument><Metadata><Information><Image><SizeC>{len(channels)}</SizeC>'
                    f'<Dimensions><Channels>{seq}</Channels></Dimensions></Image></Information>'
                    '<Scaling><Items><Distance Id="X"><Value>5e-7</Value></Distance>'
                    '<Distance Id="Y"><Value>5e-7</Value></Distance></Items></Scaling></Metadata></ImageDocument>')
    xml = metadata.encode()
    meta = segment(b"ZISRAWMETADATA", struct.pack("<ii248x", len(xml), 0) + xml)
    header_size = 32 + 96
    offset = header_size + len(meta)
    blocks, entries = [], []
    mosaic_ids = {}
    for s, c, z, x, y, pixels, *time in planes:
        h, w = pixels.shape
        m = mosaic_ids.setdefault((s, x, y), len(mosaic_ids))
        dims = [(b"X", x, w), (b"Y", y, h), (b"Z", z, 1), (b"C", c, 1),
                (b"T", time[0] if time else 0, 1), (b"S", s, 1), (b"M", m, 1)]
        entry = struct.pack("<2siqiiBB4xi", b"DV", {1: 0, 2: 1}[pixels.itemsize], offset,
                            file_part, compression, 0, 0, len(dims))
        entry += b"".join(struct.pack("<4siifi", name, start, size, 0., size) for name, start, size in dims)
        raw = pixels.tobytes()
        if compression == 4:
            raw = imagecodecs.jpegxr_encode(pixels)
        elif compression == 5:
            raw = imagecodecs.zstd_encode(raw)
        elif compression == 6:
            raw = imagecodecs.zstd1_encode(raw, itemsize=pixels.itemsize, hilo=True)
        block = segment(b"ZISRAWSUBBLOCK", struct.pack("<iiq", 0, 0, len(raw)) + entry.ljust(240, b"\0") + raw)
        entries.append(entry)
        blocks.append(block)
        offset += len(block)
    directory = segment(b"ZISRAWDIRECTORY", struct.pack("<i124x", len(entries)) + b"".join(entries))
    header = segment(b"ZISRAWFILE", struct.pack("<iiii16s16siqqiq", 1, 0, 0, 0, bytes(16),
                                               bytes(16), 0, offset, header_size, 0, 0))
    Path(path).write_bytes(b"".join([header, meta, *blocks, directory]))


class CziReaderTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "synthetic.czi"
        self.a = (np.arange(35, dtype=np.uint16).reshape(5, 7) + 500)

    def test_multiple_scenes_origin_channels_z_and_overlap(self):
        planes = []
        for scene in [0, 7]:
            for c in (1, 2):
                for z in (3, 4):
                    val = self.a + (scene + c + z) * 100
                    planes += [(scene, c, z, -33, 17000, val), (scene, c, z, -28, 17000, val + 50)]
        write_czi(self.path, planes)
        self.assertEqual([s.scene_id for s in list_czi_scenes(self.path)], [0, 7])
        with self.assertRaisesRegex(ValueError, "多个扫描区"):
            open_slide(self.path)
        with closing(open_slide(self.path, "max", scene_id=7)) as slide:
            self.assertEqual([c.name for c in slide.channels], ["dye1", "dye2"])
            self.assertEqual(slide.channels[1].file_color, (0, 255, 85))
            self.assertEqual(slide.level_shapes, [(5, 12)])
            self.assertEqual(slide.pixel_size_um, 0.5)
            expected = np.column_stack((self.a[:, :5] + 1300, self.a + 1350))
            self.assertEqual(slide.read(0).dtype, np.uint16)
            np.testing.assert_array_equal(slide.read(0, (3, 1, 6, 3), channels=[1, 0]),
                                          np.stack([expected, expected-100])[:, 1:4, 3:9])
        with closing(open_slide(self.path, "z1", scene_id=7)) as slide:
            np.testing.assert_array_equal(slide.read(0, (0, 0, 5, 5), [1])[0], self.a[:, :5]+1200)

    def test_supported_lossless_codecs_preserve_uint16(self):
        plane = np.repeat(np.repeat(self.a, 8, axis=1), 8, axis=0)
        for compression in (0, 4, 5, 6):
            with self.subTest(compression=compression):
                write_czi(self.path, [(0, 0, 0, -42, -81, plane)], compression=compression)
                with closing(open_slide(self.path)) as slide:
                    np.testing.assert_array_equal(slide.read(0)[0], plane)

    def test_first_absolute_time_and_incomplete_channels(self):
        planes = [(0, c, 0, 0, 0, self.a + 100*t+c, t) for t in (8, 5) for c in (0, 1)]
        write_czi(self.path, planes)
        notes = []
        list_czi_scenes(self.path, notes=notes)
        self.assertIn("T=5", notes[0])
        with closing(open_slide(self.path)) as slide:
            self.assertEqual((slide.czi_time_index, slide.czi_time_count), (5, 2))
            self.assertEqual(notes, slide.notes)
            self.assertIn(5, slide.scene_source_signature())
            np.testing.assert_array_equal(slide.read(0)[1], self.a + 501)
        write_czi(self.path, planes[:-1])
        with self.assertRaisesRegex(ValueError, "缺少通道或 Z"):
            open_slide(self.path)

    def test_incomplete_scale_and_external_part_are_rejected(self):
        planes = [(0, 0, 0, 10, 20, self.a)]
        write_czi(self.path, planes, metadata="<ImageDocument/>")
        with self.assertRaisesRegex(ValueError, "标定不完整"):
            open_slide(self.path)
        write_czi(self.path, planes, file_part=1)
        with self.assertRaisesRegex(ValueError, "多文件分卷"):
            list_czi_scenes(self.path)

    def test_ambiguous_raw_bytes_marked_compressed_are_rejected(self):
        # The fixture deliberately leaves a raw payload for LZW compression=2.
        write_czi(self.path, [(0, 0, 0, 0, 0, self.a)], compression=2)
        with closing(open_slide(self.path)) as slide:
            with self.assertRaisesRegex(ValueError, "压缩标记"):
                slide.read(0)


if __name__ == "__main__":
    unittest.main()
