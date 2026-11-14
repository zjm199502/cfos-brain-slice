"""生成一个小的合成 .kfbf 文件（按 cfos/core/kfbf.py 里写明的格式），检查读取、拼图、金字塔和看图视图。
用法：python tests/kfbf_synth.py <输出文件夹>"""
import struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import imagecodecs


def write_kfbf(path, img, names, px_um=0.5, mag=20, T=256):
    """img: (C, H, W) uint8，最高倍率层。写 3 层金字塔，跳过全黑的图块。"""
    C, H, W = img.shape
    levels, a = [], img
    for k in range(3):
        levels.append((mag / 2 ** k, a))
        a = a[:, ::2, ::2]
    head = bytearray(0x400)
    head[0:4] = b"\xf1\x01\xee\xee"; head[4:8] = b"KFBF"; head[0x20:0x24] = b"JPEG"
    struct.pack_into("<f", head, 0x4c, px_um)
    # ff01eeee TLV：通道数、通道名表位置
    names_at = 0x200
    p = 0x5c
    head[p:p + 4] = b"\xff\x01\xee\xee"; struct.pack_into("<I", head, p + 4, 2)
    struct.pack_into("<IIIII", head, p + 8, 0x4b, 4, C, 0, 0)
    struct.pack_into("<IIQ", head, p + 20, 0x4d, 8, names_at)
    for i, n in enumerate(names):
        b = n.encode()[:39]; head[names_at + 40 * i:names_at + 40 * i + len(b)] = b
    head[0x300:0x304] = b"\xf1\x02\xee\xee"
    entries, blobs = [], bytearray()
    for sc, a in levels:
        _, h, w = a.shape
        for y in range(0, h, T):
            for x in range(0, w, T):
                t = a[:, y:y + T, x:x + T]
                if t.max() == 0:
                    continue
                js = [imagecodecs.jpeg8_encode(np.ascontiguousarray(t[c]), level=95) for c in range(C)]
                entries.append((x, y, t.shape[2], t.shape[1], sc, js))
    idx0 = len(head)
    tables0 = idx0 + 64 * len(entries)
    out = bytearray(head) + bytearray(64 * len(entries))
    data = bytearray()
    for k, (x, y, w, h, sc, js) in enumerate(entries):
        po = tables0 + len(data); data += b"\0" * (8 * C)
        pz = tables0 + len(data); data += b"\0" * (8 * C)
        offs, szs = [], []
        for j in js:
            offs.append(tables0 + len(data)); szs.append(len(j)); data += j
        struct.pack_into(f"<{C}Q", data, po - tables0, *offs)
        struct.pack_into(f"<{C}Q", data, pz - tables0, *szs)
        e = idx0 + 64 * k
        out[e:e + 4] = b"\xf1\x04\xee\xee"
        struct.pack_into("<4if3iQQ", out, e + 4, x, y, w, h, sc, 0, 0, len(js[0]), po, pz)
    out += data
    struct.pack_into("<4I", out, 0x10, len(entries), H, W, mag)
    struct.pack_into("<Q", out, 0x44, idx0)
    Path(path).write_bytes(bytes(out))


if __name__ == "__main__":
    from cfos.core.slide_io import open_slide
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    H, W = 1500, 2300
    img = np.zeros((3, H, W), np.uint8)
    yy, xx = np.mgrid[:H, :W]
    tissue = ((yy - 750) / 600) ** 2 + ((xx - 1100) / 900) ** 2 < 1
    for c in range(3):
        img[c][tissue] = 30 + 20 * c
        for _ in range(400):
            y, x = rng.integers(0, H), rng.integers(0, W)
            if tissue[y, x]:
                img[c, max(0, y - 4):y + 4, max(0, x - 4):x + 4] = 200
    f = out / "合成.kfbf"
    write_kfbf(f, img, ["DAPI", "520", "670"])
    sl = open_slide(f)
    print("channels", [(c.name, c.color) for c in sl.channels], "px", sl.pixel_size_um, "levels", sl.level_shapes)
    assert [c.name for c in sl.channels] == ["DAPI", "520", "670"]
    assert sl.level_shapes[0] == (H, W)
    a = sl.read(0)
    err = np.abs(a.astype(int) - img.astype(int))
    print("max err (JPEG)", err.max(), "mean", err.mean())
    assert err.mean() < 2
    b = sl.read(1, (300, 200, 900, 700), [2])
    assert b.shape == (1, 350, 450), b.shape
    ch = sl._arr(0)[(1, slice(None), slice(None))]
    assert ch.shape == (H, W) and ch[100:200, 300:400].shape == (100, 100)
    assert np.array_equal(ch[100:200, 300:400], a[1, 100:200, 300:400])
    from cfos.core.sections import find_sections
    secs = find_sections(sl, min_area_mm2=0.1)
    print("sections", [(s.name, s.bbox0) for s in secs])
    assert len(secs) == 1
    sl.close()
    print("OK")
