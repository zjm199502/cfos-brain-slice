"""v0.8.4：奥林巴斯 .vsi 读取测试（不需要界面）。
1. 真实样本（tests/data/vsi，cellSens 3.1，2 通道 × 11 层 z-stack，16 位，JPEG 2000）：通道名、像素尺寸、
   z-stack 合成方式、看图用的按需读取视图；
2. 合成 .ets（按 cfos/core/vsi.py 里写明的格式）：JPEG / 不压缩 / JPEG 2000 图块、金字塔、拼接原点偏移、
   缺失图块、(Z, 通道) 两个维度；.vsi 说明信息读不懂时仍能靠 .ets 读图并给出提示；
3. 出错提示：缺少“_名字_”文件夹、明场 RGB、直接选 .ets。
用法：python tests/v084_vsi.py"""
import struct, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import imagecodecs
from cfos.core.slide_io import open_slide, guess_role
from cfos.core import pipeline, relocate

DATA = Path(__file__).parent / "data" / "vsi"


def write_ets(path, planes, comp=2, T=64, shift=(0, 0), ncomp=1, skip=()):
    """planes: {(z, c): (H, W) 数组}（第 0 层，已含原点偏移的存储网格）。坐标 = (tx, ty, z, c, level)。"""
    any_ = next(iter(planes.values()))
    dt = any_.dtype
    Hs, Ws = any_.shape[:2]
    nz = max(k[0] for k in planes) + 1; nc = max(k[1] for k in planes) + 1
    W, H = Ws - shift[0], Hs - shift[1]
    recs, blob = [], bytearray()
    base = 64 + 228
    lv, cur = 0, planes
    while True:
        h, w = next(iter(cur.values())).shape[:2]
        for (z, c), a in cur.items():
            for ty in range(-(-h // T)):
                for tx in range(-(-w // T)):
                    if (tx, ty, z, c, lv) in skip:
                        continue
                    t = np.zeros((T, T) + a.shape[2:], dt)
                    p = a[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T]
                    t[:p.shape[0], :p.shape[1]] = p
                    raw = (t.astype("<" + dt.str[1:]).tobytes() if comp == 0 else
                           imagecodecs.jpeg8_encode(t, level=95) if comp == 2 else imagecodecs.jpeg2k_encode(t, level=0))
                    recs.append(((tx, ty, z, c, lv), base + len(blob), len(raw))); blob += raw
        if h <= T and w <= T:
            break
        cur = {k: a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2].reshape(
            (a.shape[0] // 2, 2, a.shape[1] // 2, 2) + a.shape[2:]).mean((1, 3)).astype(dt) for k, a in cur.items()}
        lv += 1
    ndim = 5
    tabpos = base + len(blob)
    main = struct.pack("<4sIIIQIIQIIQII", b"SIS\0", 64, 3, ndim, 64, 228, 0, tabpos, len(recs), 0, 0, 0, 0)
    sec = bytearray(228)
    struct.pack_into("<4s9I", sec, 0, b"ETS\0", 0x30006, 2 if dt == np.uint8 else 4, ncomp, 1 if ncomp == 1 else 4,
                     comp, 90, T, T, 1)
    struct.pack_into("<I", sec, 152, 1)
    struct.pack_into("<ii", sec, 172, -shift[0], -shift[1])
    struct.pack_into("<5I", sec, 184, 4, W, H, nz, nc)
    tab = b"".join(struct.pack("<I5iQII", ndim, *c, off, ln, 0) for c, off, ln in recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(main + bytes(sec) + bytes(blob) + tab)


def smooth(rng, shape, dt):
    from scipy import ndimage as ndi
    a = ndi.gaussian_filter(rng.random(shape), 6)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * (250 if dt == np.uint8 else 4000)).astype(dt)


def test_real():
    p = DATA / "vsi-ets-test-jpg2k.vsi"
    s = open_slide(p, "max")
    assert [c.name for c in s.channels] == ["C405", "C488"], s.channels
    assert [guess_role(c.name) for c in s.channels] == ["DAPI", "Green"]
    assert abs(s.pixel_size_um - 0.108333) < 1e-5 and s.n_z == 11 and s.level_shapes == [(1682, 1645)]
    mx = s.read(0)
    assert mx.shape == (2, 1682, 1645) and mx.dtype == np.uint16
    planes = []
    for z in range(11):
        sz = open_slide(p, f"z{z + 1}"); planes.append(sz.read(0)); sz.close()
    planes = np.stack(planes)
    assert np.array_equal(mx, planes.max(0)), "最大投影应等于各层逐像素最大值"
    # 和 JPEG 2000 码流直接解码的结果一致（维度顺序是 T、Z、通道）
    kf = s._kf
    for (z, c) in [(0, 0), (4, 1), (10, 0)]:
        off, ln = kf.index.tiles[(0, 0, 0, z, c, 0)]
        ref = imagecodecs.jpeg2k_decode(open(kf.index.path, "rb").read()[off:off + ln])
        assert np.array_equal(planes[z, c], ref[:1682, :1645]), (z, c)
    sm = open_slide(p, "mean"); assert np.array_equal(sm.read(0), planes.mean(0).astype(np.uint16)); sm.close()
    se = open_slide(p, "edf"); e = se.read(0, (200, 300, 512, 400)); se.close()
    assert e.shape == (2, 400, 512) and (e <= mx[:, 300:700, 200:712]).all()
    # 区域读取 = 整幅图的切片；看图用的按需视图
    assert np.array_equal(s.read(0, (100, 50, 300, 200), [1]), mx[1:2, 50:250, 100:400])
    v = s._arr(0)[1]
    assert v.shape == (1682, 1645) and np.array_equal(np.asarray(v[10:99, 5:77]), mx[1, 10:99, 5:77])
    s.close()
    print("真实样本 OK：", [c.name for c in open_slide(p).channels], "11 层 z-stack，最大 / 平均 / 景深合成 / 指定层")


def test_synth(td):
    rng = np.random.default_rng(1)
    for comp, dt, tol in ((2, np.uint8, 6), (0, np.uint16, 0), (3, np.uint16, 0)):
        H, W, sh = 300, 437, (15, 1)
        planes = {(z, c): smooth(rng, (H + sh[1], W + sh[0]), dt) for z in range(2) for c in range(3)}
        d = td / f"c{comp}"
        vsi = d / "片子 A.vsi"
        skip = {(1, 1, 0, 2, 0)}
        write_ets(d / "_片子 A_" / "stack10001" / "frame_t_0.ets", planes, comp, shift=sh, skip=skip)
        write_ets(d / "_片子 A_" / "stack1" / "frame_t.ets", {(0, 0): smooth(rng, (40, 50), np.uint8)}, 2)   # 小的标签图
        vsi.write_bytes(b"II*\0" + b"\0" * 200)          # 读不懂的说明信息
        s = open_slide(vsi, "max")
        assert len(s.channels) == 3 and s.n_z == 2 and s.level_shapes[0] == (H, W), (s.channels, s.n_z, s.level_shapes)
        assert s.n_levels == 4 and s.level_shapes[1] == (150, 218)
        assert any("说明信息" in n for n in s.notes) and any("像素尺寸" in n for n in s.notes), s.notes
        a = s.read(0)
        ref = np.stack([np.maximum(planes[(0, c)], planes[(1, c)])[sh[1]:, sh[0]:] for c in range(3)])
        hole = np.zeros((H, W), bool); hole[64 - sh[1]:128 - sh[1], 64 - sh[0]:128 - sh[0]] = True   # 缺失图块（通道 3、z 1）
        ref[2][hole] = planes[(1, 2)][sh[1]:, sh[0]:][hole]
        assert a.dtype == dt and np.abs(a.astype(int) - ref.astype(int)).max() <= tol, np.abs(a.astype(int) - ref).max()
        b = s.read(0, (33, 70, 180, 120), [2, 0])
        assert np.array_equal(b, a[[2, 0], 70:190, 33:213])
        l1 = s.read(1)
        assert l1.shape == (3, 150, 218)
        full = np.stack([np.maximum(planes[(0, c)], planes[(1, c)]) for c in range(3)]).astype(float)
        m = full[:, :300, :452].reshape(3, 150, 2, 226, 2).mean((2, 4))[:, :, 7:7 + 218]
        assert np.abs(l1[:2].astype(float) - m[:2]).mean() < (3 if dt == np.uint8 else 40), np.abs(l1 - m).mean()
        s.close()
    print("合成 .ets OK：JPEG / 不压缩 / JPEG 2000，金字塔 4 层，原点偏移 (15, 1)，缺失图块，无说明信息时的提示")
    return td / "c2" / "片子 A.vsi"


def test_errors(td, vsi):
    lone = td / "lone" / "只有vsi.vsi"; lone.parent.mkdir(); lone.write_bytes(b"II*\0" + b"\0" * 64)
    for path, word in ((lone, "像素文件夹"), (vsi.parent / "_片子 A_" / "stack10001" / "frame_t_0.ets", "请选择 .vsi")):
        try:
            open_slide(path); raise SystemExit(f"应当报错：{path}")
        except ValueError as e:
            assert word in str(e), e
    d = td / "rgb"
    write_ets(d / "_明场_" / "stack10001" / "frame_t.ets",
              {(0, 0): np.full((64, 64, 3), 200, np.uint8)}, 2, ncomp=3)
    (d / "明场.vsi").write_bytes(b"II*\0" + b"\0" * 64)
    try:
        open_slide(d / "明场.vsi"); raise SystemExit("明场 RGB 应当报错")
    except ValueError as e:
        assert "明场" in str(e), e
    assert vsi in pipeline.list_images(vsi.parent) and ".vsi" in relocate.IMAGE_EXT
    print("出错提示 OK：缺少像素文件夹 / 直接选 .ets / 明场 RGB；批量处理能找到 .vsi")


if __name__ == "__main__":
    test_real()
    with tempfile.TemporaryDirectory() as td:
        v = test_synth(Path(td))
        test_errors(Path(td), v)
    print("v0.8.4 .vsi 测试全部通过")
