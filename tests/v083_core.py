"""v0.8.3 核心功能测试（不需要界面）。
用法：python tests/v083_core.py <示例图片 ome.tif> <已识别并配准的项目文件夹> [qptiff 图片]

1. 通道染色判断：先看名字里的染料 / 波长，文件里的显示颜色只作最后参考；显示颜色默认沿用显微镜颜色（3.8）
2. 读图多线程解码 + 图块缓存：与原来的读法逐像素一致（2.8）
3. X+ 占 DAPI+ 的比例：导出表、组间统计指标、提示列（1.8）
4. 脑区平均荧光强度按去除串色后的图计算（1.6）
5. 质控图 / 截图用自选颜色（3.8）
"""
import os, sys, json, shutil, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def check_roles():
    from cfos.core.slide_io import guess_role, ChannelInfo
    from cfos.core.project import Project
    cases = {("DAPI", None): "DAPI", ("620", (255, 0, 0)): "Red", ("AF647", (255, 255, 0)): "Far red",
             ("Cy5", None): "Far red", ("Alexa Fluor 488", None): "Green", ("670", None): "Far red",
             ("520", None): "Green", ("Far Red", (255, 255, 0)): "Far red", ("CH3", (255, 255, 0)): None,
             ("CH2", (0, 255, 0)): "Green", ("NeuN", None): None, ("Hoechst 33342", None): "DAPI",
             ("TRITC", None): "Red", ("Opal 570", None): "Red", ("PE-Cy5", None): "Far red"}
    for (n, c), want in cases.items():
        got = guess_role(n, c)
        assert got == want, (n, c, got, want)
    # 显微镜把 Far Red 存成黄色：名字里有 647 → 认成 Far red，显示颜色仍是黄色
    tmp = Path(tempfile.mkdtemp())
    p = Project(folder=tmp / "p")
    chans = [ChannelInfo(0, "DAPI", (0, 0, 255), (0, 0, 255)), ChannelInfo(1, "AF488", (0, 255, 0), (0, 255, 0)),
             ChannelInfo(2, "AF647", (255, 255, 0), (255, 255, 0)), ChannelInfo(3, "通道4", (255, 255, 0), (255, 255, 0))]
    p.ensure_channels(chans)
    assert p.channel_names == ["DAPI", "Green", "Far red", "通道4"], p.channel_names
    assert p.custom_color(2) == (255, 255, 0) and p.custom_color(0) == (0, 0, 255)
    p.save()
    p2 = Project.load(p.folder)
    assert p2.custom_color(2) == (255, 255, 0)
    # 旧版本软件（DisplayChannel 没有 color 字段）也要能打开新项目文件
    dd = json.loads((p.folder / "cfos_project.json").read_text(encoding="utf-8"))
    assert all("color" not in v for v in dd["display"].values()) and dd["display_colors"]["2"] == [255, 255, 0]
    p2.set_color(2, None); assert Project.load(p.folder).custom_color(2) is None
    # 旧项目（display 里没有 color）照样能打开，颜色按通道名
    d = json.loads((p.folder / "cfos_project.json").read_text(encoding="utf-8"))
    d.pop("display_colors", None)
    (p.folder / "cfos_project.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    p3 = Project.load(p.folder)
    assert all(p3.custom_color(c) is None for c in range(4))
    # 没有文件颜色（.kfbf）时不设颜色，按染色显示
    p4 = Project(folder=tmp / "p4"); p4.ensure_channels([ChannelInfo(0, "DAPI", (0, 0, 255)), ChannelInfo(1, "670", (255, 0, 255))])
    assert p4.channel_names == ["DAPI", "Far red"] and p4.custom_color(1) is None
    shutil.rmtree(tmp, ignore_errors=True)
    print("1. 通道判断 / 显示颜色 OK")


def check_reading(img, qptiff=None):
    import numpy as np
    from cfos.core.slide_io import open_slide

    def reads(path, fast, threads):
        os.environ["CFOS_FAST_READ"] = "1" if fast else "0"
        os.environ["CFOS_DECODE_THREADS"] = str(threads)
        sl = open_slide(path)
        H, W = sl.level_shapes[0]
        rng = np.random.default_rng(0)
        out = []
        for lv in range(min(3, sl.n_levels)):
            ds = sl.downsample(lv)
            for _ in range(5):
                w, h = int(min(W, 2500 * ds)), int(min(H, 1800 * ds))
                x, y = int(rng.integers(0, max(1, W - w))), int(rng.integers(0, max(1, H - h)))
                out.append(sl.read(lv, (x + 0.4, y + 0.7, w, h)))
                out.append(sl.read(lv, (x, y, w // 3, h // 3), [len(sl.channels) - 1]))
        out.append(sl.read(sl.n_levels - 1))
        out.append(sl.read(0, (W - 300, H - 200, 900, 900)))     # 超出图片边缘
        fast_used = any(v is not None for v in sl._fast.values())
        sl.close()
        return out, fast_used

    for path in [p for p in (img, qptiff) if p]:
        ref, used = reads(path, False, 1)
        assert not used
        for th in (1, 4):
            got, used = reads(path, True, th)
            assert used, "没有用上多线程读图"
            assert all(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(ref, got)), path
        print("2. 读图逐像素一致：", Path(path).name)
    # .kfbf：多线程 + 缓存 与 串行无缓存 一致
    from kfbf_synth import write_kfbf
    sl = open_slide(img); a = sl.read(1); sl.close()
    k = Path(tempfile.mkdtemp()) / "t.kfbf"
    write_kfbf(k, np.stack([a[0], a[-1], (a[-1] // 2).astype(np.uint8)]), ["DAPI", "520", "670"])
    res = []
    for th, cache in ((1, False), (4, True)):
        os.environ["CFOS_DECODE_THREADS"] = str(th)
        sl = open_slide(k)
        if not cache:
            sl._kf.cache.max_bytes = 0
        res.append([sl.read(0, (x, y, 1500, 1200)) for x, y in ((0, 0), (1300, 0), (1300, 1000), (5000, 3000))]
                   + [sl.read(1), sl.read(0, (10, 20, 2000, 800), [2, 0])])
        if cache:
            assert sl._kf.cache.hits > 0
            assert sl._kf.cache.nbytes <= sl._kf.cache.max_bytes
        sl.close()
    assert all(np.array_equal(x, y) for x, y in zip(*res))
    os.environ.pop("CFOS_DECODE_THREADS", None); os.environ.pop("CFOS_FAST_READ", None)
    print("2. .kfbf 多线程 + 缓存逐像素一致 OK")


def check_ratio(proj):
    import numpy as np, pandas as pd
    from cfos.core.project import Project
    from cfos.core.export import build_tables, params_table
    from cfos.core import stats as S
    from cfos.core.coloc import ratio_column, DAPI_NOTE
    p = Project.load(proj)
    assert S.metric_options(p.channel_names)[0] == "Red+ 占 DAPI+ 的比例 %"
    t = build_tables(p, intensity=False)
    s = t["按脑片汇总"]
    rc = ratio_column("Red")
    assert rc in s.columns and "比例提示" in s.columns
    num = s["DAPI+Red 共标"] if "DAPI+Red 共标" in s.columns else s["Red+DAPI 共标"]
    assert np.allclose(s[rc], (100 * num / s["DAPI+"]).round(2))
    r = t["按脑区×脑片"]
    assert rc in r.columns and "比例提示" in r.columns
    cols = list(r.columns)
    assert cols.index(rc) < cols.index("面积 (mm²)")
    dg = r[r["脑区缩写"].isin(["DG-sg", "CA1sp", "CA3sp"])]
    if len(dg):
        assert dg["比例提示"].str.contains(DAPI_NOTE).all()
    low = r[(r["DAPI+"] < 200) & (r["DAPI+"] >= 0)]
    assert low["比例提示"].str.contains("不足 200").all()
    ok = r[(r["DAPI+"] >= 200) & ~r["脑区缩写"].isin(__import__("cfos.core.regions", fromlist=["x"]).dense_regions())]
    assert (ok["比例提示"] == "").all()
    print("3. 导出表比例列 OK：", s[["DAPI+", "Red+", rc]].to_dict("records"))
    # 每只动物先合计计数再算比例：把同一张脑片当成两只动物的两张脑片……用脑片表做两份假脑片
    r2 = pd.concat([r.assign(脑片="A"), r.assign(脑片="B")])
    for c in S.count_columns(r2):
        r2.loc[r2["脑片"] == "B", c] = r2.loc[r2["脑片"] == "B", c] * 3
    samples = {f"{im}/A": {"animal": "M1", "group": "对照"} for im in r2["图片"].unique()}
    samples.update({f"{im}/B": {"animal": "M1", "group": "对照"} for im in r2["图片"].unique()})
    a = S.per_animal(r2, samples)
    row = a[(a["脑区缩写"] == "root") & (a["半脑"] == "左右合计")].iloc[0]
    want = 100 * row["DAPI+Red 共标"] / row["DAPI+"]
    assert abs(row[rc] - want) < 1e-9
    # 组间统计：选比例指标时结果表带提示列
    samples2 = {f"{im}/A": {"animal": "M1", "group": "对照"} for im in r2["图片"].unique()}
    samples2.update({f"{im}/B": {"animal": "M2", "group": "处理"} for im in r2["图片"].unique()})
    a2 = S.per_animal(r2, samples2)
    gsum, comp = S.compare_groups(a2, S.StatsParams(metric=rc))
    assert "比例提示" in gsum.columns
    pt = params_table(p)
    assert (pt["参数"] == "推荐主指标").any()
    print("3. 组间统计比例指标 OK")


def check_intensity(proj):
    import numpy as np
    from cfos.core.project import Project
    from cfos.core.slide_io import open_slide
    from cfos.core import registration as R, regions as G
    from cfos.core.unmix import UnmixModel, apply
    p = Project.load(proj)
    im = p.images[0]
    s = next(x for x in im.sections if x.registration.get("plane"))
    sl = open_slide(im.path)
    C = len(sl.channels)
    M = np.eye(C); M[C - 1, 0] = 0.12           # 假设 DAPI → 最后一个通道串色 12%
    um = UnmixModel(list(p.channel_names), M, np.array([3.0] * C, np.float32), [])
    reg = R.Registration.from_dict(s.registration)
    ds, (H, W), _ = G._thumb_geometry(sl, s)
    lab, hemi = R.label_maps(reg, W, H, None)
    lab_u, hemi_u = R.orient(lab, reg.orientation), R.orient(hemi, reg.orientation)
    tgt = 4.0       # 用粗一点的分辨率核对（算法与 1 µm 相同）
    raw = G.region_intensity_sums(sl, s, lab_u, hemi_u, ds, target_um=tgt)
    cor = G.region_intensity_sums(sl, s, lab_u, hemi_u, ds, target_um=tgt, unmix_model=um)
    # 逐像素解混后整块求和：结果应与分块计算一致
    tot_raw = sum(v[1][C - 1] for v in raw.values()); tot_cor = sum(v[1][C - 1] for v in cor.values())
    assert tot_cor < tot_raw
    # 汇总后再近似校正（Σ真 ≈ Σ观测 − a·(Σ源 − n·off)）与逐像素的差别
    worst = 0
    for key, (n, sm) in cor.items():
        if n < 1000:
            continue
        n_, smr = raw[key]
        approx = smr[C - 1] - 0.12 * (smr[0] - 3.0 * n_)
        worst = max(worst, abs(approx - sm[C - 1]) / max(sm[C - 1], 1))
    print(f"4. 脑区强度去串色 OK：最后通道总灰度 {tot_raw:.3g} → {tot_cor:.3g}；"
          f"若改用“汇总后近似校正”，最大相对误差 {worst:.2%}（所以采用逐像素校正）")
    # 签名：开启解混后缓存要作废
    sig0 = G._intensity_signature(sl, s, reg, "x", im.exclusions)
    sig1 = G._intensity_signature(sl, s, reg, "x", im.exclusions, um)
    assert sig0 != sig1
    sl.close()


def check_colors(proj):
    import numpy as np
    from cfos.core.project import Project
    from cfos.core.slide_io import open_slide
    from cfos.core import qc, figure
    p = Project.load(proj)
    im = p.images[0]; s = im.sections[0]
    sl = open_slide(im.path)
    df = p.load_cells(im)
    a = np.asarray(qc.section_qc_image(p, im, sl, s, df))
    c = len(p.channel_names) - 1
    p.display[c].color = [255, 255, 0]
    b = np.asarray(qc.section_qc_image(p, im, sl, s, df))
    assert not np.array_equal(a, b)
    bb = (s.bbox0[0], s.bbox0[1], 2000, 2000)
    rgb = figure.render_rgb(p, sl, bb, sl.level_for_pixel_size(4), [c])
    assert rgb[..., 2].max() == 0 and rgb[..., 0].max() > 0 and np.allclose(rgb[..., 0], rgb[..., 1])  # 黄色 = 红 + 绿
    p.display[c].color = None
    sl.close()
    print("5. 质控图 / 截图用自选颜色 OK")


def main():
    img, proj = sys.argv[1], sys.argv[2]
    q = sys.argv[3] if len(sys.argv) > 3 else None
    check_roles()
    check_reading(img, q)
    check_ratio(proj)
    check_intensity(proj)
    check_colors(proj)
    print("DONE")


if __name__ == "__main__":
    main()
