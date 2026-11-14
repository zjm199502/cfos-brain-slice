"""串色校正试验脚本：估算一张片子的串色系数，输出对比图和说明。

用法：
  python tools/unmix_demo.py 片子.qptiff 输出文件夹
  python tools/unmix_demo.py 片子.qptiff 输出文件夹 --add 2:1:0.15 --colabel 2:1:0.4
--add 源:目标:比例      人为加入串色（已知答案，用来验证能否估回来）
--colabel 源:目标:比例  人为制造“真共标”：把源通道中这一比例的阳性细胞复制到目标通道，
                        用来检查校正会不会把真共标也扣掉
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfos.core import unmix  # noqa: E402
from cfos.core.slide_io import open_slide  # noqa: E402


def make_synthetic(tile, offsets, add, colabel, rng, pix):
    """返回 (加工后的图, 共标细胞掩膜 or None, 非共标阳性细胞掩膜 or None)。"""
    f = tile.astype(np.float32)
    g = f.copy()
    co_m = non_m = None
    for (j, i, frac) in colabel:
        s = ndi.gaussian_filter(np.maximum(f[j] - offsets[j], 0), 1.0)
        thr = max(np.percentile(s, 97), 8)
        lab, n = ndi.label(s > thr)
        sizes = ndi.sum(np.ones_like(s), lab, range(1, n + 1))
        ok = np.nonzero(sizes * pix * pix >= 15)[0] + 1      # 至少 15 µm² 才算细胞
        pick = rng.random(ok.size) < frac
        co_m = np.isin(lab, ok[pick])
        non_m = np.isin(lab, ok[~pick])
        g[i] += 0.7 * np.where(co_m, np.maximum(f[j] - offsets[j], 0), 0)
    for (j, i, a) in add:
        g[i] += a * np.maximum(f[j] - offsets[j], 0)
    top = 255 if tile.dtype == np.uint8 else 65535
    return np.clip(np.rint(g), 0, top).astype(tile.dtype), co_m, non_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide")
    ap.add_argument("out")
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--colabel", nargs="*", default=[])
    ap.add_argument("--pairs", default="", help="手动指定通道对，如 0:1,2:1")
    ap.add_argument("--um", type=float, default=0.5)
    ap.add_argument("--tiles", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    add = [(int(a), int(b), float(c)) for a, b, c in (x.split(":") for x in args.add)]
    colabel = [(int(a), int(b), float(c)) for a, b, c in (x.split(":") for x in args.colabel)]
    rng = np.random.default_rng(1)

    s = open_slide(args.slide)
    names = [c.name for c in s.channels]
    t0 = time.time()
    tiles, masks, pix, offsets = unmix.sample_tiles(s, pixel_um=args.um, n_tiles=args.tiles)
    t_read = time.time() - t0
    co_masks = []
    if add or colabel:
        new = []
        for t in tiles:
            g, co, non = make_synthetic(t, offsets, add, colabel, rng, pix)
            new.append(g)
            co_masks.append((co, non))
        tiles = new
    pairs = None
    if args.pairs:
        pairs = [tuple(int(v) for v in p.split(":")) for p in args.pairs.split(",")]
    t1 = time.time()
    model = unmix.estimate(tiles, masks, pix, names, pairs=pairs, offsets=offsets)
    t_est = time.time() - t1
    lines = unmix.describe(model)
    print(f"取样 {len(tiles)} 块 × 250 µm，{pix:.2f} µm/像素；读图 {t_read:.1f} 秒，估算 {t_est:.1f} 秒")
    print("暗背景：", dict(zip(names, np.round(offsets, 1).tolist())))
    for l in lines:
        print(" ", l)

    # 共标保留检查
    check = {}
    if colabel:
        j, i, _ = colabel[0]
        kept_co, kept_non, raw_co, raw_non, true_co = [], [], [], [], []
        for t, (co, non) in zip(tiles, co_masks):
            fixed = unmix.apply(t, model).astype(np.float32)
            obs = t.astype(np.float32)
            src = np.maximum(obs[j] - offsets[j], 0)
            if co is not None and co.any():
                raw_co.append((obs[i] - offsets[i])[co].sum())
                kept_co.append((fixed[i] - offsets[i])[co].sum())
                true_co.append(0.7 * src[co].sum())
            if non is not None and non.any():
                raw_non.append((obs[i] - offsets[i])[non].sum())
                kept_non.append((fixed[i] - offsets[i])[non].sum())
        check = {
            "真共标细胞：校正后保留的真实信号比例": float(np.sum(kept_co) / max(np.sum(true_co), 1)),
            "非共标细胞：校正前目标通道残留（相对真共标信号）": float(np.sum(raw_non) / max(np.sum(true_co), 1)),
            "非共标细胞：校正后目标通道残留（相对真共标信号）": float(np.sum(kept_non) / max(np.sum(true_co), 1)),
        }
        # 互信息法的对比
        mi_model = unmix.UnmixModel(names, np.eye(len(names)), offsets, [], "mi")
        for e in model.pairs:
            if e.a_mi >= 0.01:
                mi_model.matrix[e.sink, e.source] = e.a_mi
        kept_mi = []
        for t, (co, non) in zip(tiles, co_masks):
            if co is not None and co.any():
                kept_mi.append((unmix.apply(t, mi_model).astype(np.float32)[i] - offsets[i])[co].sum())
        check["（对照）互信息法：真共标保留比例"] = float(np.sum(kept_mi) / max(np.sum(true_co), 1))
        for k, v in check.items():
            print(f"  {k}：{v * 100:.1f}%")

    # ---- 图 1：每对通道的散点密度 + 下包络线
    pairs_e = model.pairs
    if pairs_e:
        ncol = min(3, len(pairs_e))
        nrow = int(np.ceil(len(pairs_e) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.2 * nrow), squeeze=False)
        sig = np.concatenate([np.stack([unmix._signal(t[c].astype(np.float32), pix, 30.0)
                                        for c in range(t.shape[0])]).reshape(t.shape[0], -1) for t in tiles], 1)
        for k, e in enumerate(pairs_e):
            ax = axs[k // ncol][k % ncol]
            x, y = sig[e.source], sig[e.sink]
            hx, hy = np.percentile(x, 99.95), np.percentile(y, 99.95)
            ax.hist2d(x, y, bins=160, range=[[np.percentile(x, 0.5), hx], [np.percentile(y, 0.5), hy]],
                      cmap="magma", norm=matplotlib.colors.LogNorm())
            ax.plot(e.bins_x, e.bins_y, "c.", ms=5, label="分段 5% 低分位")
            icpt = float(np.mean(e.bins_y - e.a_envelope * e.bins_x))
            xx = np.array([0, hx])
            ax.plot(xx, e.a_envelope * xx + icpt, "c-", lw=1.6, label=f"下包络法 {e.a_envelope * 100:.1f}%")
            ax.plot(xx, e.a_mi * xx + icpt, "y--", lw=1.2, label=f"互信息法 {e.a_mi * 100:.1f}%")
            ttl = f"{names[e.source]} → {names[e.sink]}"
            tru = [a for (j, i, a) in add if (j, i) == (e.source, e.sink)]
            if tru:
                ttl += f"（加入的真值 {tru[0] * 100:.0f}%）"
            ax.set_title(ttl)
            ax.set_xlabel(f"{names[e.source]} 亮度（扣背景）")
            ax.set_ylabel(f"{names[e.sink]} 亮度（扣背景）")
            ax.legend(fontsize=8, loc="upper left")
        for k in range(len(pairs_e), nrow * ncol):
            axs[k // ncol][k % ncol].axis("off")
        fig.tight_layout()
        fig.savefig(out / "1_串色系数估算.png", dpi=130)
        plt.close(fig)

    # ---- 图 2：串色最强的一对，在源通道最亮的那块上做前后对比
    used = [(model.matrix[e.sink, e.source], e) for e in pairs_e]
    used = [u for u in used if u[0] > 0]
    if used:
        a, e = max(used, key=lambda u: u[0])
        j, i = e.source, e.sink
        k = int(np.argmax([np.percentile(t[j], 99) for t in tiles]))
        crop = tiles[k]
        fixed = unmix.apply(crop, model)

        def disp(v, lo, hi):
            return np.clip((v.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)

        lo_i, hi_i = np.percentile(crop[i], [1, 99.8])
        lo_j, hi_j = np.percentile(crop[j], [1, 99.8])
        fig, axs = plt.subplots(1, 4, figsize=(19, 5))
        axs[0].imshow(disp(crop[j], lo_j, hi_j), cmap="gray")
        axs[0].set_title(f"源通道 {names[j]}")
        axs[1].imshow(disp(crop[i], lo_i, hi_i), cmap="gray")
        axs[1].set_title(f"{names[i]} 校正前")
        axs[2].imshow(disp(fixed[i], lo_i, hi_i), cmap="gray")
        axs[2].set_title(f"{names[i]} 校正后（同样的显示范围）")
        rem = crop[i].astype(np.float32) - fixed[i].astype(np.float32)
        axs[3].imshow(rem, cmap="inferno", vmin=0, vmax=max(np.percentile(rem, 99.8), 1))
        axs[3].set_title(f"被扣掉的部分（{a * 100:.1f}% × {names[j]}）")
        for ax in axs:
            ax.axis("off")
        fig.suptitle(f"{Path(args.slide).name}：250×250 µm 小块，{pix:.2f} µm/像素", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "2_校正前后对比.png", dpi=130)
        plt.close(fig)

    (out / "串色系数.json").write_text(json.dumps(
        {**model.to_dict(), "说明": lines,
         "人为加入串色": {f"{names[j]}→{names[i]}": a for (j, i, a) in add},
         "人为共标": {f"{names[j]}→{names[i]}": a for (j, i, a) in colabel},
         "共标保留检查": check,
         "用时秒": {"读图": round(t_read, 1), "估算": round(t_est, 1)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("输出：", out)


if __name__ == "__main__":
    main()
