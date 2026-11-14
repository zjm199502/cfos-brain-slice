"""v0.8.2 核心功能测试（不需要界面）：串色校正接入识别、通道设置（不使用 / 拆分用通道）、导出。
用法：python tests/v082_core.py <示例图片> <输出文件夹>"""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from cfos.core.project import Project, ImageEntry
    from cfos.core import pipeline as P, detection as det, unmix
    from cfos.core.slide_io import open_slide
    from cfos.core.sections import find_sections
    img, out = sys.argv[1], Path(sys.argv[2])
    shutil.rmtree(out, ignore_errors=True)
    p = Project(folder=out / "proj")
    sl = open_slide(img); p.ensure_channels(sl.channels)
    print("channels", p.channel_names, "unmix", p.unmix)
    assert p.unmix.get("enabled") is True                      # 新项目默认开启
    im = ImageEntry(path=img); p.images.append(im)
    im.sections = find_sections(sl, channels=p.tissue_channels())
    # 1. 没开串色校正时的签名与旧版相同
    p.unmix = {"enabled": False}; unmix.prepare(p, sl, im)
    sig_off = det.raw_signature(p.detection)
    assert "unmix" not in sig_off
    t = time.time(); P.detect_image(p, im, sl); t_off = time.time() - t
    n_off = p.load_cells(im)["channel"].value_counts().to_dict()
    # 2. 开启：先自动估算，再识别
    p.unmix = {"enabled": True}
    t = time.time(); P.detect_image(p, im, sl); t_on = time.time() - t
    print("unmix settings", {k: v for k, v in p.unmix.items() if k != "detail"}, p.unmix.get("detail"))
    assert "coef" in p.unmix and "auto" in p.unmix
    assert "unmix" in det.raw_signature(p.detection)
    n_on = p.load_cells(im)["channel"].value_counts().to_dict()
    print(f"不校正 {n_off}（{t_off:.0f} 秒）；校正 {n_on}（{t_on:.0f} 秒，含估算）")
    # 3. 保存 / 读回
    p2 = Project.load(p.folder)
    assert p2.unmix["coef"] == p.unmix["coef"]
    from cfos.core.export import params_table, build_tables
    pt = params_table(p2)
    row = pt[pt.iloc[:, 0] == "通道串色校正"].iloc[0, 1]
    print("params:", row); assert "串色" in row or "未发现" in row
    # 4. 通道设置：重名不行；设为不使用后不识别、导出表不列
    assert p2.set_channel_role(0, "Red（AF594）" if p2.channel_names[1] == "Red" else p2.channel_names[1]) != ""
    assert p2.set_channel_role(1, "不使用") == ""
    assert 1 in p2.unused_channels and not p2.detection.channels[1].detect
    assert p2.set_channel_role(0, "不使用") != ""             # 至少保留一个
    tabs = build_tables(p2)
    cols = list(tabs["按脑片汇总"].columns)
    print("cols", cols[:12])
    assert not any("未用通道" in c for c in cols)
    assert p2.tissue_channels() == [0]
    secs = find_sections(sl, channels=p2.tissue_channels())
    print("sections with DAPI only", [s.bbox0 for s in secs])
    assert len(secs) >= 1
    assert p2.set_channel_role(1, "Red（AF594）") == "" and 1 not in p2.unused_channels
    sl.close()
    print("DONE")


if __name__ == "__main__":
    main()
