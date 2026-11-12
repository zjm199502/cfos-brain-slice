"""网络分析冒烟测试：用植入相关结构的模拟数据（2 组 × 8 只），检查能找回植入的模块，并生成表格和图。
用法：python tests/network_smoke.py <按脑区×脑片.pkl> <输出文件夹>"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from cfos.core import network as N
from cfos.core.stats import per_animal

rs = pd.read_pickle(sys.argv[1]); out = Path(sys.argv[2])
one = rs[(rs["图片"] == rs["图片"].iloc[0]) & (rs["脑片"] == rs["脑片"].iloc[0])]
anc = N._structures()
rng = np.random.default_rng(3)
frames, samples = [], {}
for grp, module in (("对照", "Isocortex"), ("电击", "非皮层")):
    for a in range(8):
        lat = rng.normal()
        t = one.copy(); t["图片"] = f"{grp}{a}"; t["脑片"] = "S01"
        maj = t["脑区缩写"].map(lambda x: N.major_of(x, anc))
        inmod = (maj == module) if module != "非皮层" else (maj != "Isocortex")
        dens = 50 * np.exp(0.8 * lat * inmod + 0.35 * rng.normal(size=len(t)))
        t["Red+"] = (dens * t["面积 (mm²)"]).round()
        frames.append(t); samples[f"{grp}{a}/S01"] = {"animal": f"{grp}{a}", "group": grp}
rs2 = pd.concat(frames, ignore_index=True)
out.mkdir(parents=True, exist_ok=True); rs2.to_pickle(out / "fake_net_rs.pkl")
import json; json.dump(samples, open(out / "fake_net_samples.json", "w", encoding="utf-8"), ensure_ascii=False)
animal = per_animal(rs2, samples)
p = N.NetParams(metric="Red+ 密度 (个/mm²)", depth=7, n_perm=300, n_boot=200)
t0 = time.time()
res = N.run_network(animal, p, progress=lambda i, n, s: print(i, n, s))
print("耗时", round(time.time() - t0, 1), "s；脑区数", len(res["regions"]))
nd = res["nodes"]; m = nd[nd["阈值"] == 0.7]
for g in ("对照", "电击"):
    s = m[m["组别"] == g]
    print(g, "平均度：", s.groupby("大脑区")["度"].mean().round(1).to_dict())
    inn, outn = s[s["大脑区"] == "新皮层"]["度"].mean(), s[s["大脑区"] != "新皮层"]["度"].mean()
    assert (inn > 3 * outn) if g == "对照" else (outn > 3 * inn)
print(res["perm"])
print(res["deg_diff"].head(5))
print(res["edges_diff"].head(5))
class P: pass
proj = P(); proj.samples = samples; proj.channel_names = ["DAPI", "Red"]; proj.grading = {}
o, summ, figs = N.run_network_analysis(proj, out, p, region_sec=rs2)
print(summ); print([f.name for f in figs])
print(pd.ExcelFile(o / "网络分析.xlsx").sheet_names)
print("OK")
