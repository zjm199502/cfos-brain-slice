"""脑区活动与行为相关：用假数据（8 只动物）检查能找出人为设定的相关脑区。
用法：python tests/behavior_smoke.py <项目> <假脑区表.pkl> <假分组.json> <输出目录>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from cfos.core.project import Project
from cfos.core.stats import per_animal, metric_options
from cfos.core.behavior import BehaviorParams, run_behavior_correlation, spearman_ci, ALL_ANIMALS

proj, fake, samples, out = sys.argv[1:5]
p = Project.load(proj)
p.samples = json.load(open(samples, encoding="utf-8"))
rs = pd.read_pickle(fake)
metric = metric_options(p.channel_names)[0]
an = per_animal(rs, p.samples)
target = an[(an["半脑"] == "左右合计") & (an["层级"] >= 5)]["脑区缩写"].value_counts().index[0]
dens = an[(an["脑区缩写"] == target) & (an["半脑"] == "左右合计")].set_index("动物")[metric]
rng = np.random.default_rng(1)
for a, v in dens.items():                     # 行为分数 = 该脑区密度的单调函数 + 少量噪声
    assert p.set_behavior(a, f"{np.log1p(v) * 10 + rng.normal(0, 0.01):.4f}")
assert not p.set_behavior("M0", "abc") and p.set_behavior("M0", f"{p.behavior['M0']}%")
lo, hi = spearman_ci(0.8, 8); print("r=0.8, n=8 的 95% CI:", round(lo, 2), round(hi, 2)); assert lo < 0.3 < hi < 0.99
bp = BehaviorParams(metric=metric, behavior_name="冻结时间 %", group=ALL_ANIMALS, max_depth=8)
o, summary, figs = run_behavior_correlation(p, out, bp, region_sec=rs)
print(summary)
res = pd.read_excel(Path(o) / "脑区与行为相关.xlsx", sheet_name="相关排序")
print(res.head(5).to_string())
top = res.iloc[0]
assert top["Spearman r"] > 0.95, top
assert all(Path(f).exists() for f in figs)
# 只用一个组（4 只）→ 应提示动物太少
try:
    run_behavior_correlation(p, out, BehaviorParams(metric=metric, group="对照"), region_sec=rs); raise SystemExit("应报错")
except ValueError as e:
    print("一个组 4 只：", e)
print("OK")
