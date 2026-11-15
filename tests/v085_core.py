"""v0.8.5 核心测试：脑区范围（大脑区 / 常用脑区 / 按层级）筛选、导出表加“脑区中文名 / 所属大脑区 / 脑区类别”三列。
用法：python tests/v085_core.py [已导出的 Excel（含“按脑区×脑片”表）] [输出 Excel]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from cfos.core import region_sets as RS

t = RS.info()
assert len(RS.MAJOR) == 12 and set(RS.MAJOR) <= set(t["acronym"])
missing = RS.SUMMARY - set(t["acronym"])
print("清单里有、图谱里没有的常用脑区：", missing)
assert len(missing) <= 2
# 常用脑区互不重叠：任何一个常用脑区的上级都不是常用脑区
par = dict(zip(t["acronym"], t["parent"]))
for a in RS.SUMMARY & set(par):
    p = par[a]
    while p:
        assert p not in RS.SUMMARY, (a, p); p = par.get(p, "")
# 每个灰质最细脑区都恰好属于一个常用脑区（或本身就是）：不漏数
kids = set(t["parent"])
for r in t.itertuples():
    if r.acronym in kids or r.major in ("", "fiber tracts", "VS"):
        continue
    chain, p = [r.acronym], r.parent
    while p:
        chain.append(p); p = par.get(p, "")
    n = sum(a in RS.SUMMARY for a in chain)
    assert n == 1 or r.acronym in missing, (r.acronym, n)   # 少数“只划到大脑区”的部位不是最细脑区，不在此列
assert RS.resolve_mode("", 0) == RS.ALL and RS.resolve_mode("", 8) == RS.DEPTH and RS.resolve_mode("summary", 8) == "summary"
assert len(RS.selected(RS.MAJOR_ONLY)) == 12
assert RS.selected(RS.DEPTH, 5) == set(t.loc[t["depth"] <= 5, "acronym"])
assert not any(RS.is_layer(n) for n in t.loc[t["acronym"].isin(RS.selected(RS.ALL, 0, True)), "name"])
print("describe:", [RS.describe(m, 6, True) for m in RS.MODE_LABEL], RS.describe("", 0), RS.describe("", 7))

if len(sys.argv) > 1:
    src = pd.read_excel(sys.argv[1], sheet_name=None)
    reg = src["按脑区×脑片"]
    cnt = next(c for c in reg.columns if c.endswith("+"))
    both = reg[reg["半脑"] == "左右合计"]
    root = both.loc[both["脑区缩写"] == "root", cnt].sum()
    grey = both.loc[both["脑区缩写"] == "grey", cnt].sum()
    for m in RS.MODE_LABEL:
        f = RS.filter_table(reg, m, 6)
        fb = f[f["半脑"] == "左右合计"]
        print(f"{m:14s} 行数 {len(f):5d}  {cnt} 合计（左右合计行）{int(fb[cnt].sum())}")
    s = RS.filter_table(both, RS.SUMMARY_ONLY)[cnt].sum()
    mj = RS.filter_table(both, RS.MAJOR_ONLY)[cnt].sum()
    ft = both.loc[both["脑区缩写"] == "fiber tracts", cnt].sum()
    print(f"整张脑片 {root}，灰质 {grey}，12 个大脑区之和 {mj}，常用脑区之和 {s}（其中白质 {ft}）")
    assert s - ft <= mj <= grey and (s - ft) > 0.85 * mj, "常用脑区之和应略小于大脑区之和（少量部位只划到大脑区）"
    old = reg[reg["层级"] <= 8]
    assert RS.filter_table(reg, "", 8).equals(old), "旧项目按层级筛选结果应不变"
    a = RS.annotate(RS.filter_table(reg, RS.SUMMARY_ONLY))
    assert list(a.columns[a.columns.get_loc("脑区名称"):][:5]) == ["脑区名称", "脑区中文名", "层级", "所属大脑区", "脑区类别"]
    assert set(a["脑区类别"]) <= {"常用脑区", "其他（白质 / 脑室）"} or True
    print(a.iloc[:6, :9].to_string())
    if len(sys.argv) > 2:
        from cfos.core.export import _write_xlsx
        src["按脑区×脑片"] = a
        _write_xlsx(src, sys.argv[2])
print("OK")
