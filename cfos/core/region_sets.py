"""脑区范围：把“按图谱层级第几级”换成不熟悉图谱的人也能看懂的几种选法（v0.8.5）。

图谱（Allen CCFv3）的脑区是一棵树：整个脑 → 大脑区（皮层、海马、丘脑…）→ 具体脑区（CA1、BLA…）→ 亚区 / 皮层分层。
同一“级”在不同部位粗细不一样（第 5 级既有“新皮层”这样的大区，也有“基底外侧杏仁核”这样的具体核团），
所以按级筛选对新手不友好。这里提供两套 Allen 研究所官方整理、文献里通用的脑区清单：
  · 12 个大脑区（Brain - Major Divisions）
  · 316 个常用脑区（Brain - Summary Structures，Wang et al., Cell 2020）：互不重叠、粗细适中，全脑 c-Fos 文章大多用它
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .resources import resources_dir

# ---------------------------------------------------------------- 清单
MAJOR = {   # 缩写 → 中文名
    "Isocortex": "新皮层", "OLF": "嗅觉区", "HPF": "海马结构", "CTXsp": "皮层下板（屏状核、杏仁核部分核团）",
    "STR": "纹状体", "PAL": "苍白球", "TH": "丘脑", "HY": "下丘脑", "MB": "中脑", "P": "脑桥", "MY": "延髓", "CB": "小脑",
}
EXTRA_MAJOR = {"fiber tracts": "白质纤维束", "VS": "脑室系统"}   # 不属于 12 个大脑区的两大块

SUMMARY = frozenset("""
FRP MOp MOs SSp-n SSp-bfd SSp-ll SSp-m SSp-ul SSp-tr SSp-un SSs GU VISC AUDd AUDp AUDpo AUDv VISal
VISam VISl VISp VISpl VISpm VISli VISpor ACAd ACAv PL ILA ORBl ORBm ORBvl AId AIp AIv RSPagl RSPd
RSPv VISa VISrl TEa PERI ECT MOB AOB AON TT DP PIR NLOT COAa COAp PAA TR CA1 CA2 CA3 DG FC IG ENTl
ENTm PAR POST PRE SUB ProS HATA APr CLA EPd EPv LA BLA BMA PA CP ACB FS OT LSc LSr LSv SF SH AAA BA
CEA IA MEA GPe GPi SI MA MS NDB TRS BST BAC VAL VM VPL VPLpc VPM VPMpc PoT SPFm SPFp SPA PP MG LGd
LP PO POL SGN AV AM AD IAM IAD LD IMD MD SMT PR PVT PT RE Xi RH CM PCN CL PF PIL RT IGL IntG LGv
SubG MH LH SO ASO PVH PVa PVi ARH ADP AVP AVPV DMH MEPO MPO OV PD PS PVp PVpo SBPV SCH SFO VMPO VLPO
AHN LM MM SUM TMd TMv MPN PMd PMv PVHd VMH PH LHA LPO PST PSTN PeF RCH STN TU ZI ME SCs IC NB SAG
PBG MEV SCO SNr VTA PN RR MRN SCm PAG APN MPT NOT NPC OP PPT CUN RN III MA3 EW IV Pa4 VTN AT LT DT
MT SNc PPN IF IPN RL CLI DR NLL PSV PB SOC B DTN PDTg PCG PG PRNc SG SUT TRN V P5 Acs5 PC5 I5 CS LC
LDT NI PRNr RPO SLC SLD AP DCO VCO CU GR ECU NTB NTS SPVC SPVI SPVO Pa5 VI VII ACVII AMB DMX GRN ICB
IO IRN ISN LIN LRN MARN MDRN PARN PAS PGRNd PGRNl NR PRP PPY LAV MV SPIV SUV x XII y RM
RPA RO LING CENT CUL DEC FOTU PYR UVU NOD SIM AN PRM COPY PFL FL FN IP DN VeCB
""".split()) | {"fiber tracts", "Eth", "RPF"}
# Eth、RPF 是图谱后来新增的两个核团，不在官方清单里，也不属于清单里任何脑区；补进来以免它们的细胞没处计。
# 官方清单里同时有 MDRN 和它的两个亚区 MDRNd、MDRNv，这里只保留 MDRN，保证互不重叠（每个细胞只计一次）

# 常见脑区的中文名（界面预览、Excel“脑区中文名”列用；没有收录的留空，不影响统计）
ZH = {
    **MAJOR, **EXTRA_MAJOR,
    "root": "整个脑", "grey": "灰质（全部细胞区）", "CH": "大脑", "CTX": "大脑皮层", "CTXpl": "皮层板", "CNU": "大脑核团",
    "BS": "脑干", "IB": "间脑", "HB": "后脑", "CBX": "小脑皮层", "CBN": "小脑核团", "HIP": "海马", "RHP": "海马旁区",
    "STRd": "背侧纹状体", "STRv": "腹侧纹状体", "LSX": "外侧隔复合体", "sAMY": "纹状体样杏仁核",
    "FRP": "额极皮层", "MO": "运动皮层", "MOp": "初级运动皮层", "MOs": "次级运动皮层",
    "SS": "躯体感觉皮层", "SSp": "初级躯体感觉皮层", "SSp-n": "初级躯体感觉皮层 鼻区", "SSp-bfd": "初级躯体感觉皮层 桶状区",
    "SSp-ll": "初级躯体感觉皮层 后肢区", "SSp-m": "初级躯体感觉皮层 口区", "SSp-ul": "初级躯体感觉皮层 前肢区",
    "SSp-tr": "初级躯体感觉皮层 躯干区", "SSp-un": "初级躯体感觉皮层 未分区", "SSs": "次级躯体感觉皮层",
    "GU": "味觉皮层", "VISC": "内脏皮层", "AUD": "听觉皮层", "AUDd": "背侧听觉皮层", "AUDp": "初级听觉皮层",
    "AUDpo": "后部听觉皮层", "AUDv": "腹侧听觉皮层", "VIS": "视觉皮层", "VISp": "初级视觉皮层",
    "VISal": "前外侧视觉皮层", "VISam": "前内侧视觉皮层", "VISl": "外侧视觉皮层", "VISpl": "后外侧视觉皮层",
    "VISpm": "后内侧视觉皮层", "VISli": "外侧中间视觉皮层", "VISpor": "嗅周后视觉皮层", "VISa": "前部视觉皮层",
    "VISrl": "吻外侧视觉皮层", "ACA": "前扣带皮层", "ACAd": "前扣带皮层 背侧部", "ACAv": "前扣带皮层 腹侧部",
    "PL": "前边缘皮层", "ILA": "下边缘皮层", "ORB": "眶额皮层", "ORBl": "眶额皮层 外侧部", "ORBm": "眶额皮层 内侧部",
    "ORBvl": "眶额皮层 腹外侧部", "AI": "无颗粒岛叶皮层", "AId": "无颗粒岛叶皮层 背侧部", "AIp": "无颗粒岛叶皮层 后部",
    "AIv": "无颗粒岛叶皮层 腹侧部", "RSP": "压后皮层", "RSPagl": "压后皮层 外侧无颗粒部", "RSPd": "压后皮层 背侧部",
    "RSPv": "压后皮层 腹侧部", "PTLp": "后顶叶联合皮层", "TEa": "颞叶联合皮层", "PERI": "嗅周皮层", "ECT": "外嗅皮层",
    "MOB": "主嗅球", "AOB": "副嗅球", "AON": "前嗅核", "TT": "盖带", "DP": "背侧脚皮层", "PIR": "梨状皮层",
    "NLOT": "外侧嗅束核", "COA": "皮质杏仁核区", "COAa": "皮质杏仁核区 前部", "COAp": "皮质杏仁核区 后部",
    "PAA": "梨状-杏仁核区", "TR": "梨状后过渡区",
    "CA1": "海马 CA1 区", "CA2": "海马 CA2 区", "CA3": "海马 CA3 区", "DG": "齿状回", "FC": "束状回", "IG": "灰被",
    "ENT": "内嗅皮层", "ENTl": "外侧内嗅皮层", "ENTm": "内侧内嗅皮层", "PAR": "旁下托", "POST": "后下托", "PRE": "前下托",
    "SUB": "下托", "ProS": "原下托",
    "CLA": "屏状核", "EP": "内梨状核", "EPd": "内梨状核 背侧部", "EPv": "内梨状核 腹侧部", "LA": "外侧杏仁核",
    "BLA": "基底外侧杏仁核", "BMA": "基底内侧杏仁核", "PA": "后杏仁核",
    "CP": "尾壳核", "ACB": "伏隔核", "FS": "纹状体底", "OT": "嗅结节", "LSc": "外侧隔核 尾侧部", "LSr": "外侧隔核 吻侧部",
    "LSv": "外侧隔核 腹侧部", "SF": "隔伞核", "SH": "隔海马核", "AAA": "杏仁前区", "CEA": "中央杏仁核",
    "IA": "杏仁核间插核", "MEA": "内侧杏仁核",
    "GPe": "苍白球外侧部", "GPi": "苍白球内侧部", "SI": "无名质", "MA": "大细胞核", "MS": "内侧隔核", "NDB": "斜角带核",
    "TRS": "三角隔核", "BST": "终纹床核",
    "VAL": "丘脑腹前外侧核", "VM": "丘脑腹内侧核", "VPL": "丘脑腹后外侧核", "VPM": "丘脑腹后内侧核",
    "MG": "内侧膝状体", "LGd": "背侧外侧膝状体", "LGv": "腹侧外侧膝状体", "LP": "丘脑外侧后核", "PO": "丘脑后核群",
    "AV": "丘脑前腹侧核", "AM": "丘脑前内侧核", "AD": "丘脑前背侧核", "LD": "丘脑外侧背核", "MD": "丘脑背内侧核",
    "PVT": "丘脑室旁核", "PT": "带旁核", "RE": "连结核", "RH": "菱形核", "CM": "丘脑中央内侧核", "CL": "丘脑中央外侧核",
    "PF": "束旁核", "RT": "丘脑网状核", "MH": "内侧缰核", "LH": "外侧缰核",
    "SO": "视上核", "PVH": "下丘脑室旁核", "ARH": "弓状核", "DMH": "下丘脑背内侧核", "MPO": "内侧视前区",
    "SCH": "视交叉上核", "SFO": "穹窿下器", "AHN": "下丘脑前核", "MM": "内侧乳头体核", "SUM": "乳头体上核",
    "MPN": "内侧视前核", "VMH": "下丘脑腹内侧核", "PH": "下丘脑后核", "LHA": "下丘脑外侧区", "LPO": "外侧视前区",
    "STN": "丘脑底核", "ZI": "未定带", "ME": "正中隆起",
    "SCs": "上丘 感觉相关层", "SCm": "上丘 运动相关层", "IC": "下丘", "SNr": "黑质网状部", "SNc": "黑质致密部",
    "VTA": "腹侧被盖区", "MRN": "中脑网状核", "PAG": "导水管周围灰质", "RN": "红核", "IPN": "脚间核", "DR": "中缝背核",
    "PPN": "脚桥核", "CUN": "楔形核", "EW": "动眼神经副核（E-W 核）",
    "PB": "臂旁核", "LC": "蓝斑", "LDT": "背外侧被盖核", "PG": "脑桥灰质", "PRNc": "脑桥网状核 尾侧部",
    "PRNr": "脑桥网状核 吻侧部", "SOC": "上橄榄复合体", "NI": "未定核", "PSV": "三叉神经感觉主核", "V": "三叉神经运动核",
    "NTS": "孤束核", "AP": "最后区", "DMX": "迷走神经背运动核", "IO": "下橄榄核", "XII": "舌下神经核", "VII": "面神经核",
    "AMB": "疑核", "RM": "中缝大核", "RPA": "中缝苍白核", "RO": "中缝隐核", "GRN": "巨细胞网状核", "MV": "前庭内侧核",
    "DCO": "耳蜗背侧核", "VCO": "耳蜗腹侧核", "CU": "楔束核", "GR": "薄束核",
    "VERM": "小脑蚓部", "HEM": "小脑半球", "FN": "顶核", "IP": "间位核", "DN": "齿状核",
}

# ---------------------------------------------------------------- 选法
ALL, MAJOR_ONLY, SUMMARY_ONLY, MAJOR_SUMMARY, DEPTH = "all", "major", "summary", "major+summary", "depth"
CUSTOM = "custom"  # Excel-only mode. The preset list is also used by other analyses.
MODES = [   # (代号, 下拉框里的名字, 一句话说明)
    (ALL, "全部脑区（从大区到最细亚区都列出）",
     "不筛选。同一个细胞会同时出现在它所在的亚区、脑区和大脑区里（例如一个 CA1 的细胞也计入“海马结构”），"
     "脑区最多（一张脑片几百行）。熟悉图谱、想自己挑脑区时选这个。"),
    (MAJOR_ONLY, "只列 12 个大脑区（新皮层、海马结构、丘脑、下丘脑…）",
     "新皮层、嗅觉区、海马结构、皮层下板、纹状体、苍白球、丘脑、下丘脑、中脑、脑桥、延髓、小脑。"
     "互不重叠，每个细胞只计一次。适合先看整体分布。"),
    (SUMMARY_ONLY, "常用脑区（约 300 个，如 CA1、BLA、PVT、MOp…）",
     "Allen 研究所整理的标准脑区清单（Summary Structures），全脑 c-Fos 文章大多按这一套报告。"
     "粗细适中：皮层到“初级运动皮层 MOp”“前边缘皮层 PL”，不再分层；皮层下到具体核团（基底外侧杏仁核 BLA、丘脑室旁核 PVT）。"
     "互不重叠，每个细胞只计一次。不确定选哪个时选这个。"
     "（图谱里有少量部位只划到大脑区、没有再细分，落在那里的细胞不在这张清单里，只计入大脑区。）"),
    (MAJOR_SUMMARY, "大脑区 + 常用脑区（上面两项合在一起）",
     "既有 12 个大脑区的合计，也有约 300 个常用脑区。大脑区的数字已经包含它下面的常用脑区，不要把两类相加。"),
    (DEPTH, "按图谱层级（高级）",
     "按 Allen 图谱树的第几级截取。注意同一级在不同部位粗细不同（第 5 级既有“新皮层”，也有“基底外侧杏仁核”），"
     "熟悉图谱的人才建议用。"),
]
MODE_LABEL = {m: lb for m, lb, _ in MODES}
MODE_HELP = {m: h for m, _, h in MODES}
MODE_LABEL[CUSTOM] = "自选 Allen 脑区"
MODE_HELP[CUSTOM] = "只列出勾选的脑区。父区的统计已包含下属区域，同时勾选父区和子区时，两行不能相加。"

CATEGORY_HELP = ("脑区类别：大脑区 = 12 个大区之一；常用脑区 = 标准清单里的脑区（互不重叠）；"
                 "细分亚区 / 皮层分层 = 常用脑区再往下分；上级汇总 = 若干脑区的合计（如“运动皮层 MO”=MOp+MOs）。")


def resolve_mode(mode: str, depth: int) -> str:
    """旧项目没有“脑区范围”这一项：以前设过“最多列到第 N 级”的按层级，否则全部。"""
    if mode in MODE_LABEL:
        return mode
    return DEPTH if depth and depth > 0 else ALL


def describe(mode: str, depth: int = 0, skip_layers: bool = False, *, region_ids=()) -> str:
    """写进“分析参数”表的一句话。"""
    mode = resolve_mode(mode, depth)
    if mode == CUSTOM:
        return f"自选 Allen 脑区（{len(set(region_ids))} 个）。{MODE_HELP[CUSTOM]}"
    if mode == DEPTH:
        s = f"按图谱层级：≤ 第 {depth} 级" if depth else "全部脑区"
    else:
        s = MODE_LABEL[mode].split("（")[0]
    if skip_layers and mode in (ALL, DEPTH):
        s += "，不列皮层分层"
    return s


def is_layer(name) -> bool:
    import re
    return bool(re.search(r"[Ll]ayer", str(name)))


@lru_cache(maxsize=1)
def info() -> pd.DataFrame:
    """每个脑区一行，含 Allen ID、缩写、名称及层级，按图谱顺序排列。"""
    st = pd.read_csv(resources_dir() / "atlas" / "structures.csv")   # 只读表格，不加载图谱体数据
    id2acr = {int(i): a for i, a in zip(st["id"], st["acronym"])}
    big = {**MAJOR, **EXTRA_MAJOR}
    rows = []
    for r in st.itertuples(index=False):
        path = [id2acr[int(x)] for x in str(r.path).split("/") if x and int(x) in id2acr]
        above = path[:-1]
        major = next((a for a in path if a in big), "")
        if r.acronym in EXTRA_MAJOR or major in EXTRA_MAJOR:
            cat = "其他（白质 / 脑室）"
        elif r.acronym in MAJOR:
            cat = "大脑区"
        elif r.acronym in SUMMARY:
            cat = "常用脑区"
        elif any(a in SUMMARY for a in above):
            cat = "皮层分层" if is_layer(r.name) else "细分亚区"
        else:
            cat = "上级汇总"
        rows.append({"id": int(r.id), "acronym": r.acronym, "name": r.name, "zh": ZH.get(r.acronym, ""), "depth": len(path) - 1,
                     "parent": above[-1] if above else "", "major": major,
                     "所属大脑区": f"{big[major]} {major}" if major else "—", "脑区类别": cat})
    return pd.DataFrame(rows)


def custom_regions(region_ids) -> pd.DataFrame:
    """Resolve explicit Allen IDs and identify stale settings before starting an export."""
    wanted = set(region_ids)
    if not wanted:
        raise ValueError("请至少勾选一个 Allen 脑区，或选择其他导出范围。")
    atlas = info()
    rows = atlas.loc[atlas["id"].isin(wanted)]
    unknown = wanted.difference(rows["id"])
    if unknown:
        raise ValueError("当前图谱中找不到已选的 Allen ID：" + "、".join(sorted(map(str, unknown))) + "。请重新选择脑区。")
    return rows


def selected(mode: str, depth: int = 0, skip_layers: bool = False, *, region_ids=()) -> set:
    """这种选法下会列出的脑区缩写。"""
    mode = resolve_mode(mode, depth)
    if mode == CUSTOM:
        return set(custom_regions(region_ids)["acronym"]) if region_ids else set()
    t = info()
    if mode == MAJOR_ONLY:
        keep = t["acronym"].isin(MAJOR)
    elif mode == SUMMARY_ONLY:
        keep = t["acronym"].isin(SUMMARY)
    elif mode == MAJOR_SUMMARY:
        keep = t["acronym"].isin(SUMMARY) | t["acronym"].isin(MAJOR)
    else:
        keep = pd.Series(True, index=t.index)
        if mode == DEPTH and depth > 0:
            keep &= t["depth"] <= depth
        if skip_layers:
            keep &= ~t["name"].map(is_layer)
    return set(t.loc[keep, "acronym"])


def filter_table(t: pd.DataFrame, mode: str, depth: int = 0, skip_layers: bool = False, *, region_ids=()) -> pd.DataFrame:
    """按脑区范围筛选一张带“脑区缩写”列的表。"""
    mode = resolve_mode(mode, depth)
    if mode == CUSTOM:
        rows = custom_regions(region_ids)
        if "Allen ID" in t:
            return t[t["Allen ID"].isin(rows["id"])]
        return t[t["脑区缩写"].isin(rows["acronym"])]
    if mode == ALL and not skip_layers:
        return t
    return t[t["脑区缩写"].isin(selected(mode, depth, skip_layers))]


def annotate(t: pd.DataFrame) -> pd.DataFrame:
    """在“层级”列后面加三列：脑区中文名、所属大脑区、脑区类别（导出的 Excel 用，方便筛选）。"""
    if t is None or (t.empty and "Allen ID" not in t.columns) or "脑区缩写" not in t.columns or "脑区类别" in t.columns:
        return t
    m = info().drop_duplicates("acronym").set_index("acronym")
    t = t.copy()
    at = (t.columns.get_loc("层级") if "层级" in t.columns else t.columns.get_loc("脑区名称")) + 1
    t.insert(at, "脑区类别", t["脑区缩写"].map(m["脑区类别"]).fillna(""))
    t.insert(at, "所属大脑区", t["脑区缩写"].map(m["所属大脑区"]).fillna("—"))
    t.insert(t.columns.get_loc("脑区名称") + 1, "脑区中文名", t["脑区缩写"].map(m["zh"]).fillna(""))
    return t
