import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
from cfos.core.atlas import load_atlas
from scipy import ndimage as ndi
S = sys.argv[1]; tags = sys.argv[2].split(',')
st = load_atlas('CCFv3').structures
def group(*ids):
    return st[st.path.astype(str).apply(lambda p: any(f"/{i}/" in f"/{p}/" for i in ids))]['index'].values
LAYERS = group(632, 407, 446, 495)
DG = group(632)
rows = []
for n in ('S01','S02','S03'):
    z = np.load(f"{S}/{n}.npz"); pix = float(z['pix'])
    obs = np.load(f"{S}/{n}_layers.npy")   # DAPI-detected principal cell layers (oriented frame)
    for tag in tags:
        reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_{tag}.json")))
        img = R.orient(z['img'], reg.orientation); mask = R.orient(z['mask'], reg.orientation); H, W = img.shape
        warp = np.load(f"{S}/{n}_{tag}_warp.npy") if reg.nonlinear else None
        lab, _ = R.label_maps(reg, W, H, warp)
        atl = np.isin(lab, LAYERS)
        d_atl = ndi.distance_transform_edt(~atl) * pix
        d_obs = ndi.distance_transform_edt(~obs) * pix
        o2a = d_atl[obs]                      # each DAPI layer pixel -> nearest atlas layer
        dg = np.isin(lab, DG); a2o = d_obs[dg]   # atlas DG -> nearest DAPI layer (DG is fully detected)
        recall = float((d_atl[obs] <= pix).mean())
        dice = R.dice(lab > 0, mask)
        rows.append((n, tag, np.mean(o2a), np.percentile(o2a, 90), np.mean(a2o), recall, dice))
print(f"{'slice':5} {'method':14} {'DAPI层→图谱 均值µm':>16} {'P90µm':>7} {'图谱DG→DAPI µm':>14} {'层内比例':>8} {'轮廓Dice':>8}")
for r in rows: print(f"{r[0]:5} {r[1]:14} {r[2]:16.1f} {r[3]:7.1f} {r[4]:14.1f} {r[5]:8.2f} {r[6]:8.3f}")
import collections
agg = collections.defaultdict(list)
for r in rows: agg[r[1]].append(r[2:])
for k, v in agg.items(): v = np.mean(v, 0); print(f"mean  {k:14} {v[0]:16.1f} {v[1]:7.1f} {v[2]:14.1f} {v[3]:8.2f} {v[4]:8.3f}")
