"""python metrics.py S tag1,tag2,...  -> distance of DAPI-detected cell layers to the atlas cell layers.

Only detections inside the hippocampus count: connected pieces of the DAPI layer mask that lie within
200 µm of the affine atlas hippocampus (CA + DG), mostly outside the ventricles, and are at least 150 px.
Piriform layer 2, choroid plexus
and other dense spots elsewhere used to dominate P90 for every method.

Each remaining piece is assigned once (from the affine fit) to DG or CA (whichever atlas layer is nearer),
so the classes do not move with the method under test:
  DG: DAPI granule-layer pixels -> nearest atlas DG-sg
  CA: DAPI pyramidal-layer pixels -> nearest atlas CA1-3 sp (dorsal CA1 is most of it)
  图谱DG→DAPI: atlas DG-sg pixels -> nearest DAPI granule layer (catches an atlas DG-sg that only got fatter;
  the CA detection is too patchy for the reverse direction)
"""
import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
from cfos.core.atlas import load_atlas
from scipy import ndimage as ndi

S = sys.argv[1]; tags = sys.argv[2].split(',')
st = load_atlas('CCFv3').structures


def group(*ids):
    return st[st.path.astype(str).apply(lambda p: any(f"/{i}/" in f"/{p}/" for i in ids))]['index'].values


DG, CA, HIP, VS = group(632), group(407, 446, 495), group(1080), group(73)   # HIP = Ammon horn + DG; VS = ventricles


def labels(n, tag, z):
    reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_{tag}.json")))
    H, W = R.orient(z['img'], reg.orientation).shape
    warp = np.load(f"{S}/{n}_{tag}_warp.npy") if reg.nonlinear else None
    return R.label_maps(reg, W, H, warp)[0], reg


def classes(n, z, pix):
    """0 = ignored, 1 = DG, 2 = CA; fixed per section (from the affine fit)."""
    obs = np.load(f"{S}/{n}_layers.npy")
    lab, _ = labels(n, 'ccf_affine', z)
    near = ndi.distance_transform_edt(~np.isin(lab, HIP)) * pix <= 200
    d_dg = ndi.distance_transform_edt(~np.isin(lab, DG))
    d_ca = ndi.distance_transform_edt(~np.isin(lab, CA))
    comp, k = ndi.label(obs)
    out = np.zeros(obs.shape, np.uint8)
    for c in range(1, k + 1):
        m = comp == c
        if m.sum() < 150 or near[m].mean() < .5 or np.isin(lab[m], VS).mean() > .3:   # ependyma / choroid plexus
            continue
        out[m] = 1 if np.median(d_dg[m]) <= np.median(d_ca[m]) else 2
    return out


rows = []
for n in ('S01', 'S02', 'S03'):
    z = np.load(f"{S}/{n}.npz"); pix = float(z['pix'])
    cls = classes(n, z, pix)
    for tag in tags:
        lab, reg = labels(n, tag, z)
        r = [n, tag]
        for k, ids in ((1, DG), (2, CA)):
            d = (ndi.distance_transform_edt(~np.isin(lab, ids)) * pix)[cls == k]
            r += [d.mean(), np.percentile(d, 90)] if d.size else [np.nan, np.nan]
        dg_atl = np.isin(lab, DG)
        r += [(ndi.distance_transform_edt(cls != 1) * pix)[dg_atl].mean() if dg_atl.any() else np.nan]
        d = np.concatenate([(ndi.distance_transform_edt(~np.isin(lab, ids)) * pix)[cls == k] for k, ids in ((1, DG), (2, CA))])
        r += [d.mean(), np.percentile(d, 90), R.dice(lab > 0, R.orient(z['mask'], reg.orientation))]
        rows.append(r)
hdr = f"{'slice':5} {'method':14} {'DG均值':>7} {'DG_P90':>7} {'CA均值':>7} {'CA_P90':>7} {'图谱DG→DAPI':>10} {'合计均值':>8} {'合计P90':>8} {'轮廓Dice':>8}"
print(hdr)
for r in rows:
    print(f"{r[0]:5} {r[1]:14} " + " ".join(f"{v:7.1f}" for v in r[2:6]) + f" {r[6]:10.1f} {r[7]:8.1f} {r[8]:8.1f} {r[9]:8.3f}")
import collections
agg = collections.defaultdict(list)
for r in rows: agg[r[1]].append(r[2:])
for k, v in agg.items():
    v = np.nanmean(np.array(v, float), 0)
    print(f"mean  {k:14} " + " ".join(f"{x:7.1f}" for x in v[:4]) + f" {v[4]:10.1f} {v[5]:8.1f} {v[6]:8.1f} {v[7]:8.3f}")
