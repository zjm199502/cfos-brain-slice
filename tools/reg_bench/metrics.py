"""python metrics.py S tag1,tag2,...  -> distance of DAPI-detected cell layers to the atlas cell layers.

Only detections inside the hippocampus count: connected pieces of the DAPI layer mask that lie within
200 µm of the affine atlas hippocampus (CA + DG), mostly outside the ventricles, and are at least 150 px.
Piriform layer 2, choroid plexus
and other dense spots elsewhere used to dominate P90 for every method.

Each remaining pixel is assigned once (from the affine fit) to DG, CA1 or CA3 (nearest atlas layer, smoothed
over ~40 µm), so the classes do not move with the method under test:
  DG: DAPI granule-layer pixels -> nearest atlas DG-sg
  CA1: DAPI pyramidal-layer pixels -> nearest atlas CA1/CA2 sp;  CA3: -> nearest atlas CA3 sp
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


DG, CA, HIP, VS = group(632), group(407, 446, 495), group(1080), group(73)
CA1, CA3 = group(407, 446), group(495)   # CA1 (+ the thin CA2) and CA3 pyramidal layers   # HIP = Ammon horn + DG; VS = ventricles


def labels(n, tag, z):
    reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_{tag}.json")))
    H, W = R.orient(z['img'], reg.orientation).shape
    warp = np.load(f"{S}/{n}_{tag}_warp.npy") if reg.nonlinear else None
    return R.label_maps(reg, W, H, warp)[0], reg


def classes(n, z, pix):
    """0 = ignored, 1 = DG, 2 = CA1/CA2, 3 = CA3; fixed per section (from the affine fit).
    Per pixel (nearest of the three atlas layers, smoothed over ~40 µm): CA3 is often one piece with the DG."""
    obs = np.load(f"{S}/{n}_layers.npy")
    lab, _ = labels(n, 'ccf_affine', z)
    near = ndi.distance_transform_edt(~np.isin(lab, HIP)) * pix <= 200
    comp, k = ndi.label(obs)
    keep = np.zeros(obs.shape, bool)
    for c in range(1, k + 1):
        m = comp == c
        if m.sum() >= 150 and near[m].mean() >= .5 and np.isin(lab[m], VS).mean() <= .3:   # not ependyma / plexus
            keep |= m
    d = np.stack([ndi.distance_transform_edt(~np.isin(lab, ids)) for ids in (DG, CA1, CA3)])
    vote = np.stack([ndi.gaussian_filter((keep & (d.argmin(0) == i)).astype(np.float32), 40 / pix) for i in range(3)])
    out = np.where(keep, vote.argmax(0) + 1, 0).astype(np.uint8)
    return out


rows = []
TARGET = {1: DG, 2: CA1, 3: CA3}
for n in ('S01', 'S02', 'S03'):
    z = np.load(f"{S}/{n}.npz"); pix = float(z['pix'])
    cls = classes(n, z, pix)
    for tag in tags:
        lab, reg = labels(n, tag, z)
        r = [n, tag]
        alld = []
        for k in (1, 2, 3):
            d = (ndi.distance_transform_edt(~np.isin(lab, TARGET[k])) * pix)[cls == k]
            alld.append(d)
            r += [d.mean(), np.percentile(d, 90)] if d.size else [np.nan, np.nan]
        dg_atl = np.isin(lab, DG)
        r += [(ndi.distance_transform_edt(cls != 1) * pix)[dg_atl].mean() if dg_atl.any() else np.nan]
        d = np.concatenate(alld)
        r += [d.mean(), np.percentile(d, 90), R.dice(lab > 0, R.orient(z['mask'], reg.orientation))]
        rows.append(r)
cols = ['DG均值', 'DG_P90', 'CA1均值', 'CA1_P90', 'CA3均值', 'CA3_P90', '图谱DG→DAPI', '合计均值', '合计P90', '轮廓Dice']
print(f"{'slice':5} {'method':26} " + " ".join(f"{c:>9}" for c in cols))
fmt = lambda v: " ".join(f"{x:9.1f}" for x in v[:-1]) + f" {v[-1]:9.3f}"
for r in rows:
    print(f"{r[0]:5} {r[1]:26} " + fmt(r[2:]))
import collections
agg = collections.defaultdict(list)
for r in rows: agg[r[1]].append(r[2:])
for k, v in agg.items():
    print(f"mean  {k:26} " + fmt(np.nanmean(np.array(v, float), 0)))
