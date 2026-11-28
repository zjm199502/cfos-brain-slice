import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
from cfos.core.atlas import load_atlas
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
S = sys.argv[1]; tags = sys.argv[2].split(',')
st = load_atlas('CCFv3').structures
def group(*ids):
    idx = st[st.path.astype(str).apply(lambda p: any(f"/{i}/" in f"/{p}/" for i in ids))]['index'].values
    return idx
GROUPS = {'layers': (group(632, 407, 446, 495), (255,60,60)),  # DG-sg, CA*sp
          'fiber': (group(1009), (60,200,255)), 'vent': (group(73), (60,255,60))}
def enhance(img):
    x = ndi.gaussian_filter(img.astype(np.float32), 1.0)
    hi = np.percentile(x[x > 0], 99.7) if (x > 0).any() else 1
    return (np.clip(x / hi, 0, 1) ** 0.6 * 255).astype(np.uint8)
def edges(m): return m ^ ndi.binary_erosion(m)
def ov(img, lab, title):
    rgb = np.stack([enhance(img)]*3, -1)
    rgb[edges(lab > 0)] = (255,255,255)
    for name,(ids,col) in GROUPS.items():
        e = edges(np.isin(lab, ids)); e = ndi.binary_dilation(e)
        rgb[e] = col
    im = Image.fromarray(rgb); ImageDraw.Draw(im).text((10,10), title, fill=(255,255,0)); return im
for n in ('S01','S02','S03'):
    z = np.load(f"{S}/{n}.npz"); ims=[]
    for tag in tags:
        reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_{tag}.json")))
        img = R.orient(z['img'], reg.orientation); H, W = img.shape
        warp = np.load(f"{S}/{n}_{tag}_warp.npy") if reg.nonlinear else None
        lab, _ = R.label_maps(reg, W, H, warp)
        ims.append(ov(img, lab, f"{n} {tag}"))
    w = sum(i.width for i in ims); out = Image.new('RGB', (w, ims[0].height)); x=0
    for i in ims: out.paste(i,(x,0)); x+=i.width
    out.save(f"{S}/{n}_{'_vs_'.join(tags)}.png")
