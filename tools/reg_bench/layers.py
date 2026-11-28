import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
from scipy import ndimage as ndi
from PIL import Image
S=sys.argv[1]
for n in ('S01','S02','S03'):
    z=np.load(f"{S}/{n}.npz"); reg=R.Registration.from_dict(json.load(open(f"{S}/{n}_ccf_affine.json")))
    img=R.orient(z['img'],reg.orientation).astype(np.float32); mask=R.orient(z['mask'],reg.orientation)
    d=ndi.gaussian_filter(img,2.0)
    bg=ndi.gaussian_filter(img,25.0)
    r=d/(bg+1)
    inner=ndi.distance_transform_edt(mask)>25
    from cfos.core.atlas import load_atlas
    st=load_atlas('CCFv3').structures
    hpf=st[st.path.astype(str).str.contains('/1089/')]['index'].values
    lab0,_=R.label_maps(reg,img.shape[1],img.shape[0],None)
    roi=ndi.binary_dilation(np.isin(lab0,hpf),iterations=60)&inner
    np.save(f'{S}/{n}_roi.npy',roi)
    thr=np.percentile(r[roi],93)
    m=(r>thr)&roi
    m=ndi.binary_opening(m,iterations=1)
    lab,k=ndi.label(m); sz=np.bincount(lab.ravel()); sz[0]=0
    keep=np.isin(lab,np.nonzero(sz>=150)[0])
    np.save(f"{S}/{n}_layers.npy",keep)
    rgb=np.stack([np.clip(img/np.percentile(img,99.7),0,1)**.6*255]*3,-1).astype(np.uint8); rgb[keep]=(rgb[keep]*0.4+np.array([255,0,0])*0.6).astype(np.uint8)
    Image.fromarray(rgb).save(f"{S}/{n}_layers.png"); print(n,thr,keep.sum())
