"""python run.py S method  -> writes {n}_{method}.json/_warp.npy"""
import sys, json, os, time, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
S, method = sys.argv[1], sys.argv[2]
os.chdir('/')   # reproduce the Finder-launched app
for n in ('S01','S02','S03'):
    z = np.load(f"{S}/{n}.npz"); reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_ccf_affine.json")))
    img, mask, pix = z['img'], z['mask'], float(z['pix'])
    t = time.time()
    if method == 'pipeline':
        R.refit_outline(reg, img, mask, S + '/res', pix)
        warp = np.load(R.warp_path(S + '/res', '', '', reg)) if reg.nonlinear else None
    elif method.startswith('deep'):
        from cfos.core import deep_refine
        reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_pipeline.json")))
        prior = tuple(np.load(f"{S}/{n}_pipeline_warp.npy")) if reg.nonlinear else None
        warp, reg.metrics = deep_refine.refine(reg, R.orient(img, reg.orientation), R.orient(mask, reg.orientation), prior, pix)
        if warp is None: warp = prior
        else: reg.nonlinear = True
    elif method == 'full':
        from cfos.core import deep_refine
        reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_pipeline.json")))
        prior = tuple(np.load(f"{S}/{n}_pipeline_warp.npy")) if reg.nonlinear else None
        oi, om = R.orient(img, reg.orientation), R.orient(mask, reg.orientation)
        changed, rep = deep_refine.search_plane(reg, oi, om, prior, pix); print(' plane', rep)
        if changed:
            for k, v in changed.items(): setattr(reg, k, v)
            R.refit_outline(reg, img, mask, S + '/res', pix)
            prior = tuple(np.load(R.warp_path(S + '/res', '', '', reg))) if reg.nonlinear else None
        warp, reg.metrics = deep_refine.refine(reg, oi, om, prior, pix)
        if warp is None: warp = prior
        else: reg.nonlinear = True
    elif method == 'e2e':
        R.refit_outline(reg, img, mask, S + '/res', pix, search_plane=True)
        warp = np.load(R.warp_path(S + '/res', '', '', reg)) if reg.nonlinear else None
        print(' ', reg.nl_method, reg.ap_auto, reg.tilt_dv, reg.tilt_ml)
    elif method == 'legacy':
        warp = R.nonlinear_fit(reg, R.orient(mask, reg.orientation), R.orient(img, reg.orientation)); reg.nonlinear = True
    print(n, method, f"{time.time()-t:.1f}s", reg.nonlinear, json.dumps(reg.metrics, ensure_ascii=False, default=float)[:400], flush=True)
    json.dump(reg.to_dict(), open(f"{S}/{n}_{method}.json", "w"), default=float)
    if warp is not None: np.save(f"{S}/{n}_{method}_warp.npy", np.asarray(warp, np.float32))
