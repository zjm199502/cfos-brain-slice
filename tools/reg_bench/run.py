"""python run.py S method [spacing_um]  -> writes {n}_{method}.json/_warp.npy

method 'series': joint AP positioning of S01-S03 by slice order (and spacing, if given) before the e2e fit."""
import sys, json, os, time, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core import registration as R
S, method = sys.argv[1], sys.argv[2]
os.chdir('/')   # reproduce the Finder-launched app
NAMES = ('S01','S02','S03')
joint = {"bounds": {}, "notes": {}}
if method == 'series':
    from types import SimpleNamespace
    from cfos.core import series
    spacing = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    new = {("bench", n): R.Registration.from_dict(json.load(open(f"{S}/{n}_ccf_affine.json"))) for n in NAMES}
    im = SimpleNamespace(key="bench", source_id="bench", sections=[SimpleNamespace(name=n, enabled=True, registration={}) for n in NAMES])
    proj = SimpleNamespace(images=[im], samples={}, registration_atlas=new[("bench", "S01")].fit_atlas, section_spacing_um=spacing)
    for n in NAMES: print(n, 'giRAff AP', f"{R.adjusted_plane(new[('bench', n)]).ap_mm():+.3f} mm")
    joint = series.position_series(proj, new)
    print(' series', json.dumps(joint["groups"], ensure_ascii=False), {k[1]: [round(v * 25) for v in b] for k, b in joint["bounds"].items()}, 'µm')
for n in NAMES:
    z = np.load(f"{S}/{n}.npz"); reg = R.Registration.from_dict(json.load(open(f"{S}/{n}_ccf_affine.json")))
    if method == 'series': reg = new[("bench", n)]
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
    elif method == 'series':
        R.refit_outline(reg, img, mask, S + '/res', pix, search_plane=True, ap_bounds=joint["bounds"].get(("bench", n)))
        warp = np.load(R.warp_path(S + '/res', '', '', reg)) if reg.nonlinear else None
        reg.warnings += joint["notes"].get(("bench", n), [])
        print(' ', reg.nl_method, f"AP {R.adjusted_plane(reg).ap_mm():+.3f} mm", reg.ap_auto, reg.tilt_dv, reg.tilt_ml, reg.warnings)
    elif method == 'legacy':
        warp = R.nonlinear_fit(reg, R.orient(mask, reg.orientation), R.orient(img, reg.orientation)); reg.nonlinear = True
    print(n, method, f"{time.time()-t:.1f}s", reg.nonlinear, json.dumps(reg.metrics, ensure_ascii=False, default=float)[:400], flush=True)
    json.dump(reg.to_dict(), open(f"{S}/{n}_{method}.json", "w"), default=float)
    if warp is not None: np.save(f"{S}/{n}_{method}_warp.npy", np.asarray(warp, np.float32))
