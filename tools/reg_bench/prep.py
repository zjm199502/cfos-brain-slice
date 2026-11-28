import sys, json, pickle, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from cfos.core.slide_io import open_slide
from cfos.core.sections import Section
from cfos.core import registration as R, autoreg
from cfos.core.atlas_choice import CCF
out = sys.argv[1]  # output folder; sys.argv[2] = cfos_project.json
d = json.load(open(sys.argv[2]))
im = d['images'][0]
slide = open_slide(im['path'], z_mode='max')
print(slide.pixel_size_um, [c.name for c in slide.channels])
for s in im['sections']:
    sec = Section(**{k: s[k] for k in ('id','bbox0','contour0','name')})
    thumb = R.section_thumbnail(slide, sec, 0)
    img, mask, ds = thumb
    pix = ds*slide.pixel_size_um
    reg = autoreg.fit_section(slide, sec, 0, out, False, CCF, None, thumb=thumb)
    np.savez_compressed(f"{out}/{sec.name}.npz", img=img, mask=mask, ds=ds, pix=pix)
    json.dump(reg.to_dict(), open(f"{out}/{sec.name}_ccf_affine.json","w"))
    print(sec.name, img.shape, pix, reg.initialization)
