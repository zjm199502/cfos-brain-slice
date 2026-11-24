"""Resample the CCF on the first flow before refining its cortex-facing surface."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours
from skimage.morphology import skeletonize
from .hippocampal_refine import propose, sample, trace, jacobian, dense_pullback, lock_mask


def surface_controls(gray, labels, ids):
    foreground=ndi.binary_fill_holes(ndi.binary_closing(gray>max(np.percentile(gray,95)*.09,3),iterations=2))
    py,px=np.gradient(ndi.gaussian_filter(ndi.distance_transform_edt(foreground),5))
    seed=np.isin(labels,[ids['CA1sp'],ids['CA2sp'],ids['CA3sp'],ids['DG-sg']])
    near=ndi.distance_transform_edt(~seed)
    controls=[]
    for layer in ("tract","pyramidal"):
        is_tract=layer=="tract"
        mask=np.isin(labels,ids['cc_tree']) if is_tract else labels==ids['CA3sp']
        smooth=ndi.gaussian_filter(mask.astype('f'),1.4)
        sy,sx=np.gradient(smooth)
        for curve in find_contours(smooth,.47):
            if len(curve)<22: continue
            y,x=curve[::2].T
            nx=-sample(sx,x,y); ny=-sample(sy,x,y)
            mag=np.hypot(nx,ny)
            nx/=np.maximum(mag,1e-5);ny/=np.maximum(mag,1e-5)
            valid=(mag>.025) & (sample(near,x,y)<(170 if is_tract else 10))
            if is_tract:
                valid &= nx*sample(px,x,y)+ny*sample(py,x,y)<-.48
            if valid.sum()<8: continue
            reach=15 if is_tract else 4
            off=np.arange(-reach,reach+.1,.5)
            profiles=[]
            for v in (-4,-2,0,2,4) if is_tract else (-1.5,0,1.5):
                profiles.append(sample(gray,x[:,None]+nx[:,None]*off-ny[:,None]*v,
                                            y[:,None]+ny[:,None]*off+nx[:,None]*v))
            profile=np.median(profiles,axis=0)
            grad=ndi.gaussian_filter1d(profile,2.7 if is_tract else 1.6,axis=1,order=1)
            # A sustained dark inner flank reduces attraction to isolated bright cells.
            if is_tract:
                pos=np.arange(len(off)); inner=profile[:,np.maximum(pos-7,0)]
                score=grad+.045*(np.median(profile,axis=1)[:,None]-inner)
            else:
                score=-grad
            score=score-.007*off**2
            breaks=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)))
            for start,stop in breaks.reshape(-1,2):
                if stop-start<7: continue
                seg=score[start:stop]
                ind=trace(seg,3,.26)
                shift=ndi.gaussian_filter1d(off[ind],2)
                quality=np.median(seg[np.arange(len(ind)),ind])
                if quality < (.5 if is_tract else 1.0): continue
                for q in range(1,len(ind)-1,3):
                    t=start+q
                    src=np.array([x[t],y[t]])
                    delta=shift[q]
                    if abs(delta)>=reach-1: continue
                    controls.append((layer,src,src+delta*np.array([nx[t],ny[t]])))
    return controls


def quality(gray,lab,ids):
    pyr=skeletonize(lab==ids['CA3sp'])
    if not pyr.any(): return np.array([-np.inf,-np.inf])
    ridge=(gray-ndi.gaussian_filter(gray,5))[pyr]
    wm=np.isin(lab,ids['cc_tree'])
    sm=ndi.gaussian_filter(wm.astype('f'),1.6)
    y,x=np.gradient(sm)
    norm=np.maximum(np.hypot(x,y),1e-4)
    near=ndi.distance_transform_edt(~pyr)<125
    tissue=ndi.binary_fill_holes(ndi.binary_closing(gray>max(np.percentile(gray,95)*.09,3),iterations=2))
    py,px=np.gradient(ndi.gaussian_filter(ndi.distance_transform_edt(tissue),5))
    choose=wm&~ndi.binary_erosion(wm)&near&((px*x+py*y)>.45*norm)
    if choose.sum()<15: return np.array([-np.inf,-np.inf])
    yy,xx=np.nonzero(choose);vx=x[choose]/norm[choose];vy=y[choose]/norm[choose]
    signal=ndi.gaussian_filter(gray,1.6)
    edge=sample(signal,xx-vx*3,yy-vy*3)-sample(signal,xx+vx*3,yy+vy*3)
    return np.array([np.median(ridge),np.median(edge)])


def surface_pass(field,info,gray,lab,ids,step,layer):
    controls=[p for p in surface_controls(gray,lab,ids) if p[0]==layer]
    if len(controls)<22:
        return field,{**info,"surface_controls":0},[]
    allca=np.isin(lab,[ids[k] for k in ('CA1sp','CA2sp','CA3sp','DG-sg')])
    keep=skeletonize(allca)
    if layer=="pyramidal":
        white=np.isin(lab,ids['cc_tree'])
        keep |= (white & ~ndi.binary_erosion(white)) & (ndi.distance_transform_edt(~allca)<130)
    y,x=np.nonzero(keep)
    pts=np.c_[x,y]
    fixed=pts[np.unique(pts//4,axis=0,return_index=True)[1]]
    pull=dense_pullback(controls,lab.shape,fixed)
    yy,xx=np.indices(field.shape[1:],dtype='f')
    u,v=[sample(ch,xx/step,yy/step)*step for ch in pull]
    for strength in (1,.7,.4):
        vx,vy=u*strength,v*strength
        proposal=np.stack([vx+sample(field[0],xx+vx,yy+vy),vy+sample(field[1],xx+vx,yy+vy)])
        j=float(jacobian(proposal).min())
        if j>=min(.06,info['min_jacobian']*.9):
            return proposal,{**info,'min_jacobian':j,'surface_controls':len(controls),'surface_strength':strength},controls
    return field,{**info,'surface_controls':0},[]


def refine_registration(structure, labels, hemispheres, prior, pixel_um, ids, render_labels):
    field,info,_=propose(structure,labels,hemispheres,prior,pixel_um,ids)
    if not info["applied"]: return field,info,[]
    step=max(1,round(16/pixel_um))
    gray=ndi.gaussian_filter(structure.astype("f"),.7*step)[::step,::step]
    lo,hi=np.percentile(gray,[5,99]);gray=np.clip((gray-lo)*140/max(hi-lo,1e-6),0,190)
    lab=render_labels(field)[::step,::step]
    score=quality(gray,lab,ids)
    history=[score.tolist()]
    controls=[]
    accepted=[]
    for layer,target in (("tract",1),("pyramidal",0)):
        for _ in range(3):
            candidate,meta,anchors=surface_pass(field,info,gray,lab,ids,step,layer)
            if not anchors: break
            trial_lab=render_labels(candidate)[::step,::step]
            trial=quality(gray,trial_lab,ids)
            gain=trial-score
            if np.any(~np.isfinite(gain)) or gain[1-target]<-max(.9,.08*abs(score[1-target])) or gain[target]<.1:
                break
            field,info,controls,lab,score=candidate,meta,anchors,trial_lab,trial
            history.append(score.tolist())
            accepted.append(layer)
    # Preserve accepted neighboring labels where the entire local correction is small.
    # This is applied after the large-displacement fit, with a native-grid topology check.
    y,x=np.indices(labels.shape,dtype='f')
    mobile=sample(lock_mask(labels[::step,::step],ids),x/step,y/step)
    delta=field-prior
    magnitude=np.hypot(delta[0],delta[1])*pixel_um
    local=ndi.maximum_filter(magnitude,size=max(3,round(75/pixel_um)))
    weight=(1-mobile)*np.clip((120-local)/60,0,1)
    weight=ndi.gaussian_filter(weight,.7*step)
    for t in (1,.7,.4):
        restored=(field-t*weight[None,:,:]*delta).astype(np.float32)
        j=float(jacobian(restored).min())
        if j>=min(.06,info['min_jacobian']*.9):
            field=restored
            info={**info,'min_jacobian':j,'protection_strength':t}
            break
    return field,{**info,"surface_scores":history,"accepted_surfaces":accepted},controls
