"""Local pull-back refinement on a prealigned, oriented coronal section.
Only the explicit structure channel is read. All fields are xy, in native pixels.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def jacobian(field):
    xy, xx = np.gradient(field[0])
    yy, yx = np.gradient(field[1])
    return (1 + xx) * (1 + yy) - xy * yx


def sample(img, x, y, order=1):
    return ndi.map_coordinates(img, [y, x], order=order, mode="nearest", prefilter=False)


def trace(score, speed, penalty):
    """Highest-scoring open curve on a polar grid; consecutive radii stay connected."""
    best = score[0].copy()
    parent = np.zeros(score.shape, np.int16)
    shifts = np.arange(-speed, speed + 1)
    columns = np.arange(len(best))
    indexes = np.clip(columns[None, :] - shifts[:, None], 0, len(best) - 1)
    for j in range(1, len(score)):
        options = best[indexes] - penalty * np.abs(shifts[:, None]) ** 1.35
        choice = options.argmax(axis=0)
        best = options[choice, columns] + score[j]
        parent[j] = indexes[choice, columns]
    path = np.empty(len(score), int)
    path[-1] = best.argmax()
    for j in range(len(score)-1, 0, -1):
        path[j-1] = parent[j, path[j]]
    return path


def smooth_trace(score, speed=7):
    """Order-two Viterbi. Curvature continuity bridges local gaps in the tract."""
    steps=np.arange(-speed,speed+1)
    columns=np.arange(score.shape[1])
    best=np.repeat(score[0,:,None],len(steps),axis=1)
    prev=np.zeros((*score.shape,len(steps)),np.int8)
    transition=1.1*(steps[:,None]-steps[None,:])**2
    for j in range(1,len(score)):
        after=np.empty_like(best)
        for t,v in enumerate(steps):
            base=best[np.clip(columns-v,0,len(columns)-1),:]-transition[t]
            choice=base.argmax(axis=1)
            after[:,t]=score[j]+base[columns,choice]-.15*v*v
            prev[j,:,t]=choice
        best=after
    r,k=np.unravel_index(best.argmax(),best.shape)
    path=np.empty(len(score),int)
    for j in range(len(score)-1,-1,-1):
        path[j]=r
        r,k=int(np.clip(r-steps[k],0,len(columns)-1)),prev[j,r,k]
    return path


def shifted(profile, offset):
    return profile[:, np.clip(np.arange(profile.shape[1]) + offset, 0, profile.shape[1]-1)]


def side_controls(gray, labels, hemi, side, ids, unit):
    dentate = (labels == ids["DG-sg"]) & (hemi == side)
    y, x = np.nonzero(dentate)
    if len(x) < 45:
        return [], {}
    other_x=np.where((labels==ids["DG-sg"]) & (hemi!=side) & (hemi!=0))[1]
    if len(other_x)<45:
        return [], {}
    sign = 1 if x.mean() > other_x.mean() else -1
    lateral = sign*x >= np.quantile(sign*x, .68)
    width = np.ptp(x)
    cx = x[lateral].mean() + sign*.06*width
    cy = y[lateral].mean() - .21*width
    theta = np.arange(-50, 236, 1.)
    radius = np.arange(0, min(int(7200/unit), int(gray.shape[1]*.60)))
    dx = (sign * np.sin(np.deg2rad(theta)))[:,None]
    dy = -np.cos(np.deg2rad(theta))[:,None]
    xx, yy = cx+dx*radius, cy+dy*radius
    profiles = sample(gray,xx,yy)
    sub = sample(labels,xx,yy,0)
    same = sample(hemi,xx,yy,0)==side
    ca = np.isin(labels,[ids[k] for k in ("CA1sp","CA2sp","CA3sp")]) & (hemi==side)
    ca_skeleton = skeletonize(ca)
    skdist = ndi.distance_transform_edt(~ca_skeleton)
    ds = sample(skdist, xx,yy)
    # Extract a connected atlas reference with support from an actual labelled layer.
    atlas_score = -np.minimum(ds,12) + .003*radius
    ca_source = trace(atlas_score,5,.17)
    near = ds[np.arange(len(theta)),ca_source] < 4
    if np.count_nonzero(near & (theta>35) & (theta<160)) < 50:
        return [], {}
    rr = radius[None,:]
    signal = profiles - .5*(shifted(profiles,-5)+shifted(profiles,5))
    signal_wide = profiles - .5*(shifted(profiles,-9)+shifted(profiles,9))
    ridge = .72*signal+.28*signal_wide
    # Broader priors allow bent arcs. A sustained bright band must still dominate.
    ca_score = ridge - .0009*(rr-ca_source[:,None])**2
    ca_score[(rr<10) | (np.abs(rr-ca_source[:,None])>64)] = -1e4
    ca_target = trace(ca_score,5,.38)
    sel = np.arange(len(theta))
    # A structural layer has a locally consistent gradient orientation.
    iy,ix=np.gradient(ndi.gaussian_filter(gray,1.1))
    st_x=ndi.gaussian_filter(ix*ix,4)
    st_y=ndi.gaussian_filter(iy*iy,4)
    st_xy=ndi.gaussian_filter(ix*iy,4)
    coherence=np.hypot(st_x-st_y,2*st_xy)/(st_x+st_y+1e-5)
    arc=(theta>=30)&(theta<=150)&near
    if not np.any(arc) or np.median(sample(coherence,xx[sel,ca_target],yy[sel,ca_target])[arc])<.52:
        return [], {}
    points = []

    def pair(kind, i, r0, r1):
        a=np.array([cx+dx[i,0]*r0,cy+dy[i,0]*r0])
        b=np.array([cx+dx[i,0]*r1,cy+dy[i,0]*r1])
        points.append((kind,a,b))

    ca_good = near & (ridge[sel,ca_target] >= 3.) & (theta>=-25) & (theta<226)
    if np.count_nonzero(ca_good & (theta>50) & (theta<160))<60:
        return [], {}
    ca3_dist=sample(ndi.distance_transform_edt(labels!=ids["CA3sp"]),xx,yy)
    ca_good &= ca3_dist[sel,ca_source]<11
    for i in np.flatnonzero(ca_good)[::2]:
        pair("CA",i,ca_source[i],ca_target[i])

    white = np.isin(sub,ids["cc_tree"]) & same
    cutoff = (theta>=-32) & (theta<=142)
    segments=[]
    for i in np.flatnonzero(cutoff):
        starts=np.where(np.diff(np.r_[False,white[i],False].astype(int))==1)[0]
        ends=np.where(np.diff(np.r_[False,white[i],False].astype(int))==-1)[0]-1
        pool=[(a,b) for a,b in zip(starts,ends) if b-a>=2 and a>20 and b<len(radius)-2]
        if pool:
            a,b=min(pool,key=lambda v: abs(v[0]-ca_source[i]-20))
            segments.append((i,a,b))
    if len(segments)<35:
        return points, {}
    i0,lo,hi = np.asarray(segments).T
    rows=np.arange(i0.min(),i0.max()+1)
    lower=np.interp(rows,i0,lo)
    upper=np.interp(rows,i0,hi)
    po=ndi.gaussian_filter(profiles,(1.6,1.5))
    # Tissue outside a white matter band is brighter, and stays bright for several samples.
    edge=(shifted(po,4)-shifted(po,-4))*.68+(shifted(po,9)-shifted(po,-7))*.32
    inner_edge=(shifted(po,-4)-shifted(po,4))
    # Find the dark tract adjoining the cortical mantle. Bright cell clusters also
    # have a leading edge but lack a consistently dark inner flank.
    arch=(theta>=-15)&(theta<=40)
    dark=np.percentile(po[sel[arch],np.minimum(ca_target[arch]+10,len(radius)-1)],40)
    inside=np.mean([shifted(po,-j) for j in (3,5,7)],axis=0)
    outside=np.mean([shifted(po,j) for j in (7,12,18,24)],axis=0)
    sustained=np.minimum(outside-inside,55)
    darkness=np.maximum(inside-dark-4,0)
    response=.55*edge+.65*sustained-1.2*darkness
    # A branch near the hippocampal tip can be much darker than the adjacent tract.
    # Estimate the cortical mantle depth on this hemisphere to disambiguate branches.
    fg=ndi.binary_fill_holes(ndi.binary_closing(gray>max(np.percentile(gray,95)*.09,3),iterations=2))
    depth=ndi.distance_transform_edt(fg)
    depths=sample(depth,xx,yy)
    dorsal=(theta[rows]>=-10)&(theta[rows]<=30)
    if np.count_nonzero(dorsal)<8:
        return points, {}
    local_depth=float(np.median(depths[rows[dorsal],np.rint(upper[dorsal]).astype(int)]))
    # Lateral cortex is thicker here. Scale the prior to the specimen's own dorsal depth.
    lateral_fraction=np.clip((theta[rows]-12)/83,0,1)
    expected=np.clip(local_depth,42,100)*(1+.41*lateral_fraction)
    deviation=np.maximum(np.abs(depths[rows]-expected[:,None])-8,0)
    # Saturation limits the leverage of an isolated hot edge at the wrong depth.
    depth_weight=.012+.10*np.clip((theta[rows]-45)/45,0,1)
    score=np.clip(response[rows],-30,35)-depth_weight[:,None]*deviation**2-.00018*(rr-upper[:,None])**2
    lowlimit=np.maximum(ca_target[rows]+9,upper-120)
    highlimit=upper+64
    score[(rr<lowlimit[:,None]) | (rr>highlimit[:,None])]=-1e4
    outer_raw=smooth_trace(score)
    outer=np.rint(ndi.gaussian_filter1d(outer_raw.astype(float),2.3)).astype(int)
    if np.median(edge[rows,outer]) < 3:
        return points, {}
    widths=np.maximum(upper-lower,3)
    score_in=inner_edge[rows]-.065*(rr-(outer-widths)[:,None])**2
    valid=(rr<=outer[:,None]-3)&(rr>=outer[:,None]-np.minimum(42,widths*1.7+5)[:,None])
    score_in[~valid]=-1e4
    inner=trace(score_in,6,.4)
    for j in range(0,len(i0),3):
        i=i0[j]; k=i-rows[0]
        adjacent=np.arange(max(0,k-4),min(len(rows),k+5))
        if np.median(edge[rows[adjacent],outer[adjacent]])<2 or score[k,outer[k]] < -100:
            continue
        pair("cc_outer",i,hi[j]+.5,outer[k])
        if inner_edge[i,inner[k]]>3 and inner[k]+3<=outer[k]:
            pair("cc_inner",i,lo[j]-.5,inner[k])
    return points, {"side":side,"origin":[cx,cy]}




def lock_mask(labels,ids):
    # Avoid incidental changes to previously accepted CA1 and granule layers.
    safe=(labels==ids["DG-sg"]) | (labels==ids["CA1sp"])
    safe &= ndi.distance_transform_edt(labels!=ids["CA3sp"])>11
    t=np.clip((ndi.distance_transform_edt(~safe)-1)/10,0,1)
    return t*t*(3-2*t)


def dense_pullback(points, shape, protected):
    src=np.array([a for _,a,b in points])
    dst=np.array([b for _,a,b in points])
    H,W=shape
    dy,dx=np.mgrid[0:H:36,0:W:36]
    grid=np.c_[dx.ravel(),dy.ravel()]
    mids=np.vstack((src,dst,(src+dst)*.5))
    grid=grid[cKDTree(mids).query(grid)[0]>65]
    fixed=np.vstack((grid,protected))
    yy,xx=np.mgrid[:H,:W].astype("f")
    gy,gx=np.mgrid[:H+6:5,:W+6:5]
    coarse=np.c_[gx.ravel(),gy.ravel()]
    accum=np.zeros((2,H,W),"f")
    # Refit short, ordered steps to avoid folds when neighboring arcs move differently.
    n=6
    for k in range(n):
        loc=src+(dst-src)*(k+1)/n
        nodes=np.vstack((loc,fixed))
        disp=np.vstack(((src-dst)/n,np.zeros_like(fixed)))
        fit=RBFInterpolator(nodes,disp,kernel="thin_plate_spline",smoothing=11.)
        v=fit(coarse).reshape((*gx.shape,2))
        vx,vy=[ndi.map_coordinates(v[:,:,j],[yy/5,xx/5],order=3,mode="nearest").astype("f") for j in (0,1)]
        # Exponentiate the incremental smooth velocity for a topology-preserving flow.
        vx=ndi.gaussian_filter(vx,1.2)/8
        vy=ndi.gaussian_filter(vy,1.2)/8

        for _ in range(3):
            vx,vy=vx+sample(vx,xx+vx,yy+vy),vy+sample(vy,xx+vx,yy+vy)
        accum=np.stack([vx+sample(accum[0],xx+vx,yy+vy),vy+sample(accum[1],xx+vx,yy+vy)])
    return accum


def propose(structure, labels, hemispheres, prior, pixel_um, ids):
    if structure.ndim!=2 or labels.shape!=structure.shape or hemispheres.shape!=labels.shape or prior.shape!=(2,*labels.shape):
        raise ValueError("Registration inputs must share the same oriented grid")
    if min(labels.shape)<3 or not np.isfinite(pixel_um) or pixel_um<=0 or not np.isfinite(prior).all() or not np.isfinite(structure).all():
        raise ValueError("Invalid pixel size or displacement field")
    if float(jacobian(prior).min())<=0:
        return prior,{"applied":False,"reason":"folded prior"},[]
    step=max(1,round(16/pixel_um))
    unit=pixel_um*step
    if not 12<=unit<=20:
        return prior, {"applied":False,"reason":"unsupported sample spacing"}, []
    a=ndi.gaussian_filter(structure.astype("f"),max(.8,step*.7))[::step,::step]
    percent=np.percentile(a,[5,99])
    if percent[1]-percent[0]<1e-6:
        return prior, {"applied":False,"reason":"no structural contrast"}, []
    a=np.clip((a-percent[0])*140/(percent[1]-percent[0]),0,190)
    lab,hemi=labels[::step,::step],hemispheres[::step,::step]
    points=[]
    for side in (1,2):
        extra,_=side_controls(a,lab,hemi,side,ids,unit)
        if len(extra)<18:
            return prior, {"applied":False,"reason":"bilateral landmarks unsupported"}, []
        points.extend(extra)
    if len(points)<18:
        return prior, {"applied":False,"reason":"insufficient supported arcs"}, []
    # Preserve identified granule blades, resampled spatially rather than by array order.
    stable_ca=(lab==ids["CA1sp"]) & (ndi.distance_transform_edt(lab!=ids["CA3sp"])>18)
    y,x=np.nonzero(skeletonize(lab==ids["DG-sg"]) | skeletonize(stable_ca))
    fixed=np.c_[x,y];fixed=fixed[np.unique(fixed//5,axis=0,return_index=True)[1]]
    pull=dense_pullback(points,lab.shape,fixed)
    yy,xx=np.indices(labels.shape,dtype="f")
    vx,vy=[sample(ch,xx/step,yy/step)*step for ch in pull]
    floor=min(.060,float(jacobian(prior).min())*.90)
    for strength in (1.,.94,.85,.75,.63,.50,.35,.20):
        u,v=vx*strength,vy*strength
        field=np.stack((u+sample(prior[0],xx+u,yy+v),v+sample(prior[1],xx+u,yy+v)))
        j=float(jacobian(field).min())
        if j>=floor and j>0:
            return field,{"applied":True,"strength":strength,"min_jacobian":j,"controls":len(points)},points
    return prior,{"applied":False,"reason":"topology guard"},points
