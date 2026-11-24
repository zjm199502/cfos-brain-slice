import unittest
from unittest.mock import patch

import numpy as np
from cfos.core.edge_drag import deform_paths, snap_path, transport
from cfos.core import registration as R
from cfos.core.atlas import Atlas, SHAPE_ASR
from cfos.core.sections import Section


class FineEdges(unittest.TestCase):
    def test_tip_and_round_trip_with_subpixel_brush(self):
        W, H = 100, 60
        start = np.array([40.2,18.9])
        end = start + [2.25,-1.5]
        pt = [*(start/[W,H]),*(end/[W,H]),.75/W,2]
        rng = np.random.default_rng(2)
        x,y = (rng.uniform(-2,4,(2000,2)) + start).T
        x,y = np.r_[start[0],x],np.r_[start[1],y]
        u,v = transport(x,y,pt,W,H)
        np.testing.assert_allclose([u[0],v[0]],end)
        np.testing.assert_allclose(transport(u,v,pt,W,H,True),[x,y],atol=1e-9)
        far = y>start[1]+.75
        np.testing.assert_array_equal(u[far],x[far])

    def test_long_polylines_remain_editable_after_decimation(self):
        path = np.array([[0,20],[100,20]],'f')
        p = [.20,.455,.21,.455,.01,2]
        new, = deform_paths([path],p,100,100)
        np.testing.assert_allclose(snap_path([new],21,45.5,.05),[21,45.5],atol=.025)
        self.assertIsNone(snap_path([new],25,45.5,.2))

    def test_cell_grid_and_roi_agree_after_sequential_strokes_and_flip(self):
        # Real sampler and composition against a planar interface, including
        # points whose exact subpixel location differs from a thumbnail lookup.
        w,h=80,60
        x=np.arange(456)
        lab=np.broadcast_to(np.where(x<240,3,7).astype("H"),(3,3,456))
        hemi=np.broadcast_to(np.where(x<240,1,2).astype("B"),lab.shape)
        atlas=Atlas(lab,None,hemi,None)
        reg=R.Registration(plane=[455-200,526,318,-80,0,0,0,0,0],mirror=True,nonlinear=True)
        # At x=39.5 the nearest reference voxel switches from 239 to 240.
        points=[[39.5/w,.5,40.5/w,.5,1.25/w,2],[40.5/w,.5,41.5/w,.5,1.25/w,2]]
        reg.control_points=points
        sec=Section(1,(12,18,w,h),[])
        zero=np.zeros((2,h,w),'f')
        with patch('cfos.core.registration.load_atlas',return_value=atlas):
            labels,sides=R.label_maps(reg,w,h,zero)
            for orient in ('r0','r180'):
                reg.orientation=orient
                xx=np.array([40.9,41.8,37.]); yy=np.array([30.,30.,30.])
                sx,sy=(xx,yy) if orient=='r0' else (w-xx,h-yy)
                vals,_=R.cell_labels(sx+12,sy+18,sec,reg,1,labels,sides,zero)
                self.assertEqual(int(vals[0]),3)
                self.assertEqual(int(vals[1]),7)
                roi=R.registered_samples(reg,w,h,zero,xx[:1],yy[:1])
                self.assertEqual(roi['annotation'][0],vals[0])
                self.assertEqual(roi['hemispheres'][0],2)
            self.assertEqual(labels.shape,(h,w))

    def test_legacy_points_keep_their_field(self):
        reg=R.Registration(plane=[220,290,200,-10,0,0,0,0,-10])
        for point in ([.3,.3,.4,.4],[.3,.3,.4,.4,.05]):
            reg.control_points=[point]
            expected = R.tps_warp(reg.control_points,60,44)
            with patch('cfos.core.registration.sample_plane',return_value={
                    'annotation':np.zeros((44,60)),'hemispheres':np.zeros((44,60))}) as render:
                R.label_maps(reg,60,44)
            np.testing.assert_array_equal(render.call_args.kwargs['warp'],expected)
