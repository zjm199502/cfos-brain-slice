"""Portable regression checks. Run with python -B -m unittest test_refinement."""
import unittest
import numpy as np
from cfos.core.hippocampal_refine import jacobian, smooth_trace, trace
from cfos.core.surface_refine import refine_registration


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.img=np.zeros((120,220),np.float32)
        self.lab=np.zeros_like(self.img,dtype=np.int32)
        self.hemi=self.lab.copy()
        self.field=np.zeros((2,*self.img.shape),np.float32)
        self.ids={'DG-sg':350,'CA1sp':334,'CA2sp':339,'CA3sp':345,'cc_tree':[793,795,799,800]}

    def check_skip(self,reason,um=8):
        def unexpected_render(_):
            self.fail("Rejected proposals should not require rendering")
        out,meta,_=refine_registration(self.img,self.lab,self.hemi,self.field,um,self.ids,unexpected_render)
        self.assertIs(out,self.field)
        self.assertEqual(meta['reason'],reason)

    def test_blank(self):
        self.check_skip('no structural contrast')

    def test_absent_landmarks(self):
        self.img[:]=np.arange(self.img.shape[1])
        self.check_skip('bilateral landmarks unsupported')

    def test_already_folded_field(self):
        self.field[0]=-2*np.arange(self.img.shape[1])
        self.check_skip('folded prior')

    def test_grid_units(self):
        self.check_skip('unsupported sample spacing',um=45)

    def test_non_finite(self):
        self.img[1,1]=np.nan
        with self.assertRaises(ValueError):
            refine_registration(self.img,self.lab,self.hemi,self.field,8,self.ids,None)

    def test_adapter_preserves_manual_work(self):
        from types import SimpleNamespace
        from cfos.core.fosfolio_adapter import preview_hippocampal_alignment
        for registration,prior in [(SimpleNamespace(nonlinear=False,control_points=[]),None),
                                   (SimpleNamespace(nonlinear=True,control_points=[[.2,.3,.4,.5]]),self.field)]:
            result,info=preview_hippocampal_alignment(registration,self.img,prior,8,self.ids,None)
            self.assertIs(result,prior)
            self.assertFalse(info['applied'])

    def test_xy_convention(self):
        y,x=np.indices(self.img.shape,dtype='f')
        self.field[0]=.25*x
        self.field[1]=.5*y
        np.testing.assert_allclose(jacobian(self.field),1.25*1.5)

    def test_disconnected_distractor(self):
        t=np.arange(60)
        truth=25+np.round(6*np.sin(t/18)).astype(int)
        costs=-np.abs(np.arange(75)[None,:]-truth[:,None]).astype(float)
        costs[20:23,60]=10
        for predicted in [smooth_trace(costs),trace(costs,4,.3)]:
            self.assertLessEqual(np.abs(predicted-truth).max(),1)

if __name__=='__main__':
    unittest.main()
