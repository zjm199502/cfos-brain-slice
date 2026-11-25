"""Exercise the installed registration panel on an atlas-derived synthetic image."""
import os
import time
from pathlib import Path
import numpy as np
import tifffile
from napari.qt import get_qapp
from scipy.ndimage import gaussian_filter, shift
from cfos.core import registration as R
from cfos.core.atlas_contours import registration_paths
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section
from cfos.gui.main_window import MainWindow

def check_gui(folder):
    app=get_qapp()
    w,h,um=510,360,8.
    reg=R.Registration(plane=[445.,270.,300.,-415.,1.,6.,-12.,12.,-296.],orientation="r0")
    reg.nonlinear=False
    lab,_=R.label_maps(reg,w,h,None)
    rgb=((lab.astype(np.int64)*1291)%701+80)*(lab>0)
    img=shift(gaussian_filter(rgb.astype(float),.9),(2.,2.),order=1,mode="nearest").astype("uint16")
    path=folder/"synthetic.ome.tif"
    tifffile.imwrite(path,img,ome=True,metadata={"axes":"YX","PhysicalSizeX":um,"PhysicalSizeY":um})
    sec=Section(1,(0,0,w,h),[],name="S01",registration=reg.to_dict())
    project=Project(folder)
    project.channel_names=["DAPI"];project.images=[ImageEntry(str(path),sections=[sec])];project.save()
    win=MainWindow()
    win._load_project(Project.load(folder));win.steps.setCurrentRow(5)
    win._reg_section_changed(win.reg_sec_combo.currentIndex())
    assert (win.reg_sliders['ap_shift'][0].minimum(),win.reg_sliders['ap_shift'][0].maximum())==(-400,400)
    curve=max(win._reg_paths,key=len)[:,::-1]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
    mid=arc[-1]*.37
    guide=np.column_stack([np.interp(np.linspace(mid-11,mid+11,5),arc,col) for col in curve.T])+2.
    win.reg_landmark_btn.click();win.reg_trace_channel.setCurrentIndex(0)
    class Event:
        type="mouse_press"
        button=1
    def click(xy):
        e=Event();e.position=tuple(xy[::-1])
        assert next(win._reg_drag(win.viewer,e),None) is None
    for xy in guide:click(xy)
    assert len(win._trace_clicks)==5 and np.allclose(win.viewer.layers['规范点'].size*um,10.)
    assert not win._reg_current()[1].control_points
    win._reg_undo_point();assert len(win._trace_clicks)==4
    click(guide[-1]);win.reg_trace_apply.click()
    saved=win._reg_current()[1].control_points
    assert saved and not win._trace_clicks,win.statusBar().currentMessage()
    end=time.monotonic()+.9
    while time.monotonic()<end:app.processEvents();time.sleep(.01)
    assert Project.load(folder).images[0].sections[0].registration['control_points']==saved
    win._load_project(Project.load(folder));win.steps.setCurrentRow(5)
    assert win._reg_current()[1].control_points==saved
    win._reg_undo_point()
    assert not Project.load(folder).images[0].sections[0].registration['control_points']
    win.close();app.processEvents()
    print("PASS five single clicks, physical point size, uptake, reopen, group undo, +/-10 mm",flush=True)
