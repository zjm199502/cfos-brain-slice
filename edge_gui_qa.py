"""Exercise the actual napari drag callback and save/reopen on the active release."""
import tempfile
import time
from pathlib import Path
import numpy as np
from napari.qt import get_qapp
from cfos.gui.main_window import MainWindow
from cfos.core.project import ImageEntry, Project
from cfos.core.sections import Section
from cfos.core.registration import Registration
from cfos.core.edge_drag import snap_path


def check_gui(folder, existing=False):
    app=get_qapp()
    window=MainWindow()
    if not existing:
        import tifffile
        image=folder/"synthetic.ome.tif"
        tifffile.imwrite(image,np.full((360,510),500,np.uint16),ome=True,
                         metadata={"axes":"YX","PhysicalSizeX":8.,"PhysicalSizeY":8.})
        project=Project(folder)
        reg=Registration(plane=[445.,270.,300.,-415.,1.,6.,-12.,12.,-296.],orientation="r180")
        sec=Section(1,(0,0,510,360),[],name="S01",registration=reg.to_dict())
        project.images=[ImageEntry(str(image),sections=[sec])]
        project.channel_names=["DAPI"]
        project.save()
    errors=[]
    window.error=errors.append
    window.info=lambda message: print("UI:",message,flush=True)
    window._load_project(Project.load(folder))
    window.steps.setCurrentRow(5)
    window._reg_section_changed(window.reg_sec_combo.currentIndex())
    s,reg=window._reg_current()
    assert reg.done and not reg.control_points and window._reg_paths
    pp=window.slide.pixel_size_um
    _,ds,lab,_=window._reg_lab
    # Choose a long, isolated curved segment in the central tissue.
    path=max(window._reg_paths,key=len)
    j=len(path)//3
    xy=(path[j-1]+path[j])*.5
    start=np.array([s.bbox0[0]+xy[1]*ds,s.bbox0[1]+xy[0]*ds],float)
    tangent=(path[j]-path[j-1])[::-1]
    normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
    delta=normal*(10./pp)
    window.reg_radius.setValue(10)
    assert window.reg_radius.minimum()==10 and window.reg_radius.value()==10
    window.reg_drag_btn.setChecked(True)
    class Event:
        type="mouse_press"
        button=1
    def drag(distance):
        e=Event()
        e.position=tuple(start[::-1])
        cb=window._reg_drag(window.viewer,e)
        next(cb)
        e.position=tuple((start+distance)[::-1])
        e.type="mouse_move"
        next(cb)
        e.type="mouse_release"
        try: next(cb)
        except StopIteration: pass
    drag(np.zeros(2))
    assert not window._reg_current()[1].control_points
    drag(delta)
    cp=window._reg_current()[1].control_points[-1]
    W,H=lab.shape[1],lab.shape[0]
    dist=np.linalg.norm((np.array(cp[2:4])-cp[:2])*[W,H])*ds*pp
    radius=cp[4]*W*ds*pp
    assert abs(dist-10)<.005 and abs(radius-10)<.005,(dist,radius)
    ex,ey=(start+delta-s.bbox0[:2])/ds
    snap=snap_path(window._reg_paths,ex,ey,.5/(ds*pp))
    assert snap is not None,"edited vector missed its 10 µm destination"
    limit=time.monotonic()+.65
    while time.monotonic()<limit: app.processEvents(); time.sleep(.02)
    assert Project.load(folder).images[0].sections[0].registration['control_points'][-1]==cp
    window._load_project(Project.load(folder)); window.steps.setCurrentRow(5)
    window._reg_section_changed(window.reg_sec_combo.currentIndex())
    assert window._reg_current()[1].control_points[-1]==cp
    window._reg_undo_point()
    assert not Project.load(folder).images[0].sections[0].registration['control_points']
    assert not errors,errors
    window.close()
    app.processEvents()
    print(f"PASS GUI drag / snapped curve / reopen / undo, radius {radius:.4f} µm, distance {dist:.4f} µm",flush=True)

