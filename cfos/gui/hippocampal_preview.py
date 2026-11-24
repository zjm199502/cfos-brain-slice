"""Preview a structural-channel refinement before replacing the section warp."""
import os
import tempfile

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QInputDialog, QLabel,
                            QScrollArea, QVBoxLayout)

from ..core import registration as R
from ..core.regions import _thumb_geometry


def confirm_preview(parent, title, channel, before, after, ids):
    from scipy.ndimage import binary_erosion, gaussian_filter
    gray = gaussian_filter(channel.astype("f"), 1.3)
    lo, hi = np.percentile(gray, (4, 99.5))
    gray = np.uint8(255 * np.clip((gray-lo)/max(hi-lo, 1e-6), 0, 1)**.78)
    y, x = np.where(np.isin(after,[ids["CA3sp"], ids["DG-sg"]]))
    if not len(x):
        return False
    margin = max(20,int(np.ptp(x)*.22))
    crop = np.s_[max(0,y.min()-margin):min(gray.shape[0],y.max()+margin+1),
                 max(0,x.min()-margin):min(gray.shape[1],x.max()+margin+1)]
    images = []
    for labels in (before[crop], after[crop]):
        rgb = np.repeat(gray[crop][:,:,None], 3, axis=2)
        for keys, color in [([ids["CA3sp"]],(255,175,35)),(ids["cc_tree"],(65,232,255)),
                            ([ids["ec"]],(250,125,235))]:
            mask = np.isin(labels, keys)
            rgb[mask & ~binary_erosion(mask)] = color
        h,w = labels.shape
        images.append(QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()))
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("橙色 CA3 锥体层　青色 胼胝体　粉色 外囊"))
    choice = QComboBox(); choice.addItems(["原配准","精修预览"]); choice.setCurrentIndex(1)
    layout.addWidget(choice)
    view = QLabel()
    scroll = QScrollArea(); scroll.setWidget(view); scroll.setAlignment(Qt.AlignCenter)
    layout.addWidget(scroll, 1)
    def show(idx):
        view.setPixmap(images[idx]); view.adjustSize()
    choice.currentIndexChanged.connect(show); show(1)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("采用精修")
    buttons.button(QDialogButtonBox.Cancel).setText("保留原配准")
    buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(min(1100,parent.width()),min(700,parent.height()))
    return dialog.exec_() == QDialog.Accepted


def _write_warp(path, array):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as f:
        name = f.name
    try:
        np.save(name,array)
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def preview_section(window):
    from ..core.atlas import load_atlas
    from ..core.fosfolio_adapter import preview_hippocampal_alignment
    section, reg = window._reg_current()
    if section is None or not reg.done or not reg.nonlinear:
        window.info("请先为这张冠状脑片完成非线性配准，再预览海马精修。"); return
    if reg.control_points:
        window.info("这张脑片已有手动修正点，已保留当前配准。"); return
    project, image = window.project, window.image
    choices = [f"{i+1}  {name}" for i,name in enumerate(window.slide.channels)]
    if not choices: return
    name, ok = QInputDialog.getItem(window,"海马局部精修","选择清楚显示锥体层和白质边缘的结构通道",choices,0,False)
    if not ok: return
    channel = choices.index(name)
    path = R.warp_path(project.results_dir,image.key,section.name)
    if not path.exists():
        window.info("请先生成这张脑片的非线性形变场。"); return
    stamp = path.stat().st_mtime_ns
    original = reg.to_dict()
    def job(task):
        slide = image.open_slide(z_mode=project.z_mode)
        try:
            ds, shape, lv = _thumb_geometry(slide,section)
            frame = R.orient(slide.read(lv,section.bbox0,[channel])[0],reg.orientation)
            unit = slide.pixel_size_um * ds
        finally:
            slide.close()
        if task.is_cancelled(): raise InterruptedError
        prior = np.load(path)
        if prior.shape != (2,*shape): return None
        st = load_atlas().structures
        ids = dict(zip(st.acronym,st["index"].astype(int)))
        cc_id = str(int(st.loc[st.acronym=="cc","id"].iloc[0]))
        ids["cc_tree"] = st.loc[st.path.map(lambda v: cc_id in str(v).split("/")),"index"].astype(int).tolist()
        candidate, qc = preview_hippocampal_alignment(reg,frame,prior,unit,ids,R.label_maps)
        if task.is_cancelled(): raise InterruptedError
        if not qc["applied"]: return None
        h,w = shape
        return (frame,prior,candidate,ids,R.label_maps(reg,w,h,prior)[0],R.label_maps(reg,w,h,candidate)[0])
    def still_current():
        return (window.project is project and window.image is image and window.stack.currentIndex()==5
                and window._reg_current()[0] is section and R.Registration.from_dict(section.registration).to_dict()==original
                and path.exists() and path.stat().st_mtime_ns==stamp)
    def done(result):
        if not still_current(): return
        if result is None:
            window.info("当前通道或初始形变未通过双侧层带、分辨率或形变检查，已保留原配准。"); return
        frame,prior,candidate,ids,before,after = result
        if not confirm_preview(window,f"{section.name} 海马局部精修",frame,before,after,ids) or not still_current(): return
        try:
            _write_warp(path,candidate)
            reg.nl_method = "海马锥体层和外侧白质局部精修"
            reg.status = "已采用海马局部精修"
            section.registration = reg.to_dict()
            try:
                project.save()
            except Exception:
                section.registration = original
                _write_warp(path,prior)
                raise
        except OSError as exc:
            window.error(str(exc)); return
        window._reg_cache.clear()
        window._reg_redraw()
    window.run_task(job,done,label="正在沿锥体层和外侧白质精修…")
