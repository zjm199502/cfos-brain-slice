"""Screen-space mouse handles for navigation, atlas corrections and polygon groups."""
import numpy as np
from napari.layers import Points, Shapes
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from qtpy.QtWidgets import QAbstractItemView, QApplication, QLineEdit, QWidget
from vispy.scene.cameras import PanZoomCamera

from ..core.group_transform import make_edit, similarity


def rectangle(a,b):
    (x,y),(r,t) = np.minimum(a,b),np.maximum(a,b)
    return np.array([[x,y],[r,y],[r,t],[x,t]])


def affine(source, target):
    # Three corresponding points completely determine this 2D affine map.
    m = np.linalg.solve(np.column_stack((source[:3],np.ones(3))),target[:3])
    return QTransform(*m.ravel())


def poly(points):
    return QPolygonF([QPointF(*p) for p in points])


def pan_rotated_view(event):
    """Pan a rotated 2D view using the same transform that draws the slide."""
    view = event.source
    camera = view.camera
    if (event.handled or not isinstance(camera, PanZoomCamera)
            or camera.pre_transform is None or not camera.interactive or not camera.mouse_pan
            or 1 not in event.buttons or event.modifiers or event.press_event is None):
        return
    previous = event.last_event
    if previous is None:
        return
    # Both event positions are local to the viewbox. Its scene transform
    # includes the preview rotation, current pan and zoom, and axis flips.
    before, after = view.scene.transform.imap([previous.pos, event.pos])
    camera.pan((before - after)[:2])
    event.handled = True


def clipped_segments(points, frame):
    """Liang–Barsky clipping in the local coordinates of a rotated marquee."""
    if len(points)<2:
        return None
    basis = np.stack((frame[1]-frame[0],frame[3]-frame[0]))
    uv = (points-frame[0]) @ np.linalg.inv(basis)
    p, d = uv[:-1], np.diff(uv,axis=0)
    low, high = np.zeros(len(d)),np.ones(len(d))
    for a,v in zip(p.T,d.T):
        flat = abs(v)<1e-12
        u = np.divide(-a,v,out=np.zeros_like(a),where=~flat)
        t = np.divide(1-a,v,out=np.ones_like(a),where=~flat)
        low = np.maximum(low,np.where(flat,0,np.minimum(u,t)))
        high = np.minimum(high,np.where(flat,1,np.maximum(u,t)))
        high[flat & ((a<0)|(a>1))]=-1
    good = high>low
    if not np.any(good):
        return None
    delta = np.diff(points,axis=0)[good]
    return points[:-1][good]+low[good,None]*delta,points[:-1][good]+high[good,None]*delta


class CanvasTransform(QWidget):
    def __init__(self, window):
        self.w = window
        self.canvas = window.qt_viewer.canvas
        super().__init__(self.canvas.native)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setGeometry(self.canvas.native.rect())
        self.mode = self.box = self.drag = self.marquee = None
        self.committing = False
        self.armed = False
        self.items = []
        self.lines = QPainterPath()
        self.native_edit = None
        self.hide()
        QApplication.instance().installEventFilter(self)
        self.canvas.view.events.mouse_move.connect(pan_rotated_view, position='first')
        window.viewer.camera.events.connect(self._repaint)

    def _repaint(self, *_):
        if self.isVisible():
            self.update()

    def map(self, xy, inverse=False):
        transform = self.canvas.view.transform * self.canvas.view.scene.transform
        xy = np.atleast_2d(xy)
        hom = np.column_stack((xy,np.zeros(len(xy)),np.ones(len(xy))))
        out = (transform.imap if inverse else transform.map)(hom)
        return out[:,:2]/out[:,3,None]

    def reset(self, restore=True):
        if restore and self.drag and self.mode=='view':
            self.w.view_angle.setValue(self.drag['view_angle'])
            self.w.viewer.camera.center=self.drag['view_center']
        self.mode=self.box=self.drag=self.marquee=None
        self.armed=False
        self.lines=QPainterPath()
        self.items=[]
        self.hide()
        if hasattr(self.w,'view_move'):
            self.w.view_move.setChecked(False)
        if hasattr(self.w,'atlas_move'):
            self.w.atlas_move.setChecked(False)

    def _screen_frame(self):
        return rectangle((self.width()*.22,self.height()*.23),(self.width()*.78,self.height()*.77))

    def begin(self, mode):
        self.escape_tools()
        if self.w.slide is None:
            return
        if mode=='atlas' and not self._atlas_ready():
            self.w.statusBar().showMessage('请先选择已配准的脑片并显示脑区轮廓。',5000)
            return
        self.mode=mode
        self.whole=mode=='atlas'
        self.box=self.map(self._screen_frame(),True)
        if mode=='atlas':
            section,ds,lab,_=self.w._reg_lab
            x,y=section.bbox0[:2]
            h,b=lab.shape
            corners=rectangle((x,y),(x+ds*b,y+ds*h))
            projected=self.map(corners)
            if np.all((projected>=[12,45]) & (projected<=[self.width()-12,self.height()-12])):
                self.box=corners
            self._atlas_lines()
        (self.w.view_move if mode=='view' else self.w.atlas_move).setChecked(True)
        self._show()

    def _show(self):
        self.setGeometry(self.canvas.native.rect())
        self.show()
        self.raise_()
        self.update()

    def _atlas_ready(self):
        w=self.w
        return (w.stack.currentIndex()==5 and w._reg_lab is not None and '脑区轮廓' in w.viewer.layers
                and w.viewer.layers['脑区轮廓'].visible and bool(w._reg_paths))

    def _atlas_lines(self):
        sec,ds,_,_=self.w._reg_lab
        self.lines=QPainterPath()
        for path in self.w._reg_paths:
            xy=path[:,::-1]*ds+sec.bbox0[:2]
            if self.whole:
                self.lines.moveTo(*xy[0])
                for p in xy[1:]:
                    self.lines.lineTo(*p)
            else:
                segments=clipped_segments(xy,self.box)
                if segments is not None:
                    for a,b in zip(*segments):
                        self.lines.moveTo(*a)
                        self.lines.lineTo(*b)

    def _select(self, a, b):
        if min(abs(b-a))<4:
            self.reset()
            return
        self.box=self.map(rectangle(a,b),True)
        w=self.w
        layer=w.viewer.layers.selection.active
        names=['自定义区域','排除区域']
        if w.stack.currentIndex()==5 and self._atlas_ready() and getattr(layer,'name','') not in names:
            self.mode='atlas'
            self.whole=False
            self._atlas_lines()
        else:
            if w.stack.currentIndex()==1 and getattr(layer,'name','')=='脑片':
                names=['脑片']
            self.mode='shapes'
            lo,hi=np.minimum(a,b),np.maximum(a,b)
            self.items=[]
            self.lines=QPainterPath()
            for name in names:
                if name not in w.viewer.layers or not w.viewer.layers[name].visible:
                    continue
                target=w.viewer.layers[name]
                for i,data in enumerate(target.data):
                    xy=np.asarray(data)[:,::-1]
                    screen=self.map(xy)
                    if len(xy)<3 or not ((screen>=lo)&(screen<=hi)).all():
                        continue
                    self.items.append((target,i,xy.copy()))
                    self.lines.addPolygon(poly(xy))
                    self.lines.closeSubpath()
        if self.lines.isEmpty():
            self.reset()
            w.statusBar().showMessage('框内没有可变换的轮廓。自定义区域和排除区域须完整框入。',6000)
            return
        self._show()

    def _frame(self):
        if self.drag and self.mode!='view':
            return self.map(similarity(self.box,self.box.mean(axis=0),*self.drag['delta']))
        return self.map(self.box)

    def _handle(self, frame):
        mid=(frame[0]+frame[1])/2
        direction=mid-frame.mean(axis=0)
        return mid+direction/max(1.,np.linalg.norm(direction))*24

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen=QPen(QColor('#54ede4'))
        pen.setWidthF(self.logicalDpiX()/72.)
        painter.setPen(pen)
        if self.marquee is not None:
            painter.drawPolygon(poly(rectangle(*self.marquee)))
        elif self.box is not None:
            frame=self._frame()
            if not self.lines.isEmpty():
                painter.drawPath(affine(self.box,frame).map(self.lines))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolygon(poly(frame))
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor('#17393d'))
            knob=self._handle(frame)
            painter.drawLine(QPointF(*((frame[0]+frame[1])/2)),QPointF(*knob))
            painter.drawEllipse(QPointF(*knob),6,6)
            if self.mode!='view':
                for x,y in frame:
                    painter.drawRect(int(x)-4,int(y)-4,8,8)
        painter.end()

    def _press(self,pos):
        frame=self._frame()
        if np.linalg.norm(pos-self._handle(frame))<=12:
            action='rotate'
        elif self.mode!='view' and np.linalg.norm(frame-pos,axis=1).min()<=12:
            action='scale'
        elif poly(frame).containsPoint(QPointF(*pos),Qt.OddEvenFill):
            action='move'
        else:
            self.reset()
            return False
        self.drag=dict(action=action,origin=self.map(pos,True)[0],start=pos,
                       center=self.map(self.box.mean(axis=0))[0],view_angle=self.w.view_angle.value(),
                       view_center=self.w.viewer.camera.center,delta=(np.zeros(2),0.,1.))
        return True

    def _move(self,pos):
        drag=self.drag
        if not drag:
            return
        origin,center=drag['origin'],self.box.mean(axis=0)
        dest=self.map(pos,True)[0]
        delta=np.zeros(2)
        angle,scale=0.,1.
        if self.mode=='view':
            if drag['action']=='rotate':
                a,b=drag['start']-drag['center'],pos-drag['center']
                degrees=np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b))
                self.w.view_angle.setValue((drag['view_angle']+degrees+180)%360-180)
                # Keep the pivot fixed on screen even after a previous pan.
                origin,dest=center,self.map(drag['center'],True)[0]
            camera=self.w.viewer.camera
            camera.center=np.asarray(camera.center)+np.r_[0.,(origin-dest)[::-1]]
        else:
            if drag['action']=='move':
                delta=dest-origin
            elif drag['action']=='rotate':
                a,b=origin-center,dest-center
                angle=np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)
            else:
                scale=max(.02,np.linalg.norm(dest-center)/max(1e-12,np.linalg.norm(origin-center)))
            drag['delta']=(delta,angle,scale)
        self.update()

    def _release(self):
        drag,self.drag=self.drag,None
        if not drag or self.mode=='view':
            return
        delta,angle,scale=drag['delta']
        if np.allclose(delta,0) and abs(angle)<1e-8 and abs(scale-1)<1e-8:
            return
        w=self.w
        center=self.box.mean(axis=0)
        new=similarity(self.box,center,delta,angle,scale)
        if self.mode=='atlas' and self._atlas_ready():
            sec,ds,lab,_=w._reg_lab
            s,reg=w._reg_current()
            if s is not sec or not reg.done:
                self.reset();return
            H,W=lab.shape
            box=(self.box-sec.bbox0[:2])/ds
            move=delta/ds
            if reg.orientation=='r180':
                box=[W,H]-box
                move=-move
            point=make_edit(box,move,angle,scale,W,H,self.whole,reg.control_points)
            reg.control_points.append(point)
            reg.status='已手动调整'
            w._reg_reviewed(reg)
            old=sec.registration
            sec.registration=reg.to_dict()
            try:
                w.project.save()
            except OSError as exc:
                sec.registration=old
                self.reset()
                w.error(str(exc))
                return
            self.committing=True
            try:
                w._reg_redraw()
            finally:
                self.committing=False
            self.box=new
            self._atlas_lines()
            w.statusBar().showMessage('轮廓变换已保存，可点“撤销上一个修正”。',5000)
        elif self.mode=='shapes':
            changed={}
            items=[]
            for layer,i,xy in self.items:
                if layer not in w.viewer.layers or i>=len(layer.data) or not np.array_equal(layer.data[i],xy[:,::-1]):
                    self.reset();return
                coords=similarity(xy,center,delta,angle,scale)
                changed.setdefault(layer,list(layer.data))[i]=coords[:,::-1]
                items.append((layer,i,coords))
            for layer,data in changed.items():
                layer.data=data
            self.items=[(l,i,l.data[i][:,::-1].copy()) for l,i,_ in items]
            self.lines=affine(self.box,new).map(self.lines)
            self.box=new
            w.statusBar().showMessage('已调整选中轮廓，可用本页的保存按钮保留修改。',5000)
        self.update()

    def escape_tools(self):
        w=self.w
        engaged=self.mode is not None or self.armed or self.marquee is not None
        self.reset()
        if w.pick_tiles.isChecked():
            w.pick_tiles.setChecked(False)
            engaged=True
        if w._reg_drag_on or w._drag or w._trace_clicks:
            w._reg_set_drag_mode(False)
            engaged=True
        for layer in list(w.viewer.layers):
            if not isinstance(layer,(Shapes,Points)) or (layer.mode=='pan_zoom' and not layer.selected_data):
                continue
            engaged=True
            for callback,gen in list(layer._mouse_drag_gen.items()):
                gen.close()
                layer._persisted_mouse_event.pop(gen,None)
                layer._mouse_drag_gen.pop(callback,None)
            if isinstance(layer,Shapes) and layer._is_creating:
                index=layer._moving_value[0]
                if index is not None:
                    layer.selected_data={index}
                    layer.remove_selected()
            if self.native_edit and self.native_edit[0] is layer:
                layer.data=self.native_edit[1]
            layer.mode='pan_zoom'
            layer.selected_data=set()
            layer.mouse_pan=True
        self.native_edit=None
        active=w.viewer.layers.selection.active
        if hasattr(w, 'section_edit_mode'):
            w.section_edit_mode.setCurrentIndex(0)
        if active is not None:
            if hasattr(active,"mode"):
                active.mode='pan_zoom'
            active.mouse_pan=True
        w.viewer.camera.mouse_pan=True
        return engaged

    def eventFilter(self,obj,event):
        w=self.w
        typ=event.type()
        if typ==QEvent.FocusIn and isinstance(obj,QLineEdit) and obj.window() is w:
            obj.setProperty('fosfolioEditStart',obj.text())
        if typ in (QEvent.KeyPress,QEvent.ShortcutOverride) and event.key()==Qt.Key_Escape:
            if (not isinstance(obj,QWidget) or obj.window() is not w or
                    QApplication.activeModalWidget() or QApplication.activePopupWidget()):
                return False
            focus=QApplication.focusWidget()
            ancestor=focus
            while ancestor is not None and ancestor is not w:
                if isinstance(ancestor,QAbstractItemView) and ancestor.state()==QAbstractItemView.EditingState:
                    return False  # The native table delegate discards its pending edit.
                ancestor=ancestor.parentWidget()
            if typ==QEvent.KeyPress and not event.isAutoRepeat():
                if isinstance(focus,QLineEdit) and focus.isModified():
                    initial=focus.property('fosfolioEditStart')
                    if initial is not None:
                        focus.setText(initial)
                        focus.setModified(False)
                        self.escape_tools()
                        self.canvas.native.setFocus()
                        event.accept()
                        return True
                engaged=self.escape_tools()
                if not engaged and w.task is not None and w.task.isRunning():
                    w.task.cancel()
                    w.statusBar().showMessage('正在请求取消当前任务…')
                else:
                    w.statusBar().showMessage('已返回拖动看图。',3000)
            event.accept();return True
        if obj is self.canvas.native and typ==QEvent.Resize:
            self.setGeometry(self.canvas.native.rect())
        if obj is not self.canvas.native and obj is not w.tile_overlay:
            return False
        if not w.slide or typ not in (QEvent.MouseButtonPress,QEvent.MouseMove,QEvent.MouseButtonRelease,QEvent.ContextMenu):
            return False
        if typ==QEvent.ContextMenu:
            return self.armed or self.mode is not None or self.marquee is not None
        local=event.position() if hasattr(event,"position") else event.localPos()
        pos=np.array([local.x(),local.y()])
        if typ==QEvent.MouseButtonPress:
            if event.button()==Qt.RightButton and not (w.task and w.task.isRunning()):
                self.escape_tools()
                self.marquee=(pos,pos)
                self._show()
                return True
            if event.button()==Qt.LeftButton:
                if self.armed:
                    self.armed=False
                    self.marquee=(pos,pos)
                    return True
                if self.box is not None:
                    return self._press(pos)
                layer=w.viewer.layers.selection.active
                if isinstance(layer,(Shapes,Points)) and layer.mode in ('direct','select'):
                    self.native_edit=(layer, [p.copy() for p in layer.data] if isinstance(layer,Shapes) else layer.data.copy())
        elif typ==QEvent.MouseMove:
            if self.marquee is not None:
                self.marquee=(self.marquee[0],pos)
                self.update();return True
            if self.drag:
                self._move(pos);return True
        elif typ==QEvent.MouseButtonRelease:
            self.native_edit=None
            if self.marquee is not None:
                a,_=self.marquee
                self.marquee=None
                if event.button()==Qt.RightButton and np.linalg.norm(pos-a)<4:
                    self.armed=True
                    w.statusBar().showMessage('拖出选框，Esc 退出。',5000)
                else:
                    self._select(a,pos)
                return True
            if self.drag:
                self._move(pos)
                self._release()
                return True
        return False
