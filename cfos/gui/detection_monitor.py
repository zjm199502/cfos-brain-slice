"""Live per-channel inference progress and the scanner-coordinate tile overlay."""
from datetime import datetime
from time import monotonic

import numpy as np
from qtpy.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPen, QPolygonF, QPainterPath, QTransform
from qtpy.QtWidgets import (QGroupBox, QLabel, QPlainTextEdit,
                            QProgressBar, QVBoxLayout, QWidget)
from ..core.log import log


class TileOverlay(QWidget):
    clicked = Signal(float, float)
    resized = Signal()

    def __init__(self, canvas, viewer):
        super().__init__(canvas.native)
        self.canvas = canvas
        self.selected = []
        self.segments = np.empty((0, 4), dtype=float)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        canvas.native.installEventFilter(self)
        viewer.camera.events.connect(self._changed)
        self.setGeometry(canvas.native.rect())

    def _changed(self, event=None):
        self.update()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
            self.resized.emit()
        return False

    def set_picking(self, enabled):
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not enabled)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            transform = self.canvas.view.transform * self.canvas.view.scene.transform
            x,y,_,w = transform.imap((pos.x(),pos.y(),0,1))
            self.clicked.emit(x/w,y/w)
            event.accept()

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments
        self._grid_path = QPainterPath()
        for x, y, u, v in segments:
            self._grid_path.moveTo(x, y)
            self._grid_path.lineTo(u, v)

    def paintEvent(self, event):
        if not len(self.segments):
            return
        transform = self.canvas.view.transform * self.canvas.view.scene.transform
        basis = transform.map([[0,0,0,1],[1,0,0,1],[0,1,0,1]])
        origin, xx, yy = basis[:, :2] / basis[:, 3:4]
        painter_transform = QTransform(*(xx-origin), *(yy-origin), *origin)
        p = QPainter(self)
        pen = QPen(QColor(88, 88, 88, 165), 1.4, Qt.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setTransform(painter_transform)
        p.drawPath(self._grid_path)
        p.resetTransform()
        p.setPen(QPen(QColor(255, 190, 55), 2))
        p.setBrush(QColor(255, 190, 55, 40))
        for l,t,r,b in self.selected:
            coords = transform.map([[l,t,0,1],[r,t,0,1],[r,b,0,1],[l,b,0,1]])
            p.drawPolygon(QPolygonF([QPointF(x/w,y/w) for x,y,_,w in coords]))
        p.end()


def clock_text(seconds, precise=False):
    tenths = round(max(0., seconds) * 10) if precise else int(max(0., seconds)) * 10
    minutes, rest = divmod(tenths, 600)
    sec, sub = divmod(rest, 10)
    return f"{minutes:02d}:{sec:02d}" + (f".{sub}" if precise else "")


class DetectionMonitor(QGroupBox):
    def __init__(self):
        super().__init__('任务进度与日志')
        layout = QVBoxLayout(self)
        self.section = QLabel('等待任务')
        self.section.setWordWrap(True)
        layout.addWidget(self.section)
        self.form = QVBoxLayout()
        layout.addLayout(self.form)
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setMaximumBlockCount(2500)
        self.messages.setMinimumHeight(110)
        layout.addWidget(self.messages)
        self.rows = {}
        self.rates = {}
        self.task_started = None
        self.total_time = QLabel()
        self.total_time.setWordWrap(True)
        layout.insertWidget(1, self.total_time)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)

        self.overall = QProgressBar()
        self.overall.hide()
        self.overall.setAccessibleName("当前阶段进度")
        layout.insertWidget(1, self.overall)
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._elapsed)
        self.task_name = ""
        self.last_message = ""

    def _elapsed(self):
        if self.task_started is None:
            return
        now = monotonic()
        eta = "估算中"
        if self.done > 0 and self.total > self.done:
            eta = clock_text((now - self.phase_started) * (self.total - self.done) / self.done)
        self.total_time.setText(f"{self.task_name}　已用 {clock_text(now-self.task_started)}　当前阶段预计剩余 {eta}")

    def report(self, done, total, text):
        if self.task_started is None:
            return
        if total != self.total or done < self.done:
            self.phase_started = monotonic()
        self.done, self.total = done, total
        self.overall.setRange(0, total if total > 1 or done else 0)
        self.overall.setValue(done)
        if text and text != self.last_message:
            self.last_message = text
            self.note(text)
            if not self.rows:
                self.section.setText(text)
        self._elapsed()

    def note(self, text):
        self.messages.appendPlainText(f'{datetime.now():%H:%M:%S}  {text}')
        log.info('%s', text)

    def begin_task(self, label="细胞识别"):
        self.task_started = monotonic()
        self.phase_started = self.task_started
        self.done = self.total = 0
        self.task_name = self.last_message = label
        self.reset()
        self.section.setText(label)
        self.overall.setRange(0, 0)
        self.overall.show()
        self.note(label)
        self.elapsed_timer.start()
        self._elapsed()

    def finish_task(self, message=""):
        self.end(message)
        if self.task_started is None:
            return
        elapsed = clock_text(monotonic() - self.task_started, precise=True)
        self.elapsed_timer.stop()
        self.task_started = None
        if message:
            self.overall.hide()
        else:
            self.overall.setRange(0, 1)
            self.overall.setValue(1)
        summary = f"{self.task_name}　实际用时 {elapsed}"
        self.total_time.setText(summary)
        self.note(summary)
        if not self.rows:
            self.section.setText(message if message else "任务已完成")

    def reset(self):
        self.timer.stop()
        self.rows.clear()
        while self.form.count():
            item = self.form.takeAt(0)
            item.widget().hide()
            item.widget().deleteLater()
        self.section.setText('正在准备识别…')

    def end(self, reason=''):
        if self.timer.isActive():
            self.timer.stop()
            now = monotonic()
            for row in self.rows.values():
                if row['begin'] is not None:
                    row['seconds'] += now - row['begin']
                    row['begin'] = None
                if row['done'] < row['total']:
                    row['stopped'] = True
            self.refresh()
        if reason:
            self.note(reason)

    def event_data(self, ev):
        kind = ev['kind']
        if kind == 'section':
            self.reset()
            self.section.setText(ev['title'])
            self.note(ev['title'] + '  ' + ev['grid'])
            for channel, name in ev['channels'].items():
                panel = QWidget()
                layout = QVBoxLayout(panel)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(2)
                title = QLabel(name)
                title.setWordWrap(True)
                bar = QProgressBar()
                bar.setAccessibleName(name)
                bar.setTextVisible(False)
                bar.setRange(0, 1000)
                label = QLabel()
                label.setWordWrap(True)
                layout.addWidget(title)
                layout.addWidget(bar)
                layout.addWidget(label)
                self.form.addWidget(panel)
                self.rows[channel] = dict(bar=bar, label=label, name=name, done=0,
                                          total=ev.get('channel_totals', {}).get(channel, ev['total']), begin=None, seconds=0.,
                                          samples=0, batch=0, batches=0, stopped=False)
            self.timer.start()
        elif kind == 'note':
            self.note(ev['text'])
        elif kind == 'end':
            self.end(ev['text'])
        elif kind in ('start', 'done', 'skip', 'batch'):
            row = self.rows[ev['channel']]
            if kind == 'batch':
                row['batch'], row['batches'] = ev['done'], ev['total']
            elif kind == 'start':
                row['begin'] = ev['started']
                row['pixels'] = ev['pixels']
                row['batch'] = row['batches'] = 0
            else:
                row['seconds'] += ev.get('duration', 0.)
                if kind == 'done':
                    row['samples'] += 1
                    self.rates[row['name']] = ev['duration'] / max(1, row['pixels'])
                row['done'] += 1
                row['begin'] = None
                row['batch'] = row['batches'] = 0
            if kind != 'batch':
                self.note(f"{self.section.text()} · {row['name']} · {ev['text']}")
                if kind in ("done", "skip") and row["done"] == row["total"]:
                    self.note(f"{self.section.text()} · {row['name']} · {row['total']} 块完成，"
                              f"通道实际识别用时 {clock_text(row['seconds'], precise=True)}")
        self.refresh()

    def refresh(self):
        now = monotonic()
        for row in self.rows.values():
            done, total = row['done'], row['total']
            active = row['begin'] is not None
            current = max(0., now - row['begin']) if active else 0.
            spent = row['seconds'] + current
            batch, batches = row['batch'], row['batches']
            # The final part of a model tile is reserved for decoding its masks.
            fraction = batch / (batches + 1) if batches else 0.
            value = int(1000 * (done + fraction) / total) if total else 1000
            row['bar'].setValue(value)
            remaining = total - done
            if not remaining:
                state, eta = '完成', '00:00'
            elif row['stopped']:
                state, eta = '已停止', '—'
            else:
                state = '等待下一块' if done else '等待'
                if active:
                    pulse = ('·', '··', '···')[int(current * 2) % 3]
                    state = f"识别第 {done + 1} 块{pulse}"
                    if batches:
                        state += f" 网络批次 {batch}/{batches}" if batch < batches else " 生成轮廓"
                estimate = None
                if row['samples']:
                    estimate = row['seconds'] * remaining / row['samples'] - current
                elif active and batch:
                    estimate = current * (batches + 1) * remaining / batch - current
                elif active and row['name'] in self.rates:
                    estimate = self.rates[row['name']] * row['pixels'] * remaining - current
                eta = '估算中' if estimate is None or estimate <= 0 else clock_text(max(1, estimate))
            pct = value // 10
            count = f"{done}/{total} 块（{pct}%）"
            if not remaining or row['stopped']:
                caption = f"{state}　{count}　实际识别用时 {clock_text(spent, precise=True)}"
            else:
                caption = f"{state}　{count}　已用 {clock_text(spent)}　预计剩余 {eta}"
            if caption != row['label'].text():
                row['label'].setText(caption)
                row['bar'].setAccessibleDescription(caption)
