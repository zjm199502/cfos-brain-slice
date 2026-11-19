"""Use Qt logical screen dimensions, including the taskbar and display scaling."""
from qtpy.QtCore import QEvent, QSize, Qt, QTimer
from qtpy.QtWidgets import QApplication, QLabel, QScrollArea


def fit_window(widget, width, height):
    parent = widget.parentWidget()
    screen = parent.screen() if parent else widget.screen()
    if screen is None:
        screen = QApplication.primaryScreen()
    area = screen.availableGeometry()
    widget.resize(QSize(width, height).boundedTo(
        QSize(max(1, area.width()-32), max(1, area.height()-72))))


class PanelScroll(QScrollArea):
    """Reserve each visible paragraph's wrapped height, including dynamic form rows."""
    def __init__(self, widget):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setWidget(widget)
        self.reflow = QTimer(self)
        self.reflow.setSingleShot(True)
        self.reflow.timeout.connect(self._fit_labels)

    def eventFilter(self, obj, event):
        if (event.type() in (QEvent.LayoutRequest, QEvent.Resize, QEvent.Show) and
                obj is self.widget() and hasattr(self, "reflow") and self.isVisible()):
            self.reflow.start(0)
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        super().showEvent(event)
        self.reflow.start(0)

    def _fit_labels(self):
        for label in self.widget().findChildren(QLabel):
            if label.wordWrap() and label.isVisible() and label.width() > 0:
                height = max(0, label.heightForWidth(label.width()))
                if label.minimumHeight() != height:
                    label.setMinimumHeight(height)


class ProjectLabel(QLabel):
    def set_project(self, text):
        self.setToolTip(text)
        self._fit_text()

    def _fit_text(self):
        self.setText(self.fontMetrics().elidedText(self.toolTip(), Qt.ElideMiddle, self.contentsRect().width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_text()
