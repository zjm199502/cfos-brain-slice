"""Use Qt logical screen dimensions, including the taskbar and display scaling."""
from qtpy.QtCore import QEvent, QSize, Qt, QTimer
from qtpy.QtGui import QPalette, QTextLayout, QTextOption
from qtpy.QtWidgets import (
    QApplication, QLabel, QListView, QScrollArea, QSizePolicy, QStyle, QStyleOptionViewItem, QVBoxLayout, QWidget,
)
from napari._qt.containers._layer_delegate import LayerDelegate


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


class LayerNameDelegate(LayerDelegate):
    """Allow napari's visibility box, layer icon and thumbnail to keep their own space."""

    def _lines(self, text, option):
        # The thumbnail, visibility indicator, type icon and item margins share the row.
        view = self.parent()
        width = max(20, view.viewport().width() - 112)
        if option.fontMetrics.horizontalAdvance(text) <= width:
            return [text]
        lines = []
        for paragraph in text.split("\n"):
            data = paragraph.encode("utf-16-le")
            layout = QTextLayout(paragraph, option.font)
            wrap = QTextOption()
            wrap.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
            layout.setTextOption(wrap)
            layout.beginLayout()
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                start = line.textStart()
                # Qt uses UTF-16 indices. Slice the encoded string for supplementary characters.
                lines.append(data[2*start:2*(start + line.textLength())].decode("utf-16-le"))
            layout.endLayout()
        return lines

    def paint(self, painter, option, index):
        if not index.isValid():
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        lines = self._lines(opt.text, opt)
        if len(lines) <= 1:
            return super().paint(painter, option, index)
        self.get_layer_icon(opt, index)
        view = self.parent()
        style = view.style()
        bounds = style.subElementRect(QStyle.SE_ItemViewItemText, opt, view)
        # Draw the native row, including its eye and type icon. Qt elides even
        # explicit newlines in this style, so draw the wrapped name separately.
        opt.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, view)
        self._paint_loading(painter, opt, index)
        self._paint_thumbnail(painter, opt, index)
        self._paint_lock_icon(painter, opt, index)
        painter.save()
        painter.setClipRect(option.rect)
        painter.setFont(opt.font)
        role = QPalette.HighlightedText if opt.state & QStyle.State_Selected else QPalette.Text
        painter.setPen(opt.palette.color(role))
        step = max(opt.fontMetrics.height(), opt.fontMetrics.lineSpacing())
        bounds.setTop(option.rect.top() + (option.rect.height() - step * len(lines)) // 2)
        bounds.setHeight(step)
        for text in lines:
            painter.drawText(bounds, Qt.AlignCenter | Qt.TextSingleLine, text)
            bounds.translate(0, step)
        painter.restore()

    def sizeHint(self, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        rows = len(self._lines(opt.text, opt))
        default_height = super().sizeHint(option, index).height()
        step = max(opt.fontMetrics.height(), opt.fontMetrics.lineSpacing())
        return QSize(1, max(default_height, rows * step + 16))


class LayerSidebar(QWidget):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("图层（点眼睛图标可显示/隐藏）")
        title.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(view, 1)
        view.setContextMenuPolicy(Qt.NoContextMenu)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setResizeMode(QListView.Adjust)
        view.setUniformItemSizes(False)
        view.setStyleSheet("QtLayerList { min-width: 0px; }")
        self.delegate = LayerNameDelegate(view)
        view.setItemDelegate(self.delegate)
        self.delegate.loading_frame_changed.connect(view.viewport().update)

    def sizeHint(self):
        fm = self.view.fontMetrics()
        scrollbar = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        return QSize(max(242, 112 + fm.horizontalAdvance("细胞-Far red") + scrollbar),
                     super().sizeHint().height())
