"""Drawing geometry for the equivalent-area cell circles used in the viewer and figures."""
import numpy as np

from .manual import ADD_RADIUS_UM


def diameters(frame, pixel_um):
    if "area_um2" not in frame:
        return np.full(len(frame), 2 * ADD_RADIUS_UM / pixel_um)
    area = frame.area_um2.to_numpy(dtype=float)
    radius = np.sqrt(np.maximum(0, area) / np.pi)
    return 2 * np.where(np.isfinite(area) & (area > 0), radius, ADD_RADIUS_UM) / pixel_um


def draw_rings(painter, xy, radii, color, width_px):
    from qtpy.QtCore import QRectF, Qt
    from qtpy.QtGui import QPainterPath, QPen

    painter.setPen(QPen(color, width_px, Qt.DashLine, Qt.FlatCap))
    painter.setBrush(Qt.NoBrush)
    for start in range(0, len(xy), 2048):
        path = QPainterPath()
        for (x, y), r in zip(xy[start:start+2048], radii[start:start+2048]):
            path.addEllipse(QRectF(x-r, y-r, 2*r, 2*r))
        painter.drawPath(path)


def draw_figure_rings(image, groups, width_pt, dpi):
    from PIL import Image
    from qtpy.QtGui import QColor, QImage, QPainter

    # Use the same anti-aliased dashed pen as the live viewer, including
    # fractional widths. The exported plate's nominal resolution is 300 dpi.
    pixels = image.tobytes()
    width, height = image.size
    surface = QImage(pixels, width, height, 3 * width, QImage.Format_RGB888).copy()
    painter = QPainter(surface)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        for xy, radii, color in groups:
            draw_rings(painter, xy, radii, QColor(*color), width_pt * dpi / 72.)
    finally:
        painter.end()
    bits = surface.constBits()
    data = bits.asstring(surface.sizeInBytes()) if hasattr(bits, "asstring") else bytes(bits)
    return Image.frombytes("RGB", image.size, data, "raw", "RGB", surface.bytesPerLine())
