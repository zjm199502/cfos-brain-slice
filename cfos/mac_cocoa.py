"""Select the Qt Cocoa accessibility backport before creating QApplication."""
import platform
from pathlib import Path

def configure_plugins():
    if platform.machine() != "arm64":
        return
    from PySide6.QtCore import QCoreApplication, QOperatingSystemVersion, qVersion
    if qVersion() != "6.12.0" or QOperatingSystemVersion.current() < QOperatingSystemVersion(QOperatingSystemVersion.MacOS, 14, 4):
        return
    root = Path(__file__).resolve().parent / "mac_plugins"
    QCoreApplication.addLibraryPath(str(root))
