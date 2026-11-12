"""让 Qt 自带的对话框（打开文件、是/否、取消等）显示中文。"""
from qtpy.QtCore import QLibraryInfo, QLocale, QTranslator


def install_chinese(app):
    QLocale.setDefault(QLocale(QLocale.Chinese, QLocale.China))
    try:
        tdir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    except AttributeError:
        tdir = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
    app._cfos_translators = []
    for name in ("qtbase_zh_CN", "qt_zh_CN"):
        tr = QTranslator()
        if tr.load(name, tdir):
            app.installTranslator(tr)
            app._cfos_translators.append(tr)
    return len(app._cfos_translators)
