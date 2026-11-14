"""“帮助”菜单：关于、检查更新、安装补丁、恢复到安装版本。"""
from __future__ import annotations

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QAction, QDesktopServices
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QLabel, QMessageBox, QTextBrowser, \
    QVBoxLayout

from .. import APP_NAME
from ..core import updater

LINK_CSS = "a { color: #6cb6ff; }"
EMAIL = "zjm199502@163.com"

FEATURES = [
    ("① 项目与图片", "导入 KFBIO .qptiff / OME-TIFF 等荧光扫描图，支持单层和 z-stack（最大投影 / 景深合成）。"),
    ("② 脑片拆分", "自动找出一张玻片上的每张脑片，可手动调整、排除破损区域。"),
    ("③ 亮度 / 对比度", "逐通道调整显示，所有截图沿用同一设置。"),
    ("④ 细胞识别", "StarDist 深度学习逐通道识别阳性细胞（不需要显卡），可人工补点 / 删点、用自己的标注微调模型，"
                  "并用“识别准确率检查”给出查准率、查全率；厚切片可识别前减背景、去卷积。"),
    ("⑤ 共标统计", "按距离 / 轮廓判断多通道共标，可按强度分级。"),
    ("⑥ 脑图谱配准", "DeepSlice 自动定位 + 轮廓与 DAPI 纹理非线性贴合到 Allen CCFv3，可拖动脑区边界手动矫正。"),
    ("⑦ 分组与统计", "按动物、组别汇总，组间统计、热图、功能网络分析。"),
    ("⑧ 导出 Excel", "按脑片、按脑区（含各级上级脑区、左右半脑）导出细胞数、密度、各通道平均荧光强度和全部分析参数。"),
    ("⑨ 截图制图", "ROI 放大截图、整张脑片配准图（矢量脑区轮廓），可编辑比例尺，PNG / TIFF / SVG。"),
]


def contact_html() -> str:
    return (f"联系：<a href='mailto:{EMAIL}'>{EMAIL}</a><br>"
            f"源码：<a href='{updater.SOURCE_URL}'>{updater.SOURCE_URL}</a>")


def about_html() -> str:
    rows = "".join(f"<li><b>{a}</b>：{b}</li>" for a, b in FEATURES)
    import os
    logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fosmap_128.png")
    img = (f"<p><img src='{QUrl.fromLocalFile(logo).toString()}' width='96' height='96'></p>"
           if os.path.exists(logo) else "")
    return (f"{img}<h2>{APP_NAME} {updater.version_text()}</h2>"
            "<p>面向小鼠脑切片荧光图像的一体化分析软件：从扫描图到按脑区统计的 cFos 阳性细胞数，"
            "全部步骤在一个中文界面里完成，只用 CPU，离线运行。原名“cFos 脑片分析”。</p>"
            f"<ul>{rows}</ul>"
            f"<p>{contact_html()}</p>"
            "<p>本软件使用了 StarDist、DeepSlice、elastix、Allen 小鼠脑图谱（CCFv3）、napari 等公开的软件和数据，"
            "完整列表和来源见 GitHub 上的 THIRD_PARTY.md。</p>")


def notes_html(text: str) -> str:
    """更新公告（Markdown 的简单子集）→ HTML。"""
    import html
    out, in_list = [], False
    for line in (text or "").splitlines():
        t = line.rstrip()
        if t.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{html.escape(t[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if t.startswith("#"):
            out.append(f"<h3>{html.escape(t.lstrip('#').strip())}</h3>")
        elif t:
            out.append(f"<p>{html.escape(t)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


class UpdateMixin:
    def _build_help_menu(self):
        m = self.menuBar().addMenu("帮助")
        a = QAction(f"关于 {APP_NAME}…", self); a.triggered.connect(self.show_about); m.addAction(a)
        m.addSeparator()
        a = QAction("检查更新…", self); a.triggered.connect(lambda: self.check_updates(silent=False)); m.addAction(a)
        a = QAction("从文件安装补丁…", self); a.triggered.connect(self.install_patch_file); m.addAction(a)
        a = QAction("恢复到安装时的版本…", self); a.triggered.connect(self.revert_patch); m.addAction(a)
        self.auto_update_act = QAction("启动时自动检查更新", self, checkable=True)
        self.auto_update_act.setChecked(updater.auto_check_enabled())
        self.auto_update_act.toggled.connect(updater.set_auto_check)
        m.addAction(self.auto_update_act)
        a = QAction("安装微调组件（离线包）…", self); a.triggered.connect(self.install_component_file); m.addAction(a)
        a = QAction("在 GitHub 上查看更新公告和补丁", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(updater.DOWNLOAD_PAGE))); m.addAction(a)

    # ---------------------------------------------------------------- 关于
    def show_about(self):
        dlg = QDialog(self); dlg.setWindowTitle(f"关于 {APP_NAME}"); dlg.resize(620, 620)
        L = QVBoxLayout(dlg)
        tb = QTextBrowser(); tb.setOpenExternalLinks(True)
        tb.document().setDefaultStyleSheet(LINK_CSS); tb.setHtml(about_html())
        L.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok); bb.accepted.connect(dlg.accept); L.addWidget(bb)
        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

    # ---------------------------------------------------------------- 检查更新
    def check_updates(self, silent: bool = True):
        """silent=True：启动时在后台检查，只有新版本时才弹窗；连不上网不提示。"""
        from .worker import Task
        if getattr(self, "_upd_task", None) is not None and self._upd_task.isRunning():
            return
        t = Task(lambda task: updater.check())
        t.finished_ok.connect(lambda res: self._on_update_checked(res, silent))
        t.failed.connect(lambda msg: None if silent else self.error(
            f"检查更新失败：{msg}\n\n可以稍后再试，或到 GitHub 手动下载补丁：\n{updater.DOWNLOAD_PAGE}"))
        self._upd_task = t
        t.start()

    def _on_update_checked(self, res, silent):
        st, m = res["status"], res["manifest"]
        if st == "latest":
            if not silent:
                self.info(f"已经是最新版本（{updater.version_text()}）。")
            return
        if silent and updater.skipped_version() == m["version"]:
            return
        self._show_update_dialog(res, silent)

    def _show_update_dialog(self, res, silent):
        m = res["manifest"]
        dlg = QDialog(self); dlg.setWindowTitle("软件更新"); dlg.resize(600, 560)
        L = QVBoxLayout(dlg)
        full = res["status"] == "full"
        head = (f"<b>有新版本 v{m['version']}</b>（你现在用的是 {updater.version_text()}）。")
        if full:
            head += ("<br>这个版本更新了软件依赖的组件，<b>需要下载完整安装包</b>"
                     "（直接安装即可，项目和结果不受影响）。")
        else:
            size = m.get("patch", {}).get("size", 0)
            head += f"<br>补丁包约 {max(size, 1) / 1e6:.1f} MB，安装后重启软件即可生效，项目和结果不受影响。"
        lb = QLabel(head); lb.setWordWrap(True); lb.setTextFormat(Qt.RichText); L.addWidget(lb)
        tb = QTextBrowser(); tb.setOpenExternalLinks(True); tb.document().setDefaultStyleSheet(LINK_CSS)
        notes = m.get("notes", "")
        tb.setHtml("<h3>更新公告</h3>" + notes_html(notes) + ("" if EMAIL in notes else f"<p>{contact_html()}</p>"))
        L.addWidget(tb, 1)
        bb = QDialogButtonBox()
        go = bb.addButton("打开下载页面" if full else "现在更新", QDialogButtonBox.AcceptRole)
        later = bb.addButton("以后再说", QDialogButtonBox.RejectRole)
        skip = bb.addButton("跳过这个版本", QDialogButtonBox.DestructiveRole)
        man = None if full else bb.addButton("手动下载…", QDialogButtonBox.HelpRole)
        L.addWidget(bb)
        choice = {}
        for b, k in ((go, "go"), (later, "later"), (skip, "skip"), (man, "manual")):
            if b is not None:
                b.clicked.connect(lambda _=False, k=k: (choice.__setitem__("k", k), dlg.accept()))
        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        k = choice.get("k", "later")
        if k == "skip":
            updater.skip_version(m["version"])
        elif k == "manual" or (k == "go" and full):
            QDesktopServices.openUrl(QUrl(m.get("full_installer", updater.FULL_PAGE) if full else updater.DOWNLOAD_PAGE))
        elif k == "go":
            self._download_and_install(res)

    def _download_and_install(self, res):
        def job(task):
            z = updater.download_patch(res, progress=task.report)
            return updater.install_patch(z)
        self.run_task(job, self._patch_installed, label="正在下载补丁…")

    # ---------------------------------------------------------------- 手动安装 / 恢复
    def install_patch_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择补丁包（从 GitHub 下载的 .zip）", "", "补丁包 (*.zip)")
        if not f:
            return
        try:
            info = updater.read_patch_info(f)
        except Exception as e:  # noqa: BLE001
            from ..core.log import friendly
            self.error(f"无法读取补丁包：{friendly(e)}")
            return
        r = QMessageBox.question(self, "安装补丁",
                                 f"安装补丁 v{info['version']}？（你现在用的是 {updater.version_text()}）\n"
                                 "安装后重启软件生效，项目和结果不受影响。")
        if r != QMessageBox.Yes:
            return
        self.run_task(lambda task: updater.install_patch(f), self._patch_installed, label="正在安装补丁…")

    def install_component_file(self):
        """没联网的电脑：选拷过来的微调组件包（TensorFlow 等，训练模型时才需要）。"""
        from ..core import components as C
        if C.tensorflow_available():
            self.info("这台电脑已经可以训练模型（微调组件已安装），不需要再装。")
            return
        f, _ = QFileDialog.getOpenFileName(self, f"选择微调组件包（{C.platform_tag()}，从 GitHub 下载的 .zip）", "",
                                           "组件包 (*.zip)")
        if not f:
            return

        def done(info):
            self.info(f"微调组件已安装并自检通过（{info.get('version')}）：{info.get('selftest', '')}\n"
                      "现在可以在“④ 细胞识别 → 微调模型”里训练了。")
        self.run_task(lambda task: C.install_pack(f, progress=task.report), done, label="正在安装微调组件…")

    def _patch_installed(self, info):
        r = QMessageBox.question(self, "补丁已安装",
                                 f"补丁 v{info['version']} 已安装，重启软件后生效。\n\n"
                                 "现在重启吗？（正在做的设置都已自动保存）",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._restart()

    def revert_patch(self):
        p = updater.active_patch()
        if p is None:
            self.info(f"现在用的就是安装时的版本（v{updater.base_version()}），没有安装补丁。")
            return
        r = QMessageBox.question(self, "恢复到安装时的版本",
                                 f"停用补丁 v{p['version']}，恢复到安装时的版本 v{updater.base_version()}？\n"
                                 "重启软件后生效。以后可以重新检查更新再装回来。")
        if r == QMessageBox.Yes:
            updater.revert_to_base()
            self._patch_installed_revert()

    def _patch_installed_revert(self):
        r = QMessageBox.question(self, "已恢复", "已停用补丁，重启软件后生效。现在重启吗？",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._restart()

    def _restart(self):
        try:
            if self.project is not None:
                self.project.save()
        except Exception:  # noqa: BLE001
            pass
        updater.restart_app()
        from qtpy.QtWidgets import QApplication
        QApplication.instance().quit()
