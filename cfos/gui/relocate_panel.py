"""项目换电脑后重新定位原图的界面部分。"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QPushButton


class RelocateMixin:
    def _relocate_widgets(self, layout):
        row = QHBoxLayout()
        b1 = QPushButton("重新定位找不到的图片…"); b1.clicked.connect(self.relocate_images)
        b1.setToolTip("项目拷到另一台电脑、或原图换了位置后使用：选择图片现在所在的文件夹（可以是整个硬盘），\n"
                      "软件会按文件名在里面（含子文件夹）搜索，找到后自动改好路径。已有的识别、配准结果都保留。")
        b2 = QPushButton("替换选中图片的文件…"); b2.clicked.connect(self.replace_image_file)
        b2.setToolTip("手动为表格中选中的图片指定新文件（例如文件改过名）。新文件必须是同一次扫描。")
        row.addWidget(b1); row.addWidget(b2)
        layout.addLayout(row)

    def _check_missing_images(self):
        """打开项目后调用：有找不到的原图时提示并引导重新定位。"""
        from ..core.relocate import missing, file_name
        p = self.project
        if getattr(p, "relocated", 0):
            self.statusBar().showMessage(f"项目文件夹位置变了，已按相对位置自动找回 {p.relocated} 张图片。", 10000)
        miss = missing(p)
        if not miss:
            return
        names = "\n".join(f"• {file_name(im.path)}" for im in miss[:8]) + ("\n…" if len(miss) > 8 else "")
        r = QMessageBox.question(
            self, "找不到原图",
            f"有 {len(miss)} 张图片在原来的位置找不到（项目可能被拷到了另一台电脑，或原图被移动了）：\n{names}\n\n"
            "已有的识别、配准结果都还在。是否现在选择图片所在的文件夹，让软件自动找回？")
        if r == QMessageBox.Yes:
            self.relocate_images()

    def relocate_images(self):
        from ..core.relocate import missing, relocate, summary_text
        if self.project is None:
            return
        if not missing(self.project):
            self.info("项目中所有图片都能找到，不需要重新定位。")
            return
        d = QFileDialog.getExistingDirectory(self, "选择图片现在所在的文件夹（会搜索所有子文件夹）",
                                             str(self.project.folder.parent))
        if not d:
            return
        project = self.project

        def done(res):
            self._refresh_images()
            if self.image is not None and self.slide is None and self.image in project.images:
                self._on_image_changed(project.images.index(self.image))
            self.info(summary_text(res))

        self.run_task(lambda task: relocate(project, d, progress=task.report, cancel=task.is_cancelled), done,
                      label="正在搜索图片…")

    def replace_image_file(self):
        from ..core.slide_io import open_slide
        r = self.images_table.currentRow()
        if self.project is None or r < 0:
            self.info("请先在“项目中的图片”表格里选中一张图片。")
            return
        im = self.project.images[r]
        f, _ = QFileDialog.getOpenFileName(self, f"为 {Path(im.path).name} 选择新文件", str(self.project.folder.parent),
                                           "荧光全片图像 (*.qptiff *.tif *.tiff);;所有文件 (*)")
        if not f:
            return
        try:
            sl = open_slide(f)
            shape, nch = sl.level_shapes[0], len(sl.channels)
            sl.close()
        except Exception as e:  # noqa: BLE001
            self.error(f"这个文件无法打开：{e}")
            return
        if nch != len(self.project.channel_names):
            self.error(f"新文件有 {nch} 个通道，项目中是 {len(self.project.channel_names)} 个，不能替换。")
            return
        bad_size = im.sections and any(s.bbox0[0] + s.bbox0[2] > shape[1] * 1.01 or s.bbox0[1] + s.bbox0[3] > shape[0] * 1.01
                                       for s in im.sections)
        if bad_size and QMessageBox.question(
                self, "尺寸不一致", "新文件的图像尺寸和原来的脑片方框对不上，可能不是同一次扫描。仍然替换吗？") != QMessageBox.Yes:
            return
        if Path(f).stem != im.key:
            self.info(f"注意：新文件名（{Path(f).stem}）和原来（{im.key}）不同。已有的识别结果按原文件名保存，"
                      "替换后会找不到，需要重新识别。建议把文件改回原来的名字后再替换。")
        im.path = f; im.error = ""
        self.project.save(); self._refresh_images()
        if self.image is im:
            self._on_image_changed(r)
