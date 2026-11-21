"""项目换电脑后重新定位原图的界面部分。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QPushButton

from ..core.relocate import IMAGE_EXT

IMAGE_FILTER = "荧光全片图像 (" + " ".join("*" + suffix for suffix in sorted(IMAGE_EXT)) + ");;所有文件 (*)"


class RelocateMixin:
    def _relocate_widgets(self, layout):
        row = QHBoxLayout()
        b1 = QPushButton("重新定位找不到的图片…"); b1.clicked.connect(self.relocate_images)
        b1.setToolTip("项目拷到另一台电脑、或原图换了位置后使用：选择图片现在所在的文件夹（可以是整个硬盘），\n"
                      "软件会按文件名在里面（含子文件夹）搜索，找到后自动改好路径。已有的识别、配准结果都保留。")
        b2 = QPushButton("替换选中图片的文件…"); b2.clicked.connect(self.replace_image_file)
        b2.setToolTip("手动为选中的图片指定新文件（例如文件改过名）。请选择同一次扫描的文件。"
                      "同一 VSI 或 CZI 的所有扫描区会一起换到新路径。")
        row.addWidget(b1); row.addWidget(b2)
        layout.addLayout(row)

    def _check_missing_images(self):
        """打开项目后调用：有找不到的原图时提示并引导重新定位。"""
        from ..core.relocate import missing
        p = self.project
        if getattr(p, "relocated", 0):
            self.statusBar().showMessage(f"项目文件夹位置变了，已按相对位置自动找回 {p.relocated} 张图片。", 10000)
        miss = missing(p)
        if not miss:
            return
        names = "\n".join(f"• {im.display_name}" for im in miss[:8]) + ("\n…" if len(miss) > 8 else "")
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
        r = self.images_table.currentRow()
        if self.project is None or r < 0:
            self.info("请先在“项目中的图片”表格里选中一张图片。")
            return
        im = self.project.images[r]
        f, _ = QFileDialog.getOpenFileName(self, f"为 {im.display_name} 选择新文件", str(self.project.folder.parent),
                                           IMAGE_FILTER)
        if not f:
            return
        source, destination = Path(im.path).resolve(), Path(f).resolve()
        related = [entry for entry in self.project.images if Path(entry.path).resolve() == source]
        if any(entry.scene_id is not None for entry in related) and source.suffix.lower() != destination.suffix.lower():
            self.error(f"请选择对应的 {source.suffix.upper()} 原文件，以保留原扫描区编号。")
            return
        if source != destination and any(Path(entry.path).resolve() == destination for entry in self.project.images):
            self.error("所选文件已经作为另一张原图加入项目。请选择选中图片对应的原文件。")
            return
        bad_size = False
        czi_times = {}
        for entry in related:
            try:
                sl = replace(entry, path=str(destination)).open_slide(z_mode=self.project.z_mode)
                try:
                    h, w = sl.level_shapes[0]
                    nch = len(sl.channels)
                    if sl.czi_time_index is not None:
                        czi_times[entry.scene_id] = (sl.czi_time_index, sl.czi_time_count)
                finally:
                    sl.close()
            except Exception as e:  # noqa: BLE001
                self.error(f"无法从新文件读取 {entry.display_name}：{e}")
                return
            if nch != len(self.project.channel_names):
                self.error(f"{entry.display_name} 在新文件中有 {nch} 个通道，项目中是 {len(self.project.channel_names)} 个，不能替换。")
                return
            bad_size |= any(x + sw > w * 1.01 or y + sh > h * 1.01
                            for x, y, sw, sh in (section.bbox0 for section in entry.sections))
        if bad_size and QMessageBox.question(
                self, "尺寸不一致", "新文件的图像尺寸和原来的脑片方框对不上，可能不是同一次扫描。仍然替换吗？") != QMessageBox.Yes:
            return
        for entry in related:
            entry.storage_key = entry.key
            entry.path = str(destination)
            entry.error = ""
            if entry.scene_id in czi_times:
                entry.czi_time_index, entry.czi_time_count = czi_times[entry.scene_id]
        self._reg_cache.clear()
        self.project.save(); self._refresh_images()
        if any(entry is self.image for entry in related):
            self._on_image_changed(self.project.images.index(self.image))
