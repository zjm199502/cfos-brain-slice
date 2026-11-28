"""教学模式：每一步顶部显示“本步要点”（做什么、好的结果长什么样、常见错误）和参考图，
并提供一个示例项目，让新同学用一张示例脑片完整练习一遍。"""
from __future__ import annotations

from ..core.atlas_choice import has_registration
import json
import shutil
from pathlib import Path

from qtpy.QtCore import Qt, QSettings
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..core.resources import resources_dir
from .layout import fit_window

TEACH_DIR = "teaching"
DEMO_FILE = "示例脑片.ome.tif"

# 每一步：(要点 HTML, [(参考图文件名, 说明, 好/坏)])
LESSONS = {
    0: ("<b>目标</b>：建一个项目文件夹，把要分析的图片加进来，并确认通道名称。<br>"
        "<b>做法</b>：新建项目（选一个空文件夹）→ 添加图片 → 在“通道名称”里把每个通道选对。<br>"
        "<b>常见错误</b><br>"
        "• 通道选错（例如把 cFos 通道选成 DAPI），共标结果会完全反过来；<br>"
        "• 一个项目里混入通道数或染色不同的图片；<br>"
        "• 项目建在 U 盘或网盘同步文件夹里：速度慢，断开时容易损坏。<br>"
        "<b>第一次用？</b>点下面的“打开示例项目”，用一张示例脑片把 8 步完整走一遍（约 20 分钟）。"
        "示例图的 620 通道是 NeuN（神经元核标记），在软件里按 Red 处理。", []),
    1: ("<b>目标</b>：每张脑片一个沿组织边缘的黄色轮廓。<br>"
        "<b>做法</b>：选择处理范围，拆分脑片，检查凹陷、破损和暗边缘。可点“手绘脑片轮廓”沿边缘连续描画，或在脑片图层选中原轮廓后插入、拖动顶点，"
        "再点“保存轮廓修改”。已有脑片可点“沿现有脑片范围重新贴边”。<br>"
        "碎片或灰尘可在表格里取消“分析”。折叠、气泡等可另画排除区域。", []),
    2: ("<b>目标</b>：调整各通道的显示和识别输入。<br>"
        "<b>做法</b>：按需开启原始通道串色校正，再逐通道调整亮度。"
        "手调亮度会勾选“识别使用此亮度和伽马”。去卷积可按通道开启，在当前视野预览效果。<br>"
        "<b>常见错误</b>：调得太亮，细胞连成一片，以为“识别漏了很多”；调得太暗，看不见弱阳性细胞。"
        "判断识别好坏时请用合适的亮度。",
        [("contrast_good.png", "合适：细胞边界清楚，背景接近黑色", True),
         ("contrast_bright.png", "太亮：细胞连成一片，看不出边界", False),
         ("contrast_dim.png", "太暗：弱阳性细胞看不见", False)]),
    3: ("<b>目标</b>：每个细胞被一个轮廓圈住，不多圈、不漏圈。<br>"
        "<b>做法</b>：选切片厚度和项目图块大小（默认边长 2048 像素，降采样的通道按模型输入像素计），为各通道选择识别分辨率。"
        "也可点“鼠标点选图块”，单击选中或取消；已有结果可直接按块导出。放大到一小块 → 点“预览所选区域的细胞识别”，查看虚线圆标出的细胞 → "
        "<b>在亮、暗、密集三种区域各预览一次</b> → 点“选择处理范围并识别…”，勾选本次要处理的图片或脑片。<br>"
        "<b>常见错误</b><br>"
        "• 只看了一个地方就认为参数合适；<br>"
        "• 不同批次（染色深浅不同）的数据随意改参数，导致组间不可比：同一个实验请用同一套参数（可存为模板）；<br>"
        "• 40 µm 厚切片没有选“厚切片”预设，失焦的模糊细胞也被计入。",
        [("detect_good.png", "合适：每个细胞一个轮廓", True),
         ("detect_low.png", "阈值太低：背景、碎片也被圈出", False),
         ("detect_high.png", "阈值太高：暗一点的细胞被漏掉", False)]),
    4: ("<b>目标</b>：确定每个阳性细胞有没有细胞核（DAPI），以及与其他标记是否共标。<br>"
        "<b>做法</b>：修改共标参数后，点“重新计算所选脑片的共标”。<br>"
        "<b>常见错误</b><br>"
        "• 共标距离设得太大，把相邻的两个细胞当成共标。",
        [("coloc_example.png", "黄圈 = 有 DAPI 核的阳性细胞（计入）；紫圈 = 没有核的信号（不计入）", True)]),
    5: ("<b>目标</b>：让图谱脑区边界与组织对齐，这样每个细胞才会被分到正确的脑区。<br>"
        "<b>做法</b>：自动配准 → 逐张检查：先看前后位置（Bregma）对不对，再看轮廓和内部结构"
        "（胼胝体、前连合、海马、脑室）→ 用前后、倾斜、缩放微调。少数自动对不上的地方（组织变形、CA1/CA3 交界）"
        "点“添加标志点”，把白线上的结构拖到它在组织上的真实位置，边界立即更新，可单独删除或撤销。"
        "同一只动物的连续切片，最后用“排序和联合校正”统一切片角度。<br>"
        "<b>常见错误</b><br>"
        "• 只看外轮廓，不看内部结构：外轮廓对上了，脑区仍可能整体错位；<br>"
        "• 没发现脑片在玻片上是左右翻转的（看一侧有划痕/标记时尤其要注意）。",
        [("reg_good.png", "正确：边界线与组织轮廓、内部结构都吻合", True),
         ("reg_bad.png", "偏了：边界线与组织对不上，细胞会被分到错误的脑区", False)]),
    6: ("<b>目标</b>：按动物比较各组，找出 cFos 有差异的脑区。<br>"
        "<b>要点</b><br>"
        "• 统计单位是<b>动物</b>，不是脑片：同一只动物的脑片先合并；<br>"
        "• 每组至少 3–4 只动物；<br>"
        "• 脑区很多，要看 <b>q 值（FDR 校正后）</b>，不能只看 p 值。<br>"
        "• 功能网络分析看的是“脑区之间是否一起变化”，每组最好 ≥ 8 只；重点报告跨阈值一致、留一法稳定的结果。<br>"
        "<b>常见错误</b>：动物编号写法不一致（“M1”和“m1”会被当成两只）；对照组选错；"
        "网络分析只挑一个阈值下“好看”的结果报告。",
        [("heatmap_example.png", "脑区热图示例：各组平均密度和组间倍数变化（仅显著脑区上色）", True)]),
    7: ("<b>目标</b>：导出结果，检查后归档。<br>"
        "<b>Excel 里的表</b>：按脑片汇总（每张脑片的细胞数、质量检查结果）、按脑区×脑片（每个脑区的数量和密度）、"
        "分析参数（所有参数，写论文方法部分和复查时用）。<br>"
        "<b>导出后先做两件事</b>：看“按脑片汇总”中标红/标黄的脑片；翻一遍“质控图”文件夹。<br>"
        "<b>常见错误</b>：导出时 Excel 文件还开着；不保存分析参数，几个月后无法复现。",
        [("qc_example.png", "质控图：圆圈是识别到的阳性细胞，白线是脑区边界", True)]),
    8: ("<b>目标</b>：把感兴趣的区域放大截图、加上比例尺，做成论文或 PPT 里用的图。<br>"
        "<b>做法</b>：画新的截图框 → 在图上拖出矩形 → 保存截图框 → 选通道、叠加内容和比例尺样式 → 预览 → 导出。<br>"
        "<b>建议</b><br>"
        "• 同一篇论文里的图用同一个比例尺长度和同样大小的截图框（勾选“固定大小”）；<br>"
        "• 用“③ 亮度 / 对比度”把亮度调好再截图，所有分组的图要用相同的亮度设置；<br>"
        "• 投稿用 TIFF（无损）；要在 Illustrator 里改比例尺位置、字体时用 SVG。<br>"
        "<b>常见错误</b>：不同组的图亮度不一致，看起来“差异很大”其实是显示设置不同。", []),
}


def teaching_dir() -> Path:
    return resources_dir() / TEACH_DIR


def demo_answer() -> dict:
    """示例项目的参考答案（默认参数下的计数），用于学生自查。"""
    f = teaching_dir() / "示例答案.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class _Thumb(QLabel):
    def __init__(self, path: Path, caption: str, good: bool, parent=None):
        super().__init__(parent)
        self.path, self.caption = path, caption
        pm = QPixmap(str(path))
        self.setPixmap(pm.scaledToWidth(104, Qt.SmoothTransformation))
        col = "#3c9a4a" if good else "#c0504d"
        self.setStyleSheet(f"border: 2px solid {col};")
        self.setToolTip(f"{'✔' if good else '✘'} {caption}\n（点击放大）")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, _e):
        dlg = QDialog(self.window()); dlg.setWindowTitle("参考图")
        L = QVBoxLayout(dlg)
        fit_window(dlg, 940, 820)
        lb = QLabel()
        lb.setAlignment(Qt.AlignCenter)
        lb.setPixmap(QPixmap(str(self.path)).scaled(
            max(1, dlg.width()-36), max(1, dlg.height()-120), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        L.addWidget(lb)
        c = QLabel(self.caption); c.setWordWrap(True); L.addWidget(c)
        b = QPushButton("关闭"); b.clicked.connect(dlg.accept); L.addWidget(b)
        dlg.show()
        self._dlg = dlg


class TeachingMixin:
    def _teaching_toggle(self, layout):
        self.teach_cb = QCheckBox("教学模式")
        self.teach_cb.setToolTip("在每一步顶部显示要点、参考图和常见错误。熟悉以后可以关掉。")
        on = QSettings("cfos", "cfos脑片分析").value("teaching", True, type=bool)
        self.teach_cb.setChecked(on)
        self.teach_cb.toggled.connect(self._set_teaching)
        layout.addWidget(self.teach_cb)

    def _add_lessons(self):
        """在每个步骤面板顶部插入“本步要点”。在面板都建好之后调用。"""
        self._lesson_boxes = []
        for i in range(self.stack.count()):
            html, imgs = LESSONS.get(i, ("", []))
            box = QGroupBox("📘 本步要点（教学模式）")
            box.setStyleSheet("QGroupBox { border: 1px solid #4a7ab8; margin-top: 8px; padding-top: 6px; }")
            L = QVBoxLayout(box)
            lb = QLabel(html); lb.setWordWrap(True); lb.setTextFormat(Qt.RichText); L.addWidget(lb)
            avail = [(teaching_dir() / f, c, g) for f, c, g in imgs if (teaching_dir() / f).exists()]
            if avail:
                g = QGridLayout(); g.setHorizontalSpacing(6)
                for k, (p, c, good) in enumerate(avail):
                    g.addWidget(_Thumb(p, c, good), 0, k)
                    cap = QLabel(("✔ " if good else "✘ ") + c.split("：")[0]); cap.setWordWrap(True)
                    cap.setMaximumWidth(110); g.addWidget(cap, 1, k)
                L.addLayout(g)
            if i == 0:
                b = QPushButton("打开示例项目（练习用）"); b.clicked.connect(self.open_demo_project); L.addWidget(b)
            if i == 3:
                ans = demo_answer()
                if ans:
                    t = "、".join(f"{k} {v:,} 个" for k, v in ans.get("counts", {}).items())
                    a = QLabel(f"<i>示例项目参考答案（默认参数，Excel“按脑片汇总”中的数字）：{t}。你的结果在 ±5% 以内就说明操作正确。</i>")
                    a.setWordWrap(True); L.addWidget(a)
            w = self.stack.widget(i).widget()
            w.layout().insertWidget(0, box)
            self._lesson_boxes.append(box)
        self._set_teaching(self.teach_cb.isChecked())

    def _set_teaching(self, on):
        QSettings("cfos", "cfos脑片分析").setValue("teaching", bool(on))
        for b in getattr(self, "_lesson_boxes", []):
            b.setVisible(bool(on))
        self._refresh_step_marks()

    # ---------------------------------------------------------- 左侧步骤打勾
    def _step_done(self, i) -> bool:
        p = self.project
        if p is None or not p.images:
            return False
        secs = [s for im in p.images for s in im.sections if s.enabled]
        if i == 0:
            return True
        if i == 1:
            return all(im.sections for im in p.images)
        if i == 3 or i == 4:
            return all(p.has_cells(im) for im in p.images)
        if i == 5:
            return bool(secs) and all(has_registration(s.registration) for s in secs)
        if i == 6:
            rows = p.sample_rows()
            return bool(rows) and all(r["动物"] and r["组别"] for r in rows)
        if i == 7:
            return any(p.folder.glob("*.xlsx"))
        if i == 8:
            return any((p.folder / "截图").glob("*.png")) or any((p.folder / "截图").glob("*.tif"))
        return False

    def _refresh_step_marks(self):
        from .main_window import STEPS
        on = getattr(self, "teach_cb", None) is not None and self.teach_cb.isChecked()
        for i, s in enumerate(STEPS):
            it = self.steps.item(i)
            if it is not None:
                it.setText(s + ("  ✓" if on and i != 2 and self._step_done(i) else ""))
                self.step_picker.setItemText(i, it.text())

    # ---------------------------------------------------------- 示例项目
    def open_demo_project(self):
        src = resources_dir() / "demo" / DEMO_FILE
        if not src.exists():
            self.error("没有找到示例图片（resources/demo）。请重新安装软件。")
            return
        base = QFileDialog.getExistingDirectory(self, "选择放示例项目的位置（会在里面新建“cFos示例项目”文件夹）",
                                                str(Path.home() / "Documents"))
        if not base:
            return
        self._when_task_stops(lambda: self._load_demo_project(src, base))

    def _load_demo_project(self, src, base):
        from ..core.project import Project
        folder = Path(base) / "cFos示例项目"
        img = folder / "原图" / DEMO_FILE
        img.parent.mkdir(parents=True, exist_ok=True)
        if not img.exists():
            shutil.copy2(src, img)
        p = Project.load(folder) if Project.is_project(folder) else Project(folder=folder)
        if not p.images:
            p.import_images(img)
            p.save()
        self._load_project(p)
        self.teach_cb.setChecked(True)
        self.steps.setCurrentRow(1)
        self.info("示例项目已打开。请按左侧 ②→⑧ 的顺序操作，每一步顶部都有要点和参考图。\n"
                  "提示：示例图的 620 通道是 NeuN（神经元核），软件中显示为 Red。")
