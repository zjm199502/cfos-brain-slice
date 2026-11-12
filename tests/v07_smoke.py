"""v0.7 新功能的界面测试：识别准确率检查、整张脑片配准图导出（矢量轮廓）、诊断包、平滑边界显示。
用法：xvfb-run python tests/run.py tests/v07_smoke.py <已配准且已识别的项目文件夹> <截图目录>"""
import sys, time, zipfile, json, faulthandler; faulthandler.enable()
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from napari.qt import get_qapp
from qtpy.QtGui import QPainter, QImage
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow, STEPS
from cfos.core.project import Project

proj, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 950); w.show()
msgs = []
w.info = lambda t: (msgs.append(t), print("INFO:", t.replace("\n", " | "), flush=True))
w.error = lambda t: (msgs.append(t), print("ERROR:", t, flush=True))
w._open_folder = lambda p: print("open folder", p)


def pump(s=1.0):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def wait_task():
    while w.task is not None and w.task.isRunning():
        pump(0.2)
    pump(1.0)


def shot(name):
    pump(1.2)
    full = w.grab().toImage()
    canvas = w.qt_viewer.screenshot(flash=False)
    h, wd = canvas.shape[:2]
    qi = QImage(canvas.copy().data, wd, h, 4 * wd, QImage.Format_RGBA8888)
    pos = w.qt_viewer.canvas.native.mapTo(w, w.qt_viewer.canvas.native.rect().topLeft())
    pt = QPainter(full); pt.drawImage(pos, qi); pt.end()
    full.save(str(out / f"{name}.png")); print("shot", name, flush=True)


w._load_project(Project.load(proj)); pump(2)
for i, nm in enumerate(STEPS):
    w.steps.setCurrentRow(i); pump(0.6)
    mw = w.stack.widget(i).minimumSizeHint().width()
    print(nm, "min width", mw, "超宽!" if mw > 440 else "")
    assert mw <= 440, nm

# ---------------------------------------------------------------- 1. 识别准确率检查
w.steps.setCurrentRow(3); pump(1)
s = w.image.sections[0]; px = w.slide.pixel_size_um
red = w.project.channel_names.index("Red")
w.acc_ch.setCurrentIndex(red)
df = w.cells_for(w.image)
g = df[(df["channel"] == red) & (df["section"] == s.name)]
# 视野中心放到一个细胞较多的地方
cx, cy = float(g["x"].median()), float(g["y"].median())
w.viewer.camera.center = (cy, cx); pump(0.3)
w.acc_size.setValue(250)
w.acc_new_box(); pump(0.5)
box = w._acc_box0(); print("box", box)
assert box is not None and abs(box[2] * px - 250) < 1
inb = g[(g["x"] >= box[0]) & (g["x"] < box[0] + box[2]) & (g["y"] >= box[1]) & (g["y"] < box[1] + box[3])]
inb = inb[inb["manual"] == 0]
n = len(inb); print("auto cells in box", n); assert n >= 10
# 模拟人工：漏点 3 个软件细胞（= 软件误检 3），另加 4 个软件没找到的细胞（= 漏检 4），其余点在细胞中心附近 2 µm 内
rng = np.random.default_rng(1)
keep = inb.iloc[3:]
man = keep[["x", "y"]].to_numpy() + rng.uniform(-2 / px, 2 / px, (len(keep), 2))
far = []
while len(far) < 4:
    p = np.array([box[0] + rng.uniform(0.05, 0.95) * box[2], box[1] + rng.uniform(0.05, 0.95) * box[3]])
    if np.min(np.hypot(*(inb[["x", "y"]].to_numpy() - p).T)) > 15 / px:
        far.append(p)
man = np.vstack([man, far])
layer = w.viewer.layers["人工计数"]
layer.add(man[:, ::-1]); pump(0.3)
shot("v1_accuracy_counting")
w.acc_compute(); pump(1)
rec = w.project.accuracy[-1]; print("record", {k: rec[k] for k in ("人工计数", "软件识别", "配对成功", "漏检", "误检", "查准率", "查全率", "F1")})
assert rec["软件识别"] == n and rec["人工计数"] == len(man)
assert rec["配对成功"] == n - 3 and rec["误检"] == 3 and rec["漏检"] == 4
shot("v2_accuracy_result")
assert "累计" in w.acc_info.text()
from cfos.core.export import params_table
pt = params_table(w.project)
row = pt[pt["参数"].str.contains("识别准确率")]
print(row.to_string()); assert len(row) == 1

# ---------------------------------------------------------------- 2. 平滑边界显示（⑥）
w.steps.setCurrentRow(5); pump(2)
lay = w.viewer.layers["脑区轮廓"]
print("boundary layer", lay.data.shape, "thumb", w._reg_lab[2].shape)
assert lay.data.shape[0] == 3 * w._reg_lab[2].shape[0]
w.viewer.reset_view(); pump(0.5)
shot("v3_smooth_boundaries")

# ---------------------------------------------------------------- 3. 整张脑片配准图（矢量轮廓）
w.steps.setCurrentRow(8); pump(1)
w.fig_svg.setChecked(True); w.fig_tif.setChecked(False); w.fig_png.setChecked(True)
w.fig_res.setCurrentIndex(w.fig_res.findData(2.0))
w.fig_region_names.setChecked(True); w.rg_color.setCurrentText("白"); w.rg_mid.setChecked(True)
pump(0.3)
w.fig_export_sections(False); wait_task()
d = proj / "截图" / "配准图"
files = sorted(d.glob("*")); print("section files", [f.name for f in files])
svg = [f for f in files if f.suffix == ".svg"]; png = [f for f in files if f.suffix == ".png"]
assert svg and png
txt = svg[0].read_text(encoding="utf-8")
n_paths = txt.count("<path "); n_text = txt.count("<text ")
print("svg paths", n_paths, "texts", n_text, "size MB %.1f" % (svg[0].stat().st_size / 1e6))
assert 'id="region-outlines"' in txt and n_paths > 50 and n_text > 10 and 'id="midline"' in txt
meta = json.loads((d / (svg[0].stem + "_参数.json")).read_text(encoding="utf-8"))
print("meta", {k: meta[k] for k in ("pixel_um", "out_w", "out_h", "regions_drawn")})
# 只画轮廓（白底黑线）+ 图谱颜色
w.rg_only.setChecked(True); pump(0.2)
w.fig_export_sections(False); wait_task()
w.rg_only.setChecked(False); w.rg_color.setCurrentText("图谱颜色"); w.fig_svg.setChecked(False); pump(0.2)
w.fig_export_sections(False); wait_task()
print("after variants", sorted(f.name for f in d.glob("*.png")))
assert any("轮廓" in f.name for f in d.glob("*.png"))
# ⑥ 里的按钮
w.steps.setCurrentRow(5); pump(1)
n_before = len(list(d.glob("*")))
w._reg_export_figure(); wait_task()
assert msgs[-1].startswith("已导出")
shot("v4_after_export")

# ---------------------------------------------------------------- 4. 诊断包
w.steps.setCurrentRow(0); pump(0.5)
from cfos.core.diagnostics import make_bundle
z = make_bundle(w.project, note="测试：自动生成")
names = zipfile.ZipFile(z).namelist(); print("bundle", z.name, z.stat().st_size, names)
assert "电脑和软件信息.json" in names and "项目/cfos_project.json" in names and "问题描述.txt" in names
info = json.loads(zipfile.ZipFile(z).read("电脑和软件信息.json"))
print("info", {k: info[k] for k in ("软件版本", "CPU 使用", "细胞识别神经网络") if k in info})
assert z.stat().st_size < 5_000_000
print("DONE")
