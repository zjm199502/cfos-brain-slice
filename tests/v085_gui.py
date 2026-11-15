"""v0.8.5 界面测试：⑧ 导出 和 ⑦ 组间统计 的“脑区范围”选择框、脑区预览窗口。
用法：xvfb-run python tests/run.py tests/v085_gui.py <截图目录>"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from napari.qt import get_qapp
from cfos.gui.i18n import install_chinese
from cfos.gui.main_window import MainWindow
from cfos.gui.region_scope import RegionPreviewDialog
from cfos.core import region_sets as RS

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = get_qapp(); install_chinese(app)
w = MainWindow(); w.resize(1600, 1000); w.show()


def pump(s=0.6):
    t = time.time()
    while time.time() - t < s:
        app.processEvents(); time.sleep(0.02)


def shot(name, widget):
    pump(); widget.grab().save(str(out / f"{name}.png")); print("shot", name, flush=True)


box = w.exp_scope
print("modes:", [box.combo.itemText(i) for i in range(box.combo.count())])
assert box.mode() == RS.ALL and box.max_depth() == 0
for m in (RS.ALL, RS.MAJOR_ONLY, RS.SUMMARY_ONLY, RS.MAJOR_SUMMARY, RS.DEPTH):
    box.set_values(m, 6 if m == RS.DEPTH else 0)
    assert box.mode() == m
    assert box.depth.isVisibleTo(box) == (m == RS.DEPTH)
    assert box.layers.isEnabled() == (m in (RS.ALL, RS.DEPTH))
    print(m, "|", box.desc.text()[:40])
assert box.max_depth() == 6
box.set_values("", 8, True)            # 旧项目：设过“第 8 级 + 不列分层”
assert box.mode() == RS.DEPTH and box.max_depth() == 8 and box.skip_layers()
box.set_values("", 0, False)
assert box.mode() == RS.ALL
w.stack.setCurrentIndex(7)
for nm, m in (("1_导出_常用脑区", RS.SUMMARY_ONLY), ("2_导出_大脑区", RS.MAJOR_ONLY), ("3_导出_按层级", RS.DEPTH)):
    box.set_values(m, 6)
    shot(nm, w.stack.currentWidget())
for nm, m in (("4_预览_常用脑区", RS.SUMMARY_ONLY), ("5_预览_大脑区", RS.MAJOR_ONLY)):
    d = RegionPreviewDialog(m, 0, False, w); d.show(); pump()
    if m == RS.SUMMARY_ONLY:
        d.search.setText("杏仁核"); d.only.setChecked(True)
    shot(nm, d); d.close()
w.stack.setCurrentIndex(6); shot("6_组间统计_脑区范围", w.st_scope.parent())
print("OK")
