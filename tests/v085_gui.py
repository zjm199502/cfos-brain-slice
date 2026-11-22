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
assert [box.combo.itemData(i) for i in range(box.combo.count())] == [RS.ALL, RS.SUMMARY_ONLY, RS.CUSTOM]
for m in (RS.ALL, RS.SUMMARY_ONLY):
    box.set_values(m)
    assert box.mode() == m
    assert box.layers.isEnabled() == (m == RS.ALL)
    print(m, "|", box.desc.text()[:40])
box.set_values("", 8, True)            # 旧项目：设过“第 8 级 + 不列分层”
assert box.mode() == RS.CUSTOM and RS.selected(RS.CUSTOM, region_ids=box.region_ids()) == RS.selected(RS.DEPTH, 8, True)
box.set_values("", 0, False)
assert box.mode() == RS.ALL
w.stack.setCurrentIndex(7)
for nm, m in (("1_导出_常用脑区", RS.SUMMARY_ONLY), ("2_导出_全部脑区", RS.ALL), ("3_导出_自选脑区", RS.CUSTOM)):
    box.set_values(m, region_ids=[382])
    shot(nm, w.stack.currentWidget())
for nm, m in (("4_预览_常用脑区", RS.SUMMARY_ONLY), ("5_预览_全部脑区", RS.ALL)):
    d = RegionPreviewDialog(m, 0, False, w); d.show(); pump()
    if m == RS.SUMMARY_ONLY:
        d.search.setText("杏仁核"); d.only.setChecked(True)
    shot(nm, d); d.close()
w.stack.setCurrentIndex(6); shot("6_组间统计_脑区范围", w.st_scope.parent())
print("OK")
