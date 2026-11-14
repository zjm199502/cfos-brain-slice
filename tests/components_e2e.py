"""微调组件按需下载的端到端测试（模拟安装包里没有 TensorFlow）。
用法：PYTHONPATH=tests/no_tf python tests/run.py tests/components_e2e.py <make_components 输出文件夹> <项目（已识别）> <临时目录>
检查：断点续传、分卷校验失败重下、解压、自检、训练（几步）后自动转 ONNX、识别不需要 TensorFlow、离线包安装、错误平台拒绝。"""
import http.server, json, os, shutil, sys, threading, time, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
srv_dir, proj_src, tmp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
assert "tensorflow" not in sys.modules
from cfos.core import components as C
os.environ["CFOS_COMPONENTS_DIR"] = str(tmp / "组件")
assert not C.tensorflow_available(), "测试环境里不应该能找到 TensorFlow（PYTHONPATH 要包含 tests/no_tf）"

# ---- 支持 Range 的本地服务器
hits = []
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(srv_dir), **k)
    def log_message(self, *a):
        pass
    def do_GET(self):
        from urllib.parse import unquote
        p = srv_dir / unquote(self.path.lstrip("/"))
        rng = self.headers.get("Range")
        hits.append((p.name, rng))
        if not p.is_file():
            self.send_error(404); return
        data = p.read_bytes(); start = 0
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data) - start)); self.end_headers()
        self.wfile.write(data[start:])
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["CFOS_COMPONENTS_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"

man = json.loads((srv_dir / "components.json").read_text(encoding="utf-8"))
pack = man["packs"][C.platform_tag()]
dl = C.components_root() / "下载中"; dl.mkdir(parents=True)
p1, p2 = pack["parts"][0], pack["parts"][1]
src1 = (srv_dir / p1["file"]).read_bytes()
(dl / (Path(p1["file"]).name + ".part")).write_bytes(src1[:10_000_000])        # 断网：第 1 卷下了 10 MB
(dl / Path(p2["file"]).name).write_bytes(b"x" * p2["size"])                     # 第 2 卷内容损坏
t = time.time(); last = []
info = C.ensure_tensorflow(progress=lambda d, n, s: last.append((d, n, s)))
print(info, f"{time.time() - t:.0f} 秒")
print("最后的进度：", last[-1][2])
r1 = [h for h in hits if h[0] == Path(p1["file"]).name]
assert r1 and r1[0][1] == "bytes=10000000-", f"第 1 卷应从 10 MB 处续传：{r1}"
assert any(h[0] == Path(p2["file"]).name for h in hits), "损坏的第 2 卷应重新下载"
assert C.tensorflow_available() and C.installed_info()["selftest_ok"]
print("自检：", C.installed_info()["selftest"])
assert not list(dl.glob("*.zip*")), "下载的临时文件应清理"

# ---- 训练几步（项目副本），训练后自动转换 ONNX，识别不用 TensorFlow
proj = tmp / "proj"; shutil.copytree(proj_src, proj, ignore=shutil.ignore_patterns("*.xlsx", "诊断包*", "截图", "质控图*", "细胞mask*"))
from cfos.core.project import Project
from cfos.core import finetune as F, detection as det
from cfos.core.slide_io import open_slide
p = Project.load(proj); im = p.images[0]; sl = open_slide(im.path); s = im.sections[0]
det.MODEL_ROOT = str(proj / "models")
c = p.channel_names.index("Red")
cx, cy = s.bbox0[0] + s.bbox0[2] / 2, s.bbox0[1] + s.bbox0[3] / 2
for dx in (-800, 0, 800):
    x, lab, meta = F.make_patch(p, im, sl, c, (cx + dx, cy))
    F.save_patch(p, c, x, lab, meta)
assert "tensorflow" not in sys.modules, "做标注块时不应加载 TensorFlow"
info = F.finetune(p, c, epochs=1, steps_per_epoch=2)
print("训练：", info)
md = proj / "models" / info["name"]
assert (md / "model.onnx").exists() and info["onnx_max_diff"] < 1e-3
det.forget_model(info["name"])
m = det.get_model(info["name"])
assert type(m).__name__ == "OnnxStarDist", type(m)

# ---- 离线包安装（新的组件目录）、错误平台拒绝
os.environ["CFOS_COMPONENTS_DIR"] = str(tmp / "组件2")
off = srv_dir / pack["offline_file"]
info2 = C.install_pack(off)
assert info2["selftest_ok"], info2
bad = tmp / "wrong.zip"
with zipfile.ZipFile(bad, "w") as z:
    z.writestr("component.json", json.dumps({"version": C.REQUIRED, "platform": "win-amd64"}))
try:
    C.install_pack(bad); raise SystemExit("错误平台的组件包应被拒绝")
except ValueError as e:
    print("错误平台：", e)
print("OK")
