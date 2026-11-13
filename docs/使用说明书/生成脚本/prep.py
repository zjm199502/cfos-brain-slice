"""把 md 中引用的图片缩放到合适大小，写 dims.json。"""
import json, re, sys
from pathlib import Path
from PIL import Image
root = Path("/home/user/manual"); out = root / "build" / "img"; out.mkdir(parents=True, exist_ok=True)
md = (root / "使用说明书.md").read_text()
dims = {}
for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
    src = root / ref
    if not src.exists():
        print("MISSING", ref); continue
    im = Image.open(src)
    has_alpha = im.mode in ("RGBA", "LA", "P")
    im = im.convert("RGB")
    if im.height > 2.2 * im.width:   # 很高的面板截图：裁掉底部空白，分成左右两栏
        import numpy as np
        a = np.asarray(im).astype(float)
        rowstd = a.std(axis=(1, 2))
        bg = a[-1].mean(axis=0)
        content = np.nonzero(np.abs(a - bg).max(axis=(1, 2)) > 20)[0]
        bottom = int(content.max()) + 12 if len(content) else im.height
        im = im.crop((0, 0, im.width, min(bottom, im.height))); a = a[:im.height]; rowstd = rowstd[:im.height]
        mid = im.height // 2
        same = np.r_[False, (np.abs(np.diff(a, axis=0)).max(axis=(1, 2)) < 3)]
        cand = [y for y in range(mid - 400, mid + 400) if 4 < y < im.height and same[y - 3:y + 1].all()]
        cut = min(cand, key=lambda y: abs(y - mid)) if cand else mid
        top, bot = im.crop((0, 0, im.width, cut)), im.crop((0, cut, im.width, im.height))
        H = max(top.height, bot.height)
        c = Image.new("RGB", (im.width * 2 + 24, H), "white")
        c.paste(top, (0, 0)); c.paste(bot, (im.width + 24, 0)); im = c
    if im.width > 1800:
        im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
    photo = src.stat().st_size > 300_000
    name = Path(ref).stem + (".jpg" if photo else ".png")
    if photo:
        im.save(out / name, quality=92, subsampling=0)
    else:
        im.save(out / name, optimize=True)
    dims[ref] = {"file": str(out / name), "w": im.width, "h": im.height, "type": "jpg" if photo else "png"}
json.dump(dims, open(root / "build" / "dims.json", "w"), ensure_ascii=False, indent=1)
print(len(dims), "images")
