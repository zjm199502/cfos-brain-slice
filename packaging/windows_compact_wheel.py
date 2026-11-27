"""Expose large deflated wheel members to Inno compression and restore the official wheel."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zipfile
import zlib


BLOCK = 1024 * 1024


def official_wheel(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entry, = (item for item in manifest["wheels"] if item["file"].startswith("torch-"))
    name = entry["file"]
    if Path(name).name != name or not name.endswith(".whl"):
        raise ValueError("Invalid torch wheel filename")
    return bundle / "wheels" / name, entry


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest() == entry["sha256"]


def copy_bytes(src, out, count):
    while count:
        data = src.read(min(BLOCK, count))
        if not data:
            raise ValueError("Unexpected end of compact wheel data")
        out.write(data)
        count -= len(data)


def deflate(src, consume, raw_copy=None):
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    for data in iter(lambda: src.read(BLOCK), b""):
        if raw_copy:
            raw_copy.write(data)
        consume(compressor.compress(data))
    consume(compressor.flush())


def pack(bundle, media):
    wheel, entry = official_wheel(bundle)
    if not valid(wheel, entry):
        raise ValueError("Original torch wheel failed SHA-256 verification")
    media.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        members = [i for i in archive.infolist()
                   if i.compress_type == zipfile.ZIP_DEFLATED and i.compress_size >= 4 * BLOCK]

    def extract(pair):
        index, member = pair
        raw = media / f"part-{index}.bin"
        # Check the complete DEFLATE stream against the upstream bytes. Different
        # producer settings are allowed: those entries simply stay in gaps.bin.
        with wheel.open("rb") as original, zipfile.ZipFile(wheel) as archive:
            original.seek(member.header_offset + 26)
            name_size, extra_size = struct.unpack("<HH", original.read(4))
            original.seek(name_size + extra_size, 1)
            offset = original.tell()

            def compare(data):
                if original.read(len(data)) != data:
                    raise ArithmeticError

            try:
                with archive.open(member) as src, raw.open("wb") as out:
                    deflate(src, compare, out)
                if original.tell() != offset + member.compress_size:
                    raise ArithmeticError
            except ArithmeticError:
                raw.unlink()
                return None
        return {"index": index, "offset": offset, "size": member.compress_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = sorted((p for p in pool.map(extract, enumerate(members)) if p),
                       key=lambda p: p["offset"])
    if not parts:
        raise ValueError("No upstream DEFLATE streams could be reproduced by this Python")
    with wheel.open("rb") as src, (media / "gaps.bin").open("wb") as gaps:
        for part in parts:
            copy_bytes(src, gaps, part["offset"] - src.tell())
            src.seek(part["size"], 1)
        shutil.copyfileobj(src, gaps, BLOCK)
    (media / "plan.json").write_text(json.dumps(parts), encoding="utf-8")
    print(f"{len(parts)} members, {sum(p['size'] for p in parts):,} original DEFLATE bytes", flush=True)


def restore(bundle, media):
    wheel, entry = official_wheel(bundle)
    if valid(wheel, entry):
        print("Official torch wheel already present.", flush=True)
        return
    parts = json.loads((media / "plan.json").read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(dir=wheel.parent, delete=False) as tmp:
        target = Path(tmp.name)
    try:
        with target.open("wb") as out, (media / "gaps.bin").open("rb") as gaps:
            for part in parts:
                offset, size = part["offset"], part["size"]
                if not (out.tell() <= offset < offset + size <= entry["size"]):
                    raise ValueError("Invalid compact wheel offsets")
                copy_bytes(gaps, out, offset - out.tell())
                out.seek(size, 1)
            copy_bytes(gaps, out, entry["size"] - out.tell())
            if gaps.read(1):
                raise ValueError("Invalid compact wheel padding")
            out.truncate(entry["size"])

        def rebuild(part):
            index = int(part["index"])
            with (media / f"part-{index}.bin").open("rb") as src, target.open("r+b") as out:
                out.seek(part["offset"])
                end = out.tell() + part["size"]

                def write(data):
                    if out.tell() + len(data) > end:
                        raise ValueError("Rebuilt stream exceeds its original size")
                    out.write(data)

                deflate(src, write)
                if out.tell() != end:
                    raise ValueError("Rebuilt stream was truncated")

        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
            list(pool.map(rebuild, parts))
        if not valid(target, entry):
            raise ValueError("Restored torch wheel does not match its official SHA-256")
        target.replace(wheel)
        print("Restored and verified official torch wheel.", flush=True)
    finally:
        target.unlink(missing_ok=True)


if __name__ == "__main__":
    {"pack": pack, "restore": restore}[sys.argv[1]](*(Path(arg) for arg in sys.argv[2:]))
