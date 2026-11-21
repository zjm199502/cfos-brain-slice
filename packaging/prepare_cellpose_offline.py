"""Build the verified Cellpose payload for a FosFolio CPython 3.13 offline installer.

Run with Python 3.11+ and packaging installed. Specify --cache more than once to
reuse official wheels, a pip http-v2 cache, or already verified model files.
The packaged application does not need network access or this build script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename


LOCKS = Path(__file__).with_name("cellpose_offline")
MODEL = {
    "file": "cpsam_v2",
    "url": "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2",
    "size": 1_233_586_851,
    "sha256": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
}
NOTICE = """FosFolio / Cellpose 4 offline components

Cellpose 4.2.1: https://github.com/MouseLand/cellpose
Cellpose-SAM v2 pretrained weights: https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2
Model card and license declaration (BSD-3-Clause): https://huggingface.co/mouseland/cellpose-sam/blob/main/README.md
The upstream Cellpose BSD-3-Clause license is reproduced in LICENSE-Cellpose.txt.
All Python wheels are unmodified upstream distributions. Their original license
notices, including PyTorch's third-party notices, are retained inside each wheel.
The manifest identifies the exact file, download source and SHA-256 for each component.
"""


def valid(file, item):
    if not file.is_file() or file.stat().st_size != item["size"]:
        return False
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]


def cached_files(directories, items):
    wanted_sizes = {item["size"] for item in items}
    matches = {}
    for directory in directories:
        files = [directory] if directory.is_file() else directory.rglob("*")
        for file in files:
            if file.name != MODEL["file"] and file.suffix not in {".body", ".whl"}:
                continue
            size = file.stat().st_size
            if size in wanted_sizes:
                matches.setdefault(size, []).append(file)
    return matches


def fetch(item, destination, cache):
    target = destination / item["file"]
    if valid(target, item):
        return
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=destination.parent) as temp:
        partial = Path(temp) / item["file"]
        for candidate in cache.get(item["size"], []):
            if valid(candidate, item):
                try:
                    os.link(candidate, partial)
                except OSError:
                    shutil.copyfile(candidate, partial)
                break
        else:
            for attempt in range(3):
                try:
                    request = Request(item["url"], headers={"Accept-Encoding": "identity"})
                    with urlopen(request, timeout=90) as src, partial.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    if not valid(partial, item):
                        raise ValueError(f"Official SHA-256 or size mismatch: {target.name}")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        os.replace(partial, target)
    print(f"Verified {target.name}", flush=True)


def check_dependencies(wheels, platform):
    windows = platform == "win-amd64"
    platforms = ["win_amd64"] if windows else list(mac_platforms((15, 0), "arm64"))
    tags = set(cpython_tags((3, 13), abis=["cp313"], platforms=platforms))
    tags.update(compatible_tags((3, 13), interpreter="cp313", platforms=platforms))
    env = default_environment()
    env.update(python_version="3.13", python_full_version="3.13.16", extra="",
               implementation_name="cpython", platform_python_implementation="CPython",
               sys_platform="win32" if windows else "darwin", os_name="nt" if windows else "posix",
               platform_system="Windows" if windows else "Darwin", platform_machine="AMD64" if windows else "arm64")
    versions = {}
    requirements = []
    for wheel in wheels:
        name, version, _, wheel_tags = parse_wheel_filename(wheel.name)
        if wheel_tags.isdisjoint(tags):
            raise ValueError(f"Incompatible {platform} wheel: {wheel.name}")
        if name in versions:
            raise ValueError(f"Duplicate distribution: {name}")
        versions[name] = version
        with ZipFile(wheel) as archive:
            metadata, = [f for f in archive.namelist() if f.count("/") == 1 and f.endswith(".dist-info/METADATA")]
            info = BytesParser().parsebytes(archive.read(metadata))
        if canonicalize_name(info["Name"]) != name or str(version) != info["Version"]:
            raise ValueError(f"Wheel metadata mismatch: {wheel.name}")
        requirements.extend((name, Requirement(raw)) for raw in info.get_all("Requires-Dist", []))
    for name, req in requirements:
        if req.marker and not req.marker.evaluate(env):
            continue
        version = versions.get(canonicalize_name(req.name))
        if version is None or version not in req.specifier:
            raise ValueError(f"{name} requires {req}; found {version}")


def build(platform, output, directories):
    lock = json.loads((LOCKS / f"{platform}.json").read_text(encoding="utf-8"))
    wheels = output / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    if set(p.name for p in wheels.iterdir()) - {i["file"] for i in lock}:
        raise ValueError("The output wheel directory contains files outside the lock.")
    items = [*lock, MODEL]
    cache = cached_files(directories, items)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: fetch(i, output if i is MODEL else wheels, cache), items))
    check_dependencies([wheels / item["file"] for item in lock], platform)
    with ZipFile(next(wheels.glob("cellpose-*.whl"))) as archive:
        license_file, = [p for p in archive.namelist() if p.endswith(".dist-info/licenses/LICENSE")]
        (output / "LICENSE-Cellpose.txt").write_bytes(archive.read(license_file))
    (output / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    manifest = {"format": 1, "platform": platform, "python": "3.13", "cellpose": "4.2.1",
                "wheels": lock, "weights": MODEL}
    temp = output / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(output / "manifest.json")
    print(f"{output}: verified {len(lock)} wheels and model", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=["win-amd64", "mac-arm64"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", action="append", type=Path, default=[])
    args = parser.parse_args()
    build(args.platform, args.output, args.cache)
