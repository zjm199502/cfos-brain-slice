"""Build the offline Apple Silicon application from the verified 0.9.0 runtime.

Usage: python packaging/mac/rebase_offline.py BASE.tar.xz OUTPUT_DIR --xz xz
"""
import argparse
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import plistlib
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfos import __version__

BASE_SHA256 = "ba1cb6f80de6a1846901d85fb80eda97230cad30d5f3a2ec0a6b3446331c2c85"
BUNDLE = "FosFolio脑片图册.app/Contents/"
APP = BUNDLE + "Resources/app/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def read_tar(path, xz):
    with subprocess.Popen([xz, "-dc", "-T4", str(path)], stdout=subprocess.PIPE) as decoder:
        with decoder.stdout, tarfile.open(fileobj=decoder.stdout, mode="r|", bufsize=1024*1024) as archive:
            yield archive
            while decoder.stdout.read(1024*1024):
                pass
        if decoder.wait():
            raise RuntimeError("XZ integrity check failed")


def retained(name):
    if name.startswith(APP):
        return name.startswith(APP + "resources/")
    return name not in (BUNDLE+"Info.plist", "更新公告.md")


def new_files():
    plist = ROOT/"packaging/mac/Info.plist"
    info = plistlib.loads(plist.read_bytes())
    assert info["CFBundleShortVersionString"] == info["CFBundleVersion"] == __version__
    notes = Path("packaging/更新公告") / f"{__version__}.md"
    yield "更新公告.md", ROOT/notes
    yield BUNDLE+"Info.plist", plist
    for rel in ("main.py", "README.md", "THIRD_PARTY.md", "requirements.txt", notes):
        yield APP + Path(rel).as_posix(), ROOT/rel
    for file in sorted((ROOT/"cfos").rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix not in {".pyc", ".pyo"}:
            yield APP + file.relative_to(ROOT).as_posix(), file


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def metadata(member):
    return member.type, member.size, member.mode, member.linkname


def build(base, out, xz, threads):
    if digest(base) != BASE_SHA256:
        raise ValueError("Unrecognized base archive")
    out.mkdir(exist_ok=True, parents=True)
    output = out / f"FosFolio-{__version__}-macOS-arm64-Offline-Compact.tar.xz"
    expected = {}
    with tempfile.TemporaryDirectory(dir=out) as work:
        partial = Path(work)/output.name
        with partial.open("wb") as dest, subprocess.Popen(
                [xz, "-4", f"-T{threads}", "-c"], stdin=subprocess.PIPE, stdout=dest) as encoder:
            with encoder.stdin, tarfile.open(fileobj=encoder.stdin, mode="w|", bufsize=1024*1024) as archive:
                with read_tar(base, xz) as originals:
                    for member in originals:
                        if not retained(member.name):
                            continue
                        if member.isfile():
                            reader = HashReader(originals.extractfile(member))
                            archive.addfile(member, reader)
                            checksum = reader.sha.hexdigest()
                        else:
                            archive.addfile(member)
                            checksum = None
                        expected[member.name] = (metadata(member), checksum)
                for name, file in new_files():
                    data = file.read_bytes()
                    member = tarfile.TarInfo(name)
                    member.mode = 0o755 if file.name in ("reg_aladin",) or file.suffix == ".dylib" else 0o644
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
                    expected[name] = (metadata(member), hashlib.sha256(data).hexdigest())
            if encoder.wait():
                raise RuntimeError("XZ compression failed")
        print("Verifying packaged bytes, modes and symbolic links", flush=True)
        with read_tar(partial, xz) as archive:
            for member in archive:
                sha = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() if member.isfile() else None
                if expected.pop(member.name) != (metadata(member), sha):
                    raise ValueError(f"Archive verification failed for {member.name}")
        if expected:
            raise ValueError("Incomplete archive")
        partial.replace(output)
    print(f"{output.name}  {output.stat().st_size}  {digest(output)}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--xz", default="xz")
    cli.add_argument("--threads", type=int, default=8)
    args = cli.parse_args()
    build(args.base, args.out, args.xz, args.threads)
