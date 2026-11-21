"""Make a Finder-extractable solid tar.xz from an existing macOS offline ZIP.

Build on any platform with Python 3.11+ and XZ Utils. End users on macOS
can also run /usr/bin/tar -xf <archive>. Verify every member before delivery.
"""
import argparse
import calendar
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile
import zipfile


@contextmanager
def source_bytes(archive, entry):
    if entry.filename == "安装说明.txt":
        guide = archive.read(entry).decode("utf-8")
        yield io.BytesIO(guide.replace("解压 ZIP", "解压 tar.xz 安装包").encode("utf-8"))
    else:
        with archive.open(entry) as stream:
            yield stream


def tar_entry(archive, entry):
    path = PurePosixPath(entry.filename)
    if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
        raise ValueError(f"Invalid archive path: {entry.filename}")
    mode = entry.external_attr >> 16
    info = tarfile.TarInfo(path.as_posix())
    info.mode = stat.S_IMODE(mode)
    info.mtime = calendar.timegm(entry.date_time)
    if stat.S_ISDIR(mode):
        info.type = tarfile.DIRTYPE
    elif stat.S_ISLNK(mode):
        info.type = tarfile.SYMTYPE
        info.linkname = archive.read(entry).decode("utf-8")
        target = PurePosixPath(info.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise ValueError(f"Invalid link target: {info.name}")
    elif stat.S_ISREG(mode):
        if entry.filename == "安装说明.txt":
            with source_bytes(archive, entry) as guide:
                info.size = len(guide.getbuffer())
        else:
            info.size = entry.file_size
    else:
        raise ValueError(f"Unsupported archive member: {info.name}")
    return info


def verify(zip_source, compact):
    with zipfile.ZipFile(zip_source) as source, tarfile.open(compact, "r|xz") as archive:
        remaining = {entry.filename.rstrip("/"): entry for entry in source.infolist()}
        if len(remaining) != len(source.infolist()):
            raise ValueError("Duplicate member in source archive")
        for member in archive:
            entry = remaining.pop(member.name)
            expected = tar_entry(source, entry)
            fields = ("mode", "type", "size", "mtime", "linkname")
            if any(getattr(member, field) != getattr(expected, field) for field in fields):
                raise ValueError(f"Metadata changed: {member.name}")
            if member.isfile():
                with source_bytes(source, entry) as original, archive.extractfile(member) as decoded:
                    digest = hashlib.file_digest(original, "sha256").digest()
                    if hashlib.file_digest(decoded, "sha256").digest() != digest:
                        raise ValueError(f"File changed: {member.name}")
        if remaining:
            raise ValueError(f"Missing files in compact archive: {list(remaining)[:4]}")


def build(zip_source, output, xz, threads):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        partial = Path(staging) / output.name
        with zipfile.ZipFile(zip_source) as source, partial.open("wb") as compressed:
            if not any(e.filename.endswith("/resources/cellpose4/cpsam_v2") for e in source.infolist()):
                raise ValueError("Expected the full Cellpose offline ZIP")
            with subprocess.Popen([xz, "-5", f"--threads={threads}", "-c"], stdin=subprocess.PIPE,
                                  stdout=compressed) as encoder:
                with encoder.stdin, tarfile.open(fileobj=encoder.stdin, mode="w|", bufsize=1024*1024) as archive:
                    for entry in source.infolist():
                        info = tar_entry(source, entry)
                        if info.isfile():
                            with source_bytes(source, entry) as stream:
                                archive.addfile(info, stream)
                        else:
                            archive.addfile(info)
                if encoder.wait():
                    raise RuntimeError("xz compression failed")
        print(f"Comparing every member of {output.name} with the source ZIP", flush=True)
        verify(zip_source, partial)
        partial.replace(output)
    print(f"{output}: {output.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xz", default="xz")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    build(args.zip_source, args.output, args.xz, args.threads)
