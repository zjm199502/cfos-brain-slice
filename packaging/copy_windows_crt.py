"""Stage Microsoft's unmodified release x64 CRT beside the embedded Python process.

Pass a Microsoft.VC*.CRT folder from a licensed Visual Studio VC/Redist/MSVC
installation. Redistribution list: https://learn.microsoft.com/visualstudio/releases/2026/redistribution
"""
from pathlib import Path
import shutil
import struct
import sys


def file_version(path):
    data = path.read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset:pe_offset + 6] != b"PE\0\0\x64\x86":
        raise ValueError(f"Not an x64 Windows binary: {path}")
    fixed_info = data.index(bytes.fromhex("bd04effe00000100"))
    version_hi, version_lo = struct.unpack_from("<II", data, fixed_info + 8)
    mask, flags = struct.unpack_from("<II", data, fixed_info + 24)
    if flags & mask & 3:  # VS_FF_DEBUG or VS_FF_PRERELEASE
        raise ValueError(f"Debug or prerelease DLL: {path}")
    return version_hi >> 16, version_hi & 0xffff, version_lo >> 16, version_lo & 0xffff


def stage(source, target):
    if not source.match("Microsoft.VC*.CRT"):
        raise ValueError("Specify a release Microsoft.VC*.CRT directory from VC/Redist/MSVC.")
    required = {"concrt140", "msvcp140", "msvcp140_1", "msvcp140_2", "msvcp140_atomic_wait",
                "msvcp140_codecvt_ids", "vccorlib140", "vcruntime140", "vcruntime140_1"}
    files = sorted(source.glob("*.dll"))
    missing = required - {p.stem.lower() for p in files}
    if missing:
        raise ValueError(f"Incomplete CRT directory (missing {sorted(missing)})")
    versions = {file_version(p) for p in files}
    if len(versions) != 1:
        raise ValueError(f"Inconsistent Microsoft CRT versions: {versions}")
    version = versions.pop()
    if version < file_version(target / "vcruntime140.dll"):
        raise ValueError("The CRT must be at least as recent as the embedded Python CRT.")
    for file in files:
        shutil.copy2(file, target / file.name)
    print(f"Staged {len(files)} x64 Microsoft CRT DLLs ({'.'.join(map(str, version))})")


if __name__ == "__main__":
    stage(Path(sys.argv[1]), Path(sys.argv[2]))
