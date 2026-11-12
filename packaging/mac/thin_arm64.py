"""把程序包里的 universal2（x86_64 + arm64）二进制文件只保留 arm64 部分，减小体积。
每个架构的部分各自带签名，单独取出后签名仍然有效。用法：python thin_arm64.py <目录>"""
import os
import struct
import sys

FAT_MAGIC, FAT_MAGIC_64 = 0xCAFEBABE, 0xCAFEBABF
CPU_ARM64 = 0x0100000C


def thin(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return 0
        magic, n = struct.unpack(">II", head)
        if magic not in (FAT_MAGIC, FAT_MAGIC_64) or not 1 < n < 10:
            return 0
        size = 20 if magic == FAT_MAGIC else 32
        for _ in range(n):
            ent = f.read(size)
            if magic == FAT_MAGIC:
                cpu, _sub, off, sz, _al = struct.unpack(">iiIII", ent)
            else:
                cpu, _sub, off, sz, _al, _r = struct.unpack(">iiQQII", ent)
            if cpu == CPU_ARM64:
                f.seek(off)
                data = f.read(sz)
                break
        else:
            return 0
    if data[:4] not in (b"\xcf\xfa\xed\xfe",):
        return 0
    old = os.path.getsize(path)
    mode = os.stat(path).st_mode
    with open(path + ".thin", "wb") as g:
        g.write(data)
    os.replace(path + ".thin", path)
    os.chmod(path, mode)
    return old - len(data)


saved = 0
for root, _dirs, files in os.walk(sys.argv[1]):
    for fn in files:
        p = os.path.join(root, fn)
        if not os.path.islink(p):
            saved += thin(p)
print(f"去掉 x86_64 部分，节省 {saved / 1e6:.0f} MB")
