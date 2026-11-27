"""Losslessly transport cpsam_v2, including its original PyTorch ZIP metadata."""
import argparse
import gzip
import hashlib
import os
from pathlib import Path
import tempfile


# This standalone tool runs before the application environment is configured.
# Keep the weight identity in sync with cfos.core.cellpose_setup.
MODEL_SIZE = 1_233_586_851
MODEL_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
MAGIC = b"FosFolio-cpsam-v2\x01"
BLOCK_SIZE = 1024 * 1024


def shuffle(block):
    return b"".join(block[lane::4] for lane in range(4))


def unshuffle(block):
    output = bytearray(len(block))
    start = 0
    for lane in range(4):
        end = start + (len(block) + 3 - lane) // 4
        output[lane::4] = block[start:end]
        start = end
    return output


def transfer(src, dst, packing):
    if packing:
        dst.write(MAGIC)
        stream = gzip.GzipFile(filename="", fileobj=dst, mode="wb", compresslevel=6, mtime=0)
    else:
        if src.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid compact Cellpose model header")
        stream = gzip.GzipFile(fileobj=src, mode="rb")

    digest = hashlib.sha256()
    with stream:
        reader, writer = (src, stream) if packing else (stream, dst)
        remaining = MODEL_SIZE
        while remaining:
            count = min(remaining, BLOCK_SIZE)
            block = reader.read(count)
            if len(block) != count:
                raise ValueError("Incomplete Cellpose model")
            original = block if packing else unshuffle(block)
            digest.update(original)
            writer.write(shuffle(original) if packing else original)
            remaining -= count
        if reader.read(1) or digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Cellpose model checksum or size mismatch")


def convert(src, dst, packing):
    if src.resolve() == dst.resolve():
        raise ValueError("Specify separate source and destination files")
    with src.open("rb") as source:
        temporary = tempfile.NamedTemporaryFile(dir=dst.parent, delete=False)
        temp_path = Path(temporary.name)
        try:
            with temporary as target:
                transfer(source, target, packing)
            os.replace(temp_path, dst)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination, args.action == "pack")
