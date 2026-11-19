"""读图时多线程解码图块 + 已解码图块的小缓存（v0.8.3）。

qptiff / ome.tif 和江丰 .kfbf 都是一块块（256 或 512 像素见方）压缩成 JPEG 存的。
原来读一块区域时逐个图块串行解码，只用一个核；JPEG 解码（imagecodecs）会释放 Python 的全局锁，
所以多开几个线程就能同时解码，结果与单线程逐像素完全相同（同一个解码器、同一份数据）。

- 线程数：最多 4 个，并且不超过主进程的线程预算（cpu.main_threads()，属于“一半核数”的预算之内：
  读图和神经网络在主进程里是先后进行的，不会同时占用）。可用环境变量 CFOS_DECODE_THREADS 改（1 = 不并行）。
- 缓存：识别时相邻分块有重叠（约 96 像素），重叠处的图块原来要解码两遍；
  每张图片保留最近解码过的图块，上限 CACHE_MB（默认 128 MB），超过后丢掉最久没用的。
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CACHE_MB = 128
MAX_THREADS = 4

_pool = None
_pool_n = 0
_pool_lock = threading.Lock()


def decode_threads() -> int:
    env = os.environ.get("CFOS_DECODE_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        from .cpu import main_threads
        n = main_threads()
    except Exception:  # noqa: BLE001
        n = 1
    return max(1, min(MAX_THREADS, n))


def _executor(n: int):
    global _pool, _pool_n
    with _pool_lock:
        if _pool is None or _pool_n != n:
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="cfos-decode")
            _pool_n = n
        return _pool


def pmap(fn, items: list) -> list:
    """按顺序返回 fn(item) 的结果；图块少或只允许 1 个线程时直接串行。"""
    n = decode_threads()
    if n < 2 or len(items) < 2 or threading.current_thread().name.startswith("cfos-decode"):
        return [fn(x) for x in items]
    return list(_executor(n).map(fn, items))


class TileCache:
    """按字节数限制大小的 LRU 缓存（线程安全）。存进去的数组设为只读，防止被意外改写。"""

    def __init__(self, max_mb: float = CACHE_MB):
        self.max_bytes = int(max_mb * 2 ** 20)
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            a = self._d.get(key)
            if a is not None:
                self._d.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return a

    def put(self, key, a: np.ndarray):
        if a is None or self.max_bytes <= 0 or a.nbytes > self.max_bytes:
            return
        a.setflags(write=False)
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._d[key] = a
            self._bytes += a.nbytes
            while self._bytes > self.max_bytes and self._d:
                _, x = self._d.popitem(last=False)
                self._bytes -= x.nbytes

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes


# ---------------------------------------------------------------- TIFF（qptiff / ome.tif）
class TiledTiffLevel:
    """一个金字塔层的所有通道（每个通道一页，图块存储、单样本）。read() 并行解码所需图块。"""

    def __init__(self, path, pages, level: int, cache: TileCache, fh_lock: threading.Lock, fh):
        self.pages = [p.aspage() if hasattr(p, "aspage") else p for p in pages]
        p0 = self.pages[0]
        self.level = level
        self.tw, self.tl = int(p0.tilewidth), int(p0.tilelength)
        self.H, self.W = int(p0.imagelength), int(p0.imagewidth)
        self.nx = -(-self.W // self.tw)
        self.dtype = np.dtype(p0.dtype)
        self.cache, self._lock, self._fh = cache, fh_lock, fh
        self._decoders = [p.decode for p in self.pages]
        self._jt = [p.jpegtables for p in self.pages]

    @staticmethod
    def supported(pages) -> bool:
        try:
            ps = [p.aspage() if hasattr(p, "aspage") else p for p in pages]
            p0 = ps[0]
            # Some QPTIFF pyramid levels omit the final partial tile row.
            # Use tifffile's chunk reader for these levels.
            if not p0.is_tiled:
                return False
            expected = (-(-p0.imagewidth // p0.tilewidth)
                        * -(-p0.imagelength // p0.tilelength))
            return all(p.is_tiled and int(p.samplesperpixel) == 1 and int(p.imagedepth) == 1
                       and len(p.dataoffsets) == len(p.databytecounts) == expected
                       and int(p.tiledepth) == 1 and p.dtype is not None
                       and (p.imagelength, p.imagewidth, p.tilelength, p.tilewidth)
                       == (p0.imagelength, p0.imagewidth, p0.tilelength, p0.tilewidth) for p in ps)
        except Exception:  # noqa: BLE001
            return False

    def _tile(self, job):
        c, ty, tx = job
        key = (self.level, c, ty, tx)
        a = self.cache.get(key)
        if a is not None:
            return a
        p = self.pages[c]
        i = ty * self.nx + tx
        off, cnt = int(p.dataoffsets[i]), int(p.databytecounts[i])
        th, tw = min(self.tl, self.H - ty * self.tl), min(self.tw, self.W - tx * self.tw)
        if cnt <= 0 or off <= 0:
            a = np.zeros((th, tw), self.dtype)          # 没有存的图块（tifffile 也按 0 填充）
        else:
            with self._lock:
                self._fh.seek(off)
                data = self._fh.read(cnt)
            seg, _, _ = self._decoders[c](data, i, jpegtables=self._jt[c])
            if seg is None:
                a = np.zeros((th, tw), self.dtype)
            else:
                seg = np.asarray(seg)
                a = seg.reshape(seg.shape[-3], seg.shape[-2])[:th, :tw]
                if a.dtype != self.dtype:
                    a = a.astype(self.dtype)
                a = np.ascontiguousarray(a)
        self.cache.put(key, a)
        return a

    def read(self, chs, y0, y1, x0, x1) -> np.ndarray:
        h, w = max(0, y1 - y0), max(0, x1 - x0)
        out = np.zeros((len(chs), h, w), self.dtype)
        if h == 0 or w == 0:
            return out
        tys = range(y0 // self.tl, (y1 - 1) // self.tl + 1)
        txs = range(x0 // self.tw, (x1 - 1) // self.tw + 1)
        ucs = list(dict.fromkeys(chs))
        jobs = [(c, ty, tx) for c in ucs for ty in tys for tx in txs]
        tiles = pmap(self._tile, jobs)
        ci = {c: [k for k, cc in enumerate(chs) if cc == c] for c in ucs}
        for (c, ty, tx), a in zip(jobs, tiles):
            gy, gx = ty * self.tl, tx * self.tw
            sy0, sy1 = max(y0, gy), min(y1, gy + a.shape[0])
            sx0, sx1 = max(x0, gx), min(x1, gx + a.shape[1])
            if sy1 > sy0 and sx1 > sx0:
                for k in ci[c]:
                    out[k, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
        return out
