"""System display/compute adapters, independent of the installed inference providers."""
from __future__ import annotations

import ctypes as ct
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID


@dataclass(frozen=True)
class GpuAdapter:
    name: str
    vendor_id: int | None = None
    dedicated_bytes: int | None = None
    dml_index: int | None = None


class _AdapterDesc1(ct.Structure):
    _fields_ = [("name", ct.c_wchar * 128), ("vendor", ct.c_uint32),
                ("device", ct.c_uint32), ("subsystem", ct.c_uint32), ("revision", ct.c_uint32),
                ("video_memory", ct.c_size_t), ("system_memory", ct.c_size_t), ("shared_memory", ct.c_size_t),
                ("luid_low", ct.c_uint32), ("luid_high", ct.c_int32), ("flags", ct.c_uint32)]


def _com_call(obj, slot, *argtypes):
    vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(ct.c_int32, ct.c_void_p, *argtypes)(vtable[slot])


def _dxgi_adapters() -> tuple[GpuAdapter, ...]:
    """Use the same unfiltered DXGI ordinal that DirectML accepts as device_id."""
    # The only COM methods used below are IUnknown::Release, IDXGIFactory1::EnumAdapters1
    # and IDXGIAdapter1::GetDesc1, in their Windows SDK vtable order.
    iid = (ct.c_ubyte * 16).from_buffer_copy(UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    create = ct.WinDLL("dxgi.dll").CreateDXGIFactory1
    create.argtypes = (ct.c_void_p, ct.POINTER(ct.c_void_p))
    create.restype = ct.c_int32
    factory = ct.c_void_p()
    if create(iid, ct.byref(factory)) < 0:
        return ()
    adapters = []
    try:
        enum = _com_call(factory, 12, ct.c_uint32, ct.POINTER(ct.c_void_p))
        index = 0
        while True:
            adapter = ct.c_void_p()
            if enum(factory, index, ct.byref(adapter)) < 0:
                break
            try:
                desc = _AdapterDesc1()
                get_desc = _com_call(adapter, 10, ct.POINTER(_AdapterDesc1))
                if get_desc(adapter, ct.byref(desc)) >= 0 and desc.name and not desc.flags & 2:
                    adapters.append(GpuAdapter(desc.name.strip(), desc.vendor, desc.video_memory, index))
            finally:
                _com_call(adapter, 2)(adapter)
            index += 1
    finally:
        _com_call(factory, 2)(factory)
    return tuple(adapters)


def _output(args: list[str]) -> str:
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.decode("utf-8-sig", errors="replace") if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _models(names) -> tuple[GpuAdapter, ...]:
    return tuple(GpuAdapter(name.strip()) for name in names if isinstance(name, str) and name.strip())


@lru_cache(maxsize=1)
def gpu_adapters() -> tuple[GpuAdapter, ...]:
    """Detect OS-reported hardware once per process; call off the UI thread on first use."""
    if sys.platform == "win32":
        try:
            adapters = _dxgi_adapters()
        except (OSError, AttributeError):
            adapters = ()
        if adapters:
            return adapters
        cmd = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
               "(Get-CimInstance Win32_VideoController -ErrorAction Stop).Name")
        return _models(_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd]).splitlines())
    if sys.platform == "darwin":
        output = _output(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
        try:
            entries = json.loads(output)["SPDisplaysDataType"]
        except (ValueError, KeyError, TypeError):
            return ()
        if not isinstance(entries, list):
            return ()
        return _models(e.get("sppci_model") or e.get("_name") for e in entries if isinstance(e, dict))
    if sys.platform.startswith("linux"):
        output = _output(["lspci", "-D", "-nn"])
        names = re.findall(r"^\S+ [^\n]*?\[(?:0300|0302|0380)\]: ([^\n]+)", output, re.MULTILINE)
        if names:
            return _models(names)
        return _models(_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines())
    return ()


def gpu_names() -> tuple[str, ...]:
    return tuple(adapter.name for adapter in gpu_adapters())
