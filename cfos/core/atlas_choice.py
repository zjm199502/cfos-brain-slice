"""Keep atlas-specific reporting lookups scoped to the current project and worker."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

CCF = "CCFv3"
STAM = "STAM"
_report_atlas = ContextVar("fosfolio_report_atlas", default=CCF)


def current():
    return _report_atlas.get()


@contextmanager
def use_atlas(name):
    if name not in (CCF, STAM):
        raise ValueError(f"未知图谱：{name}")
    token = _report_atlas.set(name)
    try:
        yield
    finally:
        _report_atlas.reset(token)


def project_atlas(fn):
    @wraps(fn)
    def scoped(project, *args, **kwargs):
        with use_atlas(project.registration_atlas):
            return fn(project, *args, **kwargs)
    return scoped


def structure_id():
    return "STAM 结构 ID" if current() == STAM else "Allen ID"


def has_registration(record):
    return bool(record and (record.get("plane_native_mm") or record.get("plane")))
