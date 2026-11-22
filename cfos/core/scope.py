"""Resolve an explicit image/section selection without changing the saved project."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields
import hashlib
import json

from .project import ImageEntry, Project

Selection = Mapping[str, Sequence[str] | None]


def selected_images(project: Project, selection: Selection | None):
    if selection is None:
        return list(project.images)
    return [im for im in project.images if im.key in selection
            and (selection[im.key] is None or selection[im.key])]


def selected_sections(im: ImageEntry, selection: Selection | None):
    names = None if selection is None else selection.get(im.key, ())
    return [s for s in im.sections if s.enabled and (names is None or s.name in names)]


class _AnalysisView(Project):
    """Project metadata snapshot for reporting; result files belong to the original project."""

    def __post_init__(self):
        # Creating a report must not change the process-wide default Z projection.
        pass

    def load_cells(self, image):
        names = self.scope_sections.get(image.key)
        if names is None:
            return None
        df = super().load_cells(image)
        if df is None or "section" not in df:
            return df
        result = df[df["section"].isin(names)]
        for key in ("processed_sections", "carried_forward_sections"):
            if key in result.attrs:
                result.attrs[key] = [name for name in result.attrs[key] if name in names]
        for key in ("section_models", "section_devices", "section_settings"):
            if key in result.attrs:
                result.attrs[key] = {name: value for name, value in result.attrs[key].items() if name in names}
        return result

    def save(self):
        raise RuntimeError("不能用筛选后的项目覆盖原项目。")

    def save_cells(self, image, df):
        raise RuntimeError("不能用筛选后的细胞覆盖全图结果。")

    def save_part(self, image, section, signature, df):
        raise RuntimeError("请通过原项目保存识别结果。")


def scoped_project(project: Project, selection: Selection | None):
    """Take a read-only analysis snapshot, retaining original stable image/section keys."""
    if selection is None:
        return project
    data = {f.name: deepcopy(getattr(project, f.name)) for f in fields(Project)
            if f.name not in {"images", "accuracy", "samples", "behavior"}}
    view = _AnalysisView(**data)
    keys = set()
    regions = {}
    view.scope_sections = {}
    for im in selected_images(project, selection):
        chosen = selected_sections(im, selection)
        names = {s.name for s in chosen}
        if selection[im.key] is not None and not names:
            continue
        clone = deepcopy(im)
        clone.sections = [s for s in clone.sections if s.name in names]
        clone.field_reviews = [row for row in clone.field_reviews if row.get("脑片") in names]
        if selection[im.key] is not None and clone.figure_rois:
            from .figure import section_for
            clone.figure_rois = [r for r in clone.figure_rois
                                 if (s := section_for(im, r["x"] + r["w"] / 2, r["y"] + r["h"] / 2))
                                 is not None and s.name in names]
        if selection[im.key] is not None and names != {s.name for s in im.sections}:
            # Hand-drawn ROIs span the image. Their metrics must use the same
            # section bounds as the counts when a report covers only part of it.
            regions[im.key] = [s.bbox0 for s in chosen]
        view.images.append(clone)
        view.scope_sections[im.key] = names
        keys.update((im.key, name) for name in names)
    view.scope_boxes = regions
    ids = sorted((im.key, sorted(s.name for s in im.sections)) for im in view.images)
    view.scope_tag = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()[:12]
    view.accuracy = [deepcopy(r) for r in project.accuracy if (r.get("图片"), r.get("脑片")) in keys]
    sample_keys = {f"{im}/{sec}" for im, sec in keys}
    view.samples = {k: deepcopy(v) for k, v in project.samples.items() if k in sample_keys}
    animals = {s.get("animal") for s in view.samples.values()}
    view.behavior = {k: v for k, v in project.behavior.items() if k in animals}
    return view
