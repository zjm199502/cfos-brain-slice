"""Count stored detections in scanner coordinates, optionally merging cross-channel IDs."""
import numpy as np
import pandas as pd
from matplotlib.path import Path

from .pipeline import inside_any
from .project import processed_section_names


def saved_coverage(image, frame):
    processed = processed_section_names(image, frame)
    return {s.name: [s.vertices()] for s in image.sections if s.enabled and s.name in processed}


def count_rois(frame, rois, coverage, channels, exclusions=()):
    """Accept section contours or preview/tile boxes (x, y, width, height) in scanner pixels."""
    if not rois:
        return pd.DataFrame()
    rows = []
    if frame is not None and len(frame):
        required = ["section", "channel", "x", "y"]
        if all(c in frame for c in required):
            columns = [*required, "merged_id"] if "merged_id" in frame else required
            valid = frame.section.isin(coverage) & frame.channel.isin([c for c, _ in channels])
            frame = frame.loc[valid, columns]
        else:
            frame = None
    xy = frame[["x", "y"]].to_numpy(float) if frame is not None and len(frame) else np.empty((0, 2))
    footprints = []
    for parts in coverage.values():
        for part in parts:
            vertices = np.asarray(part)
            if vertices.ndim == 1:
                x, y, width, height = vertices
                r, b = x + width, y + height
                vertices = np.array([[x, y], [r, y], [r, b], [x, b]])
            footprints.append(Path(np.vstack([vertices, vertices[0]]), closed=True))
    for roi in rois:
        vertices = np.asarray(roi["polygon"], dtype=float)
        if len(vertices) < 3:
            continue
        if np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - vertices[:, 1] * np.roll(vertices[:, 0], -1)) < 0:
            vertices = vertices[::-1]
        path = Path(np.vstack([vertices, vertices[:1]]), closed=True)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        covered = frame is not None and any(path.intersects_path(p, filled=True) for p in footprints)
        row = {"ROI": roi["name"], "识别范围": "已统计交集" if covered else "未覆盖"}
        row.update({f"C{c + 1} {name}": None for c, name in channels})
        row["合并细胞数"] = None
        if covered:
            candidates = np.flatnonzero(np.all((xy >= low) & (xy <= high), axis=1))
            pts = xy[candidates]
            keep = path.contains_points(pts, radius=1e-6)
            if exclusions and keep.any():
                keep &= ~inside_any(pts[:, 0], pts[:, 1], exclusions)
            cells = frame.iloc[candidates[keep]]
            counts = cells["channel"].value_counts() if len(cells) else {}
            row.update({f"C{c + 1} {name}": int(counts.get(c, 0)) for c, name in channels})
            if "merged_id" in cells:
                row["合并细胞数"] = len(cells[["section", "merged_id"]].drop_duplicates()) if cells.merged_id.notna().all() else None
        rows.append(row)
    return pd.DataFrame(rows)
