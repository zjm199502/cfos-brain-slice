# CCFv3 registration benchmark (A海马3-4 style projects)

Run with the FosFolio app Python (needs itk-elastix):

    PY=/Applications/FosFolio脑片图册.app/Contents/Resources/python/bin/python3
    OUT=/tmp/regbench; mkdir -p $OUT
    $PY tools/reg_bench/prep.py $OUT /path/to/project/cfos_project.json   # thumbnails + giRAff CCFv3 affine (slow)
    $PY tools/reg_bench/layers.py $OUT                                   # DAPI-detected DG/CA layers
    $PY tools/reg_bench/run.py $OUT pipeline                             # outline-only (old behaviour)
    $PY tools/reg_bench/run.py $OUT e2e                                  # plane search + deep texture fit
    $PY tools/reg_bench/run.py $OUT series 0                             # e2e + joint AP by slice order (spacing µm, 0 = unknown)
    $PY tools/reg_bench/metrics.py $OUT ccf_affine,pipeline,e2e,series
    $PY tools/reg_bench/overlay.py $OUT pipeline,e2e                     # side-by-side PNGs

Deep-refine variants (2026-11-28): `run.py $OUT e2e:<texture>[+guard][+ridge][+search]` with texture
paint (default) / layers (cell layers painted bright) / nissl (Allen CCFv3 Nissl, install with nissl.py);
+guard rejects fits that move DG-sg or CA-sp away from the in-app DAPI ridges, +ridge adds the ridge metric,
+search also scores the plane search with that texture.  Results are tagged e.g. e2e_nissl_guard.

Metrics (metrics.py): DAPI-detected cell layers inside the hippocampus only (piriform, choroid plexus removed),
split into DG (granule layer -> atlas DG-sg) and CA (pyramidal layer -> atlas CA1-3 sp), plus atlas DG-sg ->
DAPI granule layer.  The DG/CA split is fixed per section from the affine fit.
Sections are hard-coded as S01–S03.
