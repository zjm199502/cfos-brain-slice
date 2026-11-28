# CCFv3 registration benchmark (A海马3-4 style projects)

Run with the FosFolio app Python (needs itk-elastix):

    PY=/Applications/FosFolio脑片图册.app/Contents/Resources/python/bin/python3
    OUT=/tmp/regbench; mkdir -p $OUT
    $PY tools/reg_bench/prep.py $OUT /path/to/project/cfos_project.json   # thumbnails + giRAff CCFv3 affine (slow)
    $PY tools/reg_bench/layers.py $OUT                                   # DAPI-detected DG/CA layers
    $PY tools/reg_bench/run.py $OUT pipeline                             # outline-only (old behaviour)
    $PY tools/reg_bench/run.py $OUT e2e                                  # plane search + deep texture fit
    $PY tools/reg_bench/metrics.py $OUT ccf_affine,pipeline,e2e
    $PY tools/reg_bench/overlay.py $OUT pipeline,e2e                     # side-by-side PNGs

Primary metric: mean distance (µm) from atlas DG-sg to the DAPI-detected granule layer.
Sections are hard-coded as S01–S03.
