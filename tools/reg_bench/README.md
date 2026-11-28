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

Defaults since 2026-11-28 round 3: TEXTURE=nissl (falls back to paint without the Nissl file), PREPROC=clahe,
LAYER_PULL on (CA pyramidal layer pulled onto the DAPI ridges, DG pinned).  Mean of S01-S03 (DG / CA / overall
P90, µm): old 56 / 74 / 217, paint+clahe+pull 57 / 45 / 181, nissl+clahe+pull 33 / 41 / 93.
Any `e2e:<...>` method switches LAYER_PULL off unless +pull is given; +pre=flat|flat150|clahe|zscore picks PREPROC.

Deep-refine variants (2026-11-28): `run.py $OUT e2e:<texture>[+guard][+ridge][+search][+pull][+pre=X]` with texture
paint (default) / layers (cell layers painted bright) / nissl (Allen CCFv3 Nissl, install with nissl.py);
+guard rejects fits that move DG-sg or CA-sp away from the in-app DAPI ridges, +ridge adds the ridge metric,
+search also scores the plane search with that texture.  Results are tagged e.g. e2e_nissl_guard.

Metrics (metrics.py): DAPI-detected cell layers inside the hippocampus only (piriform, choroid plexus removed),
split into DG (granule layer -> atlas DG-sg) and CA (pyramidal layer -> atlas CA1-3 sp), plus atlas DG-sg ->
DAPI granule layer.  The DG/CA split is fixed per section from the affine fit.
Sections are hard-coded as S01–S03.

## 'apl'：按海马细胞层选切面前后位置（实验，默认关闭）
`python run.py S apl` 跑 autoreg 里的 AP_BY_LAYERS 流程：每个候选前后位置重贴轮廓，按 DAPI 细胞层脊线与图谱细胞层的
双向平均距离打分，同一动物按切片顺序动态规划选位置，然后只微调倾斜。2026-11-28 在 S01–S03 上：评分选了
S01 +150、S02 −100、S03 −300 µm，深部贴合后合计 P90 129 → 179 µm（更差）。在固定位置 −300…+200 µm 上分别做完整
贴合，结果随前后位置剧烈跳动（例如 S03 −100 µm 好、−200 µm 很差），重贴轮廓阶段的评分预测不了深部贴合后的结果。
