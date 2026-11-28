"""python nissl.py ara_nissl_25.nrrd  -> cfos/core/ccf_data/allen_ccfv3_nissl_25um.npz (uint8, AP/DV/ML 25 µm)

Allen CCFv3-aligned Nissl volume, used by deep_refine when TEXTURE = "nissl".  Download (≈130 MB):
https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_25.nrrd
Needs `pip install pynrrd`.  Not bundled with the app yet (≈31 MB compressed)."""
import sys, numpy as np, nrrd
from pathlib import Path
v, _ = nrrd.read(sys.argv[1])
assert v.shape == (528, 320, 456), v.shape
u = np.clip(v / np.percentile(v[v > 0], 99.8) * 255, 0, 255).astype(np.uint8)
out = Path(__file__).resolve().parents[2] / "cfos/core/ccf_data/allen_ccfv3_nissl_25um.npz"
np.savez_compressed(out, nissl=u)
print(out, u.shape)
