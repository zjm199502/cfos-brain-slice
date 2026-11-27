"""Bundled native STAM volumes. The label lookup is recorded in structures.csv."""
from functools import lru_cache
from pathlib import Path
import numpy as np

DATA = Path(__file__).with_name("stam_data")


@lru_cache(maxsize=1)
def volumes():
    with np.load(DATA / "stam.npz", allow_pickle=False) as z:
        return z["reference"], z["annotation"]
