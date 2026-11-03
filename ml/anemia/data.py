"""Load the fingernail haemoglobin dataset (Yakimov et al., Scientific Data 2024).

250 adults (18-95 years) photographed with a fixed camera; haemoglobin (Hb)
measured from venous blood. Each photo has bounding boxes for three nails and
three adjacent skin patches. License: CC0.
Source: https://doi.org/10.6084/m9.figshare.c.6760179
"""

import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from ml.nhanes.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw" / "anemia_yakimov"
URL = "https://ndownloader.figshare.com/files/49504041"


def download() -> None:
    if (RAW_DIR / "metadata.csv").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / "data.zip"
    print(f"[anemia] downloading dataset (92 MB) -> {zpath}")
    req = urllib.request.Request(URL, headers={"User-Agent": "research-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW_DIR)


def _boxes(cell: str) -> list[tuple[int, int, int, int]]:
    """Boxes are stored as [y1, x1, y2, x2]; return PIL order (x1, y1, x2, y2)."""
    return [(x1, y1, x2, y2) for y1, x1, y2, x2 in json.loads(cell)]


def load() -> pd.DataFrame:
    """One row per patient: hb (g/L), session id, nail crops and skin crops (RGB arrays)."""
    download()
    meta = pd.read_csv(RAW_DIR / "metadata.csv")
    rows = []
    for r in meta.itertuples():
        img = Image.open(RAW_DIR / "photo" / f"{r.PATIENT_ID}.jpg").convert("RGB")
        rows.append({
            "patient_id": int(r.PATIENT_ID),
            "session": r.MEASUREMENT_DATE,  # anonymised session hash; shared lighting
            "hb": float(r.HB_LEVEL_GperL),
            "nails": [np.asarray(img.crop(b)) for b in _boxes(r.NAIL_BOUNDING_BOXES)],
            "skins": [np.asarray(img.crop(b)) for b in _boxes(r.SKIN_BOUNDING_BOXES)],
        })
    return pd.DataFrame(rows)
