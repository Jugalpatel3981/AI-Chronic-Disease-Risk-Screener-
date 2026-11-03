"""DeepDRiD regular-fundus data (Liu et al., Patterns 2022; CC BY-SA 4.0).

2,000 fundus images from 500 patients, graded 0-4 on the international DR
scale by ophthalmologists:
  - training + validation (1,600 images, 400 patients): used for 5-fold CV
  - online-challenge evaluation (400 images, 100 other patients): held-out test

Raw data: git clone https://github.com/deepdrdoc/DeepDRiD  into  RETINA_DATA_DIR
(default D:/screener-data, override with the RETINA_DATA_DIR environment variable).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from .preprocess import prepare

DATA_DIR = Path(os.environ.get("RETINA_DATA_DIR", "D:/screener-data"))
ROOT = DATA_DIR / "DeepDRiD" / "regular_fundus_images"
CACHE = DATA_DIR / "cache" / "deepdrid_512"

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def _dev_split(name: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / name / f"{name}.csv")
    src = pd.read_csv(ROOT / name / f"{name.replace('regular-fundus', 'regular-fundus-source')}.csv")
    d = d.merge(src[["image_id", "Source"]], on="image_id", how="left")
    d["grade"] = d["left_eye_DR_Level"].fillna(d["right_eye_DR_Level"]).astype(int)
    # CSV paths look like \regular-fundus-training\1\1_l1.jpg; files live under Images/.
    d["path"] = d.apply(lambda r: ROOT / name / "Images" / str(r.patient_id) / f"{r.image_id}.jpg",
                        axis=1)
    return d[["patient_id", "image_id", "grade", "path", "Source"]]


def _test_split() -> pd.DataFrame:
    folder = ROOT / "Online-Challenge1&2-Evaluation"
    lab = pd.read_excel(folder / "Challenge1_labels.xlsx")
    lab["patient_id"] = lab["image_id"].str.split("_").str[0].astype(int)
    lab["grade"] = lab["DR_Levels"].astype(int)
    lab["path"] = lab.apply(lambda r: folder / "Images" / str(r.patient_id) / f"{r.image_id}.jpg", axis=1)
    lab["Source"] = "challenge-evaluation"
    return lab[["patient_id", "image_id", "grade", "path", "Source"]]


def _cache_one(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    bgr = cv2.imread(str(src))
    if bgr is None:
        raise FileNotFoundError(src)
    rgb = prepare(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(dst), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (development, test) tables with a `cached` path to 512px images."""
    if not ROOT.exists():
        raise FileNotFoundError(
            f"{ROOT} not found. Run: git clone https://github.com/deepdrdoc/DeepDRiD {DATA_DIR / 'DeepDRiD'}")
    dev = pd.concat([_dev_split("regular-fundus-training"),
                     _dev_split("regular-fundus-validation")], ignore_index=True)
    test = _test_split()
    assert not set(dev.patient_id) & set(test.patient_id), "patients overlap between dev and test"
    CACHE.mkdir(parents=True, exist_ok=True)
    for df in (dev, test):
        df["cached"] = df["image_id"].map(lambda i: CACHE / f"{i}.jpg")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(_cache_one, df["path"], df["cached"]))
    return dev, test
