"""Hand-crafted colour features of a nail crop (and optionally adjacent skin).

Pale nail beds are a classic clinical sign of anaemia: with less haemoglobin
the nail bed looks less red. These features measure that redness in several
colour spaces. Only numpy + OpenCV are needed, so the API can reuse this file.
"""

import cv2
import numpy as np


def _center(crop: np.ndarray, keep: float = 0.6) -> np.ndarray:
    """Central region of the crop: avoids the cuticle, nail edge and background."""
    h, w = crop.shape[:2]
    dh, dw = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    return crop[dh:h - dh, dw:w - dw] if h - 2 * dh > 2 and w - 2 * dw > 2 else crop


def crop_features(crop_rgb: np.ndarray) -> dict:
    c = _center(crop_rgb).astype(np.uint8)
    rgb = c.reshape(-1, 3).astype(np.float64)
    lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)
    total = rgb.sum(axis=1) + 1e-6
    return {
        "r": rgb[:, 0].mean(), "g": rgb[:, 1].mean(), "b": rgb[:, 2].mean(),
        "r_chroma": (rgb[:, 0] / total).mean(),       # share of red, robust to brightness
        "g_chroma": (rgb[:, 1] / total).mean(),
        "L": lab[:, 0].mean(), "a": lab[:, 1].mean(), "b_lab": lab[:, 2].mean(),
        "sat": hsv[:, 1].mean(), "val": hsv[:, 2].mean(),
        "r_minus_g": (rgb[:, 0] - rgb[:, 1]).mean(),
    }


def mean_features(crops: list[np.ndarray]) -> dict:
    feats = [crop_features(c) for c in crops]
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}


def patient_features(nails: list[np.ndarray], skins: list[np.ndarray] | None = None) -> dict:
    """Nail colour, plus (if skin crops are given) skin colour and nail-minus-skin
    differences, which cancel out lighting and camera colour shifts."""
    nail = mean_features(nails)
    out = {f"nail_{k}": v for k, v in nail.items()}
    if skins:
        skin = mean_features(skins)
        out.update({f"skin_{k}": v for k, v in skin.items()})
        out.update({f"diff_{k}": nail[k] - skin[k] for k in nail})
    return out
