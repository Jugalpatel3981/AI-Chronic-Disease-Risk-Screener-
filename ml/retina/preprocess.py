"""Fundus image preprocessing shared by training and the API (numpy + OpenCV only).

Steps: crop the black border around the circular fundus, pad to a square,
resize. Keeping one implementation guarantees the API sees images exactly the
way the model was trained on them.
"""

import cv2
import numpy as np

CACHE_SIZE = 512   # images are cached at this size before training
INPUT_SIZE = 384   # network input size


def crop_fundus(rgb: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Crop to the bounding box of the bright (non-black) fundus region."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if mask.sum() < 0.05 * mask.size:  # nearly black image: leave it alone
        return rgb
    ys, xs = np.where(mask)
    return rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pad_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=rgb.dtype)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = rgb
    return out


def prepare(rgb: np.ndarray, size: int = CACHE_SIZE) -> np.ndarray:
    """RGB uint8 image of any size -> cropped, square, resized RGB uint8."""
    sq = pad_square(crop_fundus(rgb))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def is_probably_fundus(rgb: np.ndarray) -> bool:
    """Cheap sanity check used by the API: fundus photos are a bright, reddish
    disc on a dark background. Rejects most selfies, documents and blank images."""
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = small.mean(axis=2)
    corners = np.concatenate([gray[:12, :12].ravel(), gray[:12, -12:].ravel(),
                              gray[-12:, :12].ravel(), gray[-12:, -12:].ravel()])
    centre = small[44:84, 44:84].reshape(-1, 3).mean(axis=0)
    dark_corners = corners.mean() < 40
    reddish_centre = centre[0] > centre[2] * 1.2 and centre[0] > 40
    return bool(dark_corners and reddish_centre)
