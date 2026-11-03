"""Nail/skin crop preprocessing shared by training and the API (numpy + OpenCV)."""

import cv2
import numpy as np

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(crop_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 crop -> float32 array of shape (3, 224, 224), ImageNet-normalised."""
    img = cv2.resize(crop_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return img.transpose(2, 0, 1)


def batch(crops: list[np.ndarray]) -> np.ndarray:
    return np.stack([to_tensor(c) for c in crops]).astype(np.float32)
