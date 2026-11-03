"""Image upload validation, decoding and heatmap rendering."""

import base64

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from .config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB

MIN_SIDE = 64


async def read_image(upload: UploadFile) -> np.ndarray:
    """Validated upload -> RGB uint8 array. Raises 400/413/415 with a clear message."""
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported file type '{upload.content_type}'. Use JPEG, PNG or WebP.")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB:g} MB.")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "The file could not be read as an image.")
    if min(bgr.shape[:2]) < MIN_SIDE:
        raise HTTPException(400, f"Image is too small; it must be at least {MIN_SIDE} pixels on each side.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Blend a (h, w) activation map over an RGB image. Higher = more attention."""
    cam = cam.astype(np.float32)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=max(rgb.shape[:2]) / 60)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    weight = alpha * cam[..., None]
    if mask is not None:
        weight = weight * mask[..., None]
    out = rgb.astype(np.float32) * (1 - weight) + heat * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def to_png_base64(rgb: np.ndarray, max_side: int = 512) -> str:
    scale = max_side / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
