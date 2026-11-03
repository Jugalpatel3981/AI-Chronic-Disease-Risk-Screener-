"""Anaemia screening from fingernail photos (ONNX backbone + ridge head)."""

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from scipy.stats import norm

from ml.anemia.preprocess import batch

from ..images import overlay_heatmap, to_png_base64

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels
MIN_BOX = 12
DETECTION_SCORE_MIN = 0.5
TRAIN_LONG_SIDE = 800  # every training photo is 800 x 600


def crop(rgb: np.ndarray, box: Box) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
        raise HTTPException(400, f"Each box must be at least {MIN_BOX}x{MIN_BOX} pixels inside the image.")
    return rgb[y1:y2, x1:x2]


class AnemiaService:
    def __init__(self, model_dir: Path):
        d = model_dir / "anemia"
        self.head = json.loads((d / "head.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(d / self.head["onnx"]), providers=["CPUExecutionProvider"])
        w = np.asarray(self.head["weights"], dtype=np.float32)
        half = len(w) // 2
        self.w_nail, self.w_diff = w[:half], w[half:]
        self.bias = float(self.head["bias"])
        self.sigma = float(self.head["residual_sd_gl"])
        self.thresholds = self.head["who_threshold_gl"]
        det = d / "nail_detector.onnx"
        self.detector = (ort.InferenceSession(str(det), providers=["CPUExecutionProvider"])
                         if det.exists() else None)

    def _run_detector(self, rgb: np.ndarray) -> tuple[list[Box], list[Box], float]:
        image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        try:
            boxes, labels, scores = self.detector.run(None, {"image": image})
        except ort.capi.onnxruntime_pybind11_state.Fail:
            # The exported Faster R-CNN graph cannot represent "zero candidate boxes";
            # that only happens when there is nothing nail-like in the photo.
            return [], [], 0.0

        def top(label: int) -> tuple[list[Box], float]:
            m = (labels == label) & (scores >= DETECTION_SCORE_MIN)
            order = np.argsort(-scores[m])[:3]
            return [tuple(float(v) for v in b) for b in boxes[m][order]], float(scores[m][order].sum())

        (nails, s_n), (skins, s_s) = top(1), top(2)
        return nails, skins, s_n + s_s

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, list[Box], list[Box]]:
        """Find up to 3 nails and 3 skin patches with the ONNX detector.

        Photos are first scaled to the training resolution (long side 800 px): colour
        embeddings of a nail crop depend on how many pixels it covers. Training photos
        are all landscape, so a portrait phone photo is also tried rotated 90 degrees
        both ways and the most confident orientation is kept. Returns the image the
        boxes refer to together with the boxes."""
        if self.detector is None:
            raise HTTPException(503, "Automatic nail detection is not available on this server.")
        scale = TRAIN_LONG_SIDE / max(rgb.shape[:2])
        if scale != 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        best = None
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            img = rgb if rot is None else np.ascontiguousarray(cv2.rotate(rgb, rot))
            nails, skins, score = self._run_detector(img)
            if nails and skins and (best is None or score > best[3]):
                best = (img, nails, skins, score)
            if best is not None and rot is None and len(nails) == 3:
                break  # upright landscape photo with three nails: no need to try rotations
        if best is None:
            raise HTTPException(422, "We couldn't find your nails in this photo. Please take a clear photo of "
                                     "your fingers resting flat on white paper, in daylight.")
        return best[0], best[1], best[2]

    def _feature_maps(self, crops: list[np.ndarray]) -> np.ndarray:
        return self.session.run(None, {"image": batch(crops)})[0]  # (n, 1280, 7, 7)

    def predict(self, rgb: np.ndarray, nail_boxes: list[Box] | None, skin_boxes: list[Box] | None,
                sex: str | None):
        if nail_boxes is None or skin_boxes is None:
            rgb, nail_boxes, skin_boxes = self.detect(rgb)
        if not 1 <= len(nail_boxes) <= 3 or not 1 <= len(skin_boxes) <= 3:
            raise HTTPException(400, "Mark between 1 and 3 nails and between 1 and 3 skin patches.")
        nails = [crop(rgb, b) for b in nail_boxes]
        skins = [crop(rgb, b) for b in skin_boxes]
        nail_maps = self._feature_maps(nails)
        nail_emb = nail_maps.mean(axis=(2, 3)).mean(0)
        skin_emb = self._feature_maps(skins).mean(axis=(2, 3)).mean(0)
        # Hb = w_nail . n + w_diff . (n - s) + b
        hb_gl = float(self.w_nail @ nail_emb + self.w_diff @ (nail_emb - skin_emb) + self.bias)

        threshold = self.thresholds.get(sex or "female")
        probability = float(norm.cdf((threshold - hb_gl) / self.sigma))
        level = "low" if probability < 0.3 else ("high" if probability >= 0.6 else "moderate")

        # Pallor map: spatial contribution of each nail location to a LOWER Hb estimate.
        contrib = np.tensordot(self.w_nail + self.w_diff, nail_maps[0], axes=(0, 0))  # (7, 7)
        heatmap = to_png_base64(overlay_heatmap(nails[0], -contrib, alpha=0.5), max_side=256)

        lo, hi = hb_gl - 1.96 * self.sigma, hb_gl + 1.96 * self.sigma
        return {
            "hemoglobin_g_dl": round(hb_gl / 10, 1),
            "hemoglobin_range_g_dl": (round(max(lo, 0) / 10, 1), round(hi / 10, 1)),
            "probability": probability, "risk_level": level,
            "threshold_g_dl": threshold / 10, "sex_used": sex, "nails_used": len(nails),
            "heatmap_png": heatmap,
            "note": ("Estimated from nail colour relative to nearby skin. Accuracy is best with 3 nails, "
                     "daylight and a white background, without nail polish."
                     + ("" if sex else " Sex not given: the 12 g/dL threshold for women was used.")),
        }
