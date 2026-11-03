"""Diabetic-retinopathy grading from fundus photos (ONNX ensemble)."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from PIL import Image

from ml.retina.preprocess import CACHE_SIZE, is_probably_fundus, prepare

from ..images import overlay_heatmap, to_png_base64


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


class RetinaService:
    """Expects models/retina/retina.onnx (image -> score, cam) and meta.json."""

    def __init__(self, model_dir: Path):
        d = model_dir / "retina"
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(d / self.meta["onnx"]), providers=["CPUExecutionProvider"])
        self.size = int(self.meta["input_size"])
        self.mean = np.asarray(self.meta["mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["std"], dtype=np.float32)
        self.cuts = np.asarray(self.meta["cutpoints"])

    def _tensor(self, rgb_512: np.ndarray) -> np.ndarray:
        # Same resize as training: torchvision Resize on a PIL image = PIL bilinear.
        img = np.asarray(Image.fromarray(rgb_512).resize((self.size, self.size), Image.BILINEAR))
        x = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return np.stack([x, x[:, :, ::-1], x[:, ::-1, :]]).astype(np.float32)  # test-time flips

    def predict(self, rgb: np.ndarray) -> dict:
        if not is_probably_fundus(rgb):
            raise HTTPException(422, "This doesn't look like a retina (fundus) photo. Please upload a "
                                     "fundus image taken with a retinal camera.")
        fundus = prepare(rgb, CACHE_SIZE)
        score_views, cam_views = self.session.run(None, {"image": self._tensor(fundus)})
        score = float(np.mean(score_views))
        # Undo the flips before averaging the attention maps.
        cam = (cam_views[0] + cam_views[1][:, ::-1] + cam_views[2][::-1, :]) / 3

        grade = int(np.digitize(score, self.cuts))
        a, b = self.meta["calibration"]["referable"]
        p_ref = _sigmoid(a * score + b)
        a, b = self.meta["calibration"]["any_dr"]
        p_any = _sigmoid(a * score + b)
        thr = float(self.meta["referable_threshold"])
        referable = p_ref >= thr
        level = "high" if grade >= 3 else ("moderate" if referable or grade == 2 else "low")

        yy, xx = np.mgrid[:fundus.shape[0], :fundus.shape[1]]
        r = fundus.shape[0] / 2
        disc = (((yy - r) ** 2 + (xx - r) ** 2) <= (0.98 * r) ** 2).astype(np.float32)
        heatmap = to_png_base64(overlay_heatmap(fundus, cam, alpha=0.5, mask=disc))
        return {
            "grade": grade, "grade_name": self.meta["grade_names"][grade],
            "probability_referable": p_ref, "probability_any_dr": p_any,
            "risk_level": level, "referable": bool(referable), "heatmap_png": heatmap,
            "note": ("Referable diabetic retinopathy (moderate or worse) should be reviewed by an eye "
                     "specialist." if referable else
                     "No referable diabetic retinopathy detected. Regular eye checks are still "
                     "recommended for people with diabetes."),
        }
