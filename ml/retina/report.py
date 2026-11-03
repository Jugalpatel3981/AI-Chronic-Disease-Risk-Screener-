"""Charts for the retina model card.

Usage (after train + export):
    python -m ml.retina.report

Writes to docs/models/retina/:
  roc.png        referable-DR ROC, cross-validation vs held-out test
  confusion.png  test-set confusion matrix of the 5 grades
  gradcam.png    attention maps from the deployed API on one test image per grade
"""

import onnxruntime  # noqa: F401  (load before torch: the other order segfaults on Windows)
import base64
import io
import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from ml.tabular import plots
from ml.tabular.config import DOCS_DIR, REPO_ROOT

from .data import GRADE_NAMES, load
from .train import WORK_DIR

OUT = DOCS_DIR / "retina"
SHORT = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def confusion_plot(cm: np.ndarray, path):
    rows = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + plots.SEQ_BLUE)
    ax.imshow(rows, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            dark = rows[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{rows[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if dark else plots.INK_2)
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(5), SHORT)
    ax.set(xlabel="Predicted grade", ylabel="Grade by ophthalmologists")
    ax.set_title("Test set: predicted vs true DR grade (row %)")
    for s in ax.spines.values():
        s.set_visible(False)
    plots._save(fig, path)


def gradcam_plot(test, path):
    sys.path.insert(0, str(REPO_ROOT))
    from backend.app.models.retina import RetinaService
    svc = RetinaService(REPO_ROOT / "backend" / "models")
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), layout="constrained")
    for g in range(5):
        row = test[test["grade"] == g].iloc[rng.integers((test["grade"] == g).sum())]
        rgb = cv2.cvtColor(cv2.imread(str(row["path"])), cv2.COLOR_BGR2RGB)
        out = svc.predict(rgb)
        heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(out["heatmap_png"].split(",")[-1]))))
        axes[0, g].imshow(np.asarray(Image.open(row["cached"])))
        axes[1, g].imshow(heat)
        axes[0, g].set_title(f"True: {SHORT[g]}", fontsize=10)
        axes[1, g].set_title(f"Predicted: {SHORT[out['grade']]}", fontsize=10, color=plots.INK_2,
                             fontweight="normal")
        for ax in axes[:, g]:
            ax.axis("off")
    fig.suptitle("Where the model looks: one held-out test image per grade", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    plots._save(fig, path)


def main() -> int:
    run = WORK_DIR / "efficientnet_b0"
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    dev, test = load()
    oof, ts = np.load(run / "oof.npy"), np.load(run / "test_score.npy")
    a, b = res["calibration"]["referable"]
    y_dev, y_test = (dev["grade"] >= 2).to_numpy(), (test["grade"] >= 2).to_numpy()
    p_dev, p_test = sigmoid(a * oof + b), sigmoid(a * ts + b)
    plots.roc_plot({"Cross-validation (1,600 images)": (y_dev, p_dev, roc_auc_score(y_dev, p_dev)),
                    "Held-out test (400 images)": (y_test, p_test, roc_auc_score(y_test, p_test))},
                   "Referable DR (moderate or worse)", OUT / "roc.png")
    confusion_plot(np.array(res["test_confusion"]), OUT / "confusion.png")
    gradcam_plot(test, OUT / "gradcam.png")
    assert len(GRADE_NAMES) == 5
    print(f"[report] wrote roc.png, confusion.png, gradcam.png to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
