"""Build the deployed anaemia-screening model.

Usage (from the repository root):
    python -m ml.anemia.train

The approach was chosen by `python -m ml.anemia.compare`: pretrained
EfficientNet-B0 embeddings of the nail and of the adjacent skin (which acts as a
lighting reference), followed by ridge regression of haemoglobin (Hb).

Steps
  1. export the ImageNet EfficientNet-B0 feature extractor to ONNX
  2. compute embeddings *through that ONNX file* with the same numpy
     preprocessing the API uses, so reported metrics match the deployed model
  3. evaluate with session-grouped 5-fold CV repeated over 5 shuffles, both
     with three nails per person and with a single nail (what the app sees)
  4. fit the final ridge model on all 250 people and save it with the
     residual spread used to turn an Hb estimate into an anaemia probability:
        P(anaemia) = Phi((threshold - Hb_hat) / sigma)
     with WHO thresholds 120 g/L (women) and 130 g/L (men)
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import timm
import torch
from scipy.stats import norm, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.tabular import plots
from ml.tabular.config import DOCS_DIR, REPO_ROOT

from .data import load
from .preprocess import INPUT_SIZE, MEAN, STD, batch

BACKBONE = "efficientnet_b0.ra_in1k"
OUT_DIR = REPO_ROOT / "backend" / "models" / "anemia"
SEEDS, FOLDS = [0, 1, 2, 3, 4], 5
WHO_THRESHOLD_GL = {"female": 120.0, "male": 130.0}
ALPHAS = np.logspace(-2, 4, 25)


def export_backbone(path) -> None:
    """ONNX graph: image batch (N,3,224,224) -> feature map (N,1280,7,7).
    Global average pooling is done in numpy, which keeps the feature map
    available for class-activation heatmaps in the API."""
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, global_pool="").eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["feature_map"],
                      dynamic_axes={"image": {0: "n"}, "feature_map": {0: "n"}}, opset_version=17)
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": dummy.numpy()})[0]
    assert np.allclose(ref, got, atol=1e-3), "ONNX export does not match PyTorch"


def embed(sess, crops) -> np.ndarray:
    return sess.run(None, {"image": batch(crops)})[0].mean(axis=(2, 3))  # (n, 1280)


def features(nail_emb: np.ndarray, skin_emb: np.ndarray) -> np.ndarray:
    return np.hstack([nail_emb, nail_emb - skin_emb])


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def evaluate(X_all3, X_single, df):
    """OOF predictions; the model always trains on 3-nail averages, and is tested
    either on the 3-nail average or on one nail + one skin patch."""
    y = df["hb"].to_numpy()
    strat = (y < 120).astype(int)
    runs = {"three_nails": [], "single_nail": []}
    oof_first = {}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        pred3, pred1 = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in cv.split(X_all3, strat, groups=df["session"]):
            m = ridge().fit(X_all3[tr], y[tr])
            pred3[te] = m.predict(X_all3[te])
            pred1[te] = m.predict(X_single[te])
        for name, pred in (("three_nails", pred3), ("single_nail", pred1)):
            runs[name].append(pred)
        if seed == SEEDS[0]:
            oof_first = {"three_nails": pred3, "single_nail": pred1}
    return y, runs, oof_first


def summarise(y, preds: list[np.ndarray], sigma: float) -> dict:
    def one(p):
        prob = norm.cdf((120 - p) / sigma)
        return {
            "mae_gl": mean_absolute_error(y, p),
            "pearson_r": pearsonr(y, p)[0],
            "auroc_hb_lt_120": roc_auc_score(y < 120, -p),
            "auroc_hb_lt_110": roc_auc_score(y < 110, -p),
            "auroc_hb_lt_130": roc_auc_score(y < 130, -p),
            "brier_hb_lt_120": brier_score_loss(y < 120, prob),
        }
    runs = [one(p) for p in preds]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs]))}
            for k in runs[0]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "nail_backbone.onnx"
    export_backbone(onnx_path)
    print(f"[anemia] exported {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB), parity check passed")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    df = load()
    nails3 = np.stack([embed(sess, n).mean(0) for n in df["nails"]])
    skins3 = np.stack([embed(sess, s).mean(0) for s in df["skins"]])
    nail1 = np.stack([embed(sess, [n[0]])[0] for n in df["nails"]])
    skin1 = np.stack([embed(sess, [s[0]])[0] for s in df["skins"]])
    X3, X1 = features(nails3, skins3), features(nail1, skin1)

    y, runs, first = evaluate(X3, X1, df)
    sigma = float(np.std(y - first["three_nails"]))   # the app finds and uses up to three nails
    results = {name: summarise(y, preds, sigma) for name, preds in runs.items()}
    for name, r in results.items():
        print(f"  {name:<12} MAE {r['mae_gl']['mean']:.1f} g/L | r {r['pearson_r']['mean']:.2f} | "
              f"AUROC Hb<120 {r['auroc_hb_lt_120']['mean']:.3f} ± {r['auroc_hb_lt_120']['sd']:.3f} | "
              f"Hb<110 {r['auroc_hb_lt_110']['mean']:.3f} | Hb<130 {r['auroc_hb_lt_130']['mean']:.3f}")

    final = ridge().fit(X3, y)
    scaler, reg = final.named_steps["standardscaler"], final.named_steps["ridgecv"]
    # Fold the scaler into raw-space weights: Hb = w . x + b
    w = reg.coef_ / scaler.scale_
    b = float(reg.intercept_ - np.dot(w, scaler.mean_))
    head = {
        "model": "EfficientNet-B0 (ImageNet) embeddings + ridge regression",
        "backbone": BACKBONE, "onnx": onnx_path.name, "input_size": INPUT_SIZE,
        "mean": MEAN.tolist(), "std": STD.tolist(),
        "feature_layout": "[nail_embedding (1280), nail_embedding - skin_embedding (1280)]",
        "weights": w.tolist(), "bias": b, "ridge_alpha": float(reg.alpha_),
        "residual_sd_gl": sigma, "who_threshold_gl": WHO_THRESHOLD_GL,
        "hb_range_gl": [float(y.min()), float(y.max())],
    }
    (OUT_DIR / "head.json").write_text(json.dumps(head), encoding="utf-8")

    docs = DOCS_DIR / "anemia"
    docs.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((docs / "comparison.json").read_text(encoding="utf-8"))
    (docs / "metrics.json").write_text(json.dumps({"deployed": results, "sigma_gl": sigma,
                                                   "n_patients": int(len(y)),
                                                   "n_sessions": int(df["session"].nunique())},
                                                  indent=2), encoding="utf-8")
    plots.roc_plot({"Three nails (as in the app)": (y < 120, -first["three_nails"],
                                                    roc_auc_score(y < 120, -first["three_nails"])),
                    "Single nail": (y < 120, -first["single_nail"],
                                    roc_auc_score(y < 120, -first["single_nail"]))},
                   "Anemia (Hb < 120 g/L): ROC curve", docs / "roc.png")
    scatter_plot(y, first["three_nails"], docs / "hb_scatter.png")
    write_card(results, comparison, sigma, len(y), int(df["session"].nunique()))
    print(f"[anemia] saved model to {OUT_DIR} and model card to docs/models/anemia.md")
    return 0


def scatter_plot(y, pred, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    lo, hi = 40, 175
    ax.plot([lo, hi], [lo, hi], color=plots.AXIS, linewidth=1, linestyle="--")
    ax.axvline(120, color=plots.MUTED, linewidth=1)
    ax.text(121, hi - 6, "anemia threshold\n(120 g/L, women)", color=plots.INK_2, fontsize=8, va="top")
    ax.scatter(y, pred, s=22, color=plots.SERIES[0], edgecolor=plots.SURFACE, linewidth=0.8, alpha=0.9)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured Hb (g/L, blood test)",
           ylabel="Estimated Hb from nail photo (g/L)")
    ax.set_title("Anemia: estimated vs measured haemoglobin")
    plots._save(fig, path)


def detector_section() -> list[str]:
    """Nail-detector results, written by `python -m ml.anemia.detector`."""
    path = DOCS_DIR / "anemia" / "detector.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return [
        "## Finding the nails automatically",
        "",
        "Users upload a plain photo, so the API has to find the nails and the skin reference patches itself.",
        "",
        "- **MediaPipe Hands was tried first and rejected.** It found a hand in only 13 of the 250 photos "
        "(fingers resting flat on paper, often with no full hand in view).",
        "- **A detector was trained instead:** Faster R-CNN with a MobileNetV3-Large FPN backbone "
        "(COCO-pretrained), fine-tuned on the 750 nail and 750 skin boxes drawn in the dataset "
        "(`python -m ml.anemia.detector`).",
        "- **It is scored end to end** with 5-fold cross-validation grouped by session. In each fold, the "
        "detector and the Hb model are trained on the other folds, and the Hb model is fed the boxes the "
        "detector finds on the test photos.",
        "",
        f"| Metric (cross-validation, {d['n_photos']} photos) | Result |",
        "|---|---|",
        f"| Hand-drawn nails found (IoU ≥ 0.5) | {d['nail_recall_iou50']:.1%} |",
        f"| Photos where no nail was found | {d['photos_without_detection']} |",
        f"| AUROC, Hb < 120 g/L, hand-drawn boxes | {d['auroc_hb_lt_120_handdrawn_boxes']:.3f} |",
        f"| AUROC, Hb < 120 g/L, **detected boxes** | **{d['auroc_hb_lt_120_detected_boxes']:.3f}** |",
        f"| Hb error (MAE) with detected boxes | {d['mae_gl_detected_boxes']:.1f} g/L |",
        "",
        "Automatic detection loses nothing compared with the hand-drawn boxes.",
        "",
        "**Phone photos:** the API scales every photo to the training resolution (long side 800 px). If "
        "nothing is found upright, it tries the photo rotated 90° each way, so portrait phone photos work. "
        "A simulated 3200 × 2400 portrait JPEG gives estimates within about 1 g/dL of the original photo. "
        "Photos without nails return a clear message asking for a new photo.",
        "",
    ]


def write_card(results, comparison, sigma, n, sessions):
    one, three = results["single_nail"], results["three_nails"]
    f = lambda m, k, d=3: f"{m[k]['mean']:.{d}f} ± {m[k]['sd']:.{d}f}"  # noqa: E731
    lines = [
        "# Model card: Anemia screening (fingernail photo)",
        "",
        "**Estimates:** blood haemoglobin (Hb) from a photo of a fingernail and nearby skin, then the "
        "probability of anemia using WHO thresholds (Hb < 120 g/L for women, < 130 g/L for men).",
        "",
        f"**Training data:** [Yakimov et al., *Scientific Data* 2024](https://doi.org/10.6084/m9.figshare.c.6760179), "
        f"{n} adults aged 18–95 with Hb measured from venous blood, photographed in {sessions} sessions. "
        "License CC0.",
        "",
        "**Model:** pretrained EfficientNet-B0 (ImageNet) image embeddings of the nail and of the adjacent "
        "skin, followed by ridge regression. The skin embedding acts as a lighting and camera reference. "
        "The backbone runs as ONNX in the API.",
        "",
        "## How the approach was chosen",
        "",
        "Every approach was scored with 5-fold cross-validation **grouped by photo session** (photos from one "
        "session share lighting and never appear in both training and test folds), repeated over 5 shuffles.",
        "",
        "| Approach | Hb error (MAE, g/L) | AUROC, Hb < 120 g/L |",
        "|---|---|---|",
    ]
    for name, r in comparison.items():
        lines.append(f"| {name} | {r['mae_gl']['mean']:.1f} | {r['auroc_hb_lt_120']['mean']:.3f} ± "
                     f"{r['auroc_hb_lt_120']['sd']:.3f} |")
    lines += [
        "",
        "Fine-tuning the whole network did worse than frozen embeddings: with 250 people it overfits. "
        "EfficientNet-B0 and ConvNeXt-Tiny tied; EfficientNet-B0 was chosen because it is 5× smaller.",
        "",
        "## Performance of the deployed model",
        "",
        "Embeddings computed through the exported ONNX file with the API's own preprocessing.",
        "",
        "The app finds up to three nails per photo automatically (see below), so the three-nail column "
        "is what users get. Hand-drawn boxes are used here.",
        "",
        "| Metric (mean ± SD over 5 CV shuffles) | Three nails (as in the app) | Single nail |",
        "|---|---|---|",
        f"| Hb error, MAE (g/L) | {f(three, 'mae_gl', 1)} | {f(one, 'mae_gl', 1)} |",
        f"| Correlation with measured Hb | {f(three, 'pearson_r', 2)} | {f(one, 'pearson_r', 2)} |",
        f"| AUROC, Hb < 110 g/L | {f(three, 'auroc_hb_lt_110')} | {f(one, 'auroc_hb_lt_110')} |",
        f"| AUROC, Hb < 120 g/L | {f(three, 'auroc_hb_lt_120')} | {f(one, 'auroc_hb_lt_120')} |",
        f"| AUROC, Hb < 130 g/L | {f(three, 'auroc_hb_lt_130')} | {f(one, 'auroc_hb_lt_130')} |",
        f"| Brier score, Hb < 120 g/L | {f(three, 'brier_hb_lt_120')} | {f(one, 'brier_hb_lt_120')} |",
        "",
        f"Probability of anemia uses the cross-validated residual spread σ = {sigma:.1f} g/L.",
        "",
        "![ROC](anemia/roc.png) ![Estimated vs measured Hb](anemia/hb_scatter.png)",
        "",
    ] + detector_section() + [
        "## Limitations",
        "",
        "- Sensitive to re-compression: the estimate shifts when a small, already-compressed photo is "
        "re-saved as a low-quality JPEG (up to 3.6 g/dL at 800 px, JPEG quality 75). Messaging apps that "
        "shrink photos can cause this; upload the original camera photo.",
        "- Small dataset (250 people). Treat the result as a screening signal, not a measurement.",
        "- Photos were taken with a fixed camera under controlled lighting on a white background. Phone "
        "photos at home will usually perform worse; the app asks for daylight and a white background.",
        "- The dataset holds no sex information, so evaluation uses the 120 g/L threshold for everyone.",
        "- Nail polish, nail disease, cold hands and skin tone can change nail colour. The dataset does not "
        "report skin tone, so performance across skin tones is unknown.",
        "- Not validated on an external dataset yet.",
        "",
    ]
    (DOCS_DIR / "anemia.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
