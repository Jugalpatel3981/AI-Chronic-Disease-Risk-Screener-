"""Compare approaches for estimating haemoglobin from fingernail photos.

Usage (from the repository root):
    python -m ml.anemia.compare

Every approach is scored with the same protocol: 5-fold cross-validation,
grouped by photo session (photos from one session share lighting and must not
appear in both training and test folds), stratified by anaemia status, and
repeated with 5 different shuffles. Results are written to
docs/models/anemia/comparison.json.
"""

import json
import sys

import numpy as np
import pandas as pd
import timm
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.tabular.config import DOCS_DIR

from .color_features import patient_features
from .data import load

SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5
ANEMIA_GL = 120  # WHO threshold for non-pregnant women; sex is not in this dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONES = {
    "EfficientNet-B0": "efficientnet_b0.ra_in1k",
    "ConvNeXt-Tiny": "convnext_tiny.fb_in22k",
    "DINOv2 ViT-S/14": "vit_small_patch14_dinov2.lvd142m",
}


def splits(df: pd.DataFrame, seed: int):
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    return cv.split(df, (df["hb"] < ANEMIA_GL).astype(int), groups=df["session"])


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def oof_predictions(X: np.ndarray, df: pd.DataFrame, seed: int, make_model=ridge) -> np.ndarray:
    pred = np.zeros(len(df))
    y = df["hb"].to_numpy()
    for tr, te in splits(df, seed):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae_gl": float(mean_absolute_error(y, pred)),
        "pearson_r": float(pearsonr(y, pred)[0]) if np.std(pred) > 0 else 0.0,
        "auroc_hb_lt_120": float(roc_auc_score(y < 120, -pred)),
        "auroc_hb_lt_110": float(roc_auc_score(y < 110, -pred)),
        "auroc_hb_lt_130": float(roc_auc_score(y < 130, -pred)),
    }


@torch.no_grad()
def embed(crops_per_patient: list[list[np.ndarray]], model_name: str) -> np.ndarray:
    model = timm.create_model(model_name, pretrained=True, num_classes=0,
                              **({"dynamic_img_size": True} if "dinov2" in model_name else {}))
    model.eval().to(DEVICE)
    cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224)}, model=model)
    tf = timm.data.create_transform(**cfg)
    from PIL import Image
    out = []
    for crops in crops_per_patient:
        batch = torch.stack([tf(Image.fromarray(c)) for c in crops]).to(DEVICE)
        out.append(model(batch).mean(0).cpu().numpy())  # average the 3 nails
    return np.stack(out)


def finetune_oof(df: pd.DataFrame, seed: int, epochs: int = 25) -> np.ndarray:
    """End-to-end fine-tuned EfficientNet-B0 regressing Hb from single nail crops."""
    from PIL import Image
    from torchvision import transforms as T

    torch.manual_seed(seed)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Geometric augmentation only: colour jitter would destroy the pallor signal.
    train_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(15), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    y_mean, y_std = df["hb"].mean(), df["hb"].std()
    pred = np.zeros(len(df))
    for tr, te in splits(df, seed):
        model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        items = [(Image.fromarray(c), (h - y_mean) / y_std)
                 for i in tr for c, h in zip(df.iloc[i]["nails"], [df.iloc[i]["hb"]] * 3)]
        for _ in range(epochs):
            model.train()
            perm = np.random.default_rng(seed).permutation(len(items))
            for s in range(0, len(perm), 16):
                idx = perm[s:s + 16]
                xb = torch.stack([train_tf(items[k][0]) for k in idx]).to(DEVICE)
                yb = torch.tensor([items[k][1] for k in idx], dtype=torch.float32, device=DEVICE)
                loss = torch.nn.functional.smooth_l1_loss(model(xb).squeeze(1), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                xb = torch.stack([test_tf(Image.fromarray(c)) for c in df.iloc[i]["nails"]]).to(DEVICE)
                pred[i] = model(xb).mean().item() * y_std + y_mean
        del model
        torch.cuda.empty_cache()
    return pred


def main() -> int:
    df = load()
    y = df["hb"].to_numpy()
    print(f"[anemia] {len(df)} patients, {df['session'].nunique()} sessions, "
          f"{(y < ANEMIA_GL).mean():.0%} with Hb < {ANEMIA_GL} g/L, device={DEVICE}")

    nail_only = pd.DataFrame([patient_features(n) for n in df["nails"]]).to_numpy()
    nail_skin = pd.DataFrame([patient_features(n, s) for n, s in zip(df["nails"], df["skins"])]).to_numpy()
    features = {
        "Baseline (predict mean Hb)": np.zeros((len(df), 1)),
        "Colour features: nail": nail_only,
        "Colour features: nail + skin reference": nail_skin,
    }
    for name, model_name in BACKBONES.items():
        print(f"[anemia] embedding with {name}")
        nail_emb = embed(list(df["nails"]), model_name)
        skin_emb = embed(list(df["skins"]), model_name)
        features[f"{name} embedding: nail"] = nail_emb
        features[f"{name} embedding: nail + skin"] = np.hstack([nail_emb, nail_emb - skin_emb])

    results = {}
    for name, X in features.items():
        runs = [scores(y, oof_predictions(X, df, s)) for s in SEEDS]
        results[name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                             "sd": float(np.std([r[k] for r in runs]))} for k in runs[0]}
        r = results[name]
        print(f"  {name:<45} MAE {r['mae_gl']['mean']:5.1f} g/L | r {r['pearson_r']['mean']:.2f} "
              f"| AUROC(<120) {r['auroc_hb_lt_120']['mean']:.3f} ± {r['auroc_hb_lt_120']['sd']:.3f}")

    print("[anemia] fine-tuning EfficientNet-B0 (2 seeds x 5 folds)")
    runs = [scores(y, finetune_oof(df, s)) for s in SEEDS[:2]]
    name = "Fine-tuned EfficientNet-B0: nail"
    results[name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                         "sd": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    r = results[name]
    print(f"  {name:<45} MAE {r['mae_gl']['mean']:5.1f} g/L | r {r['pearson_r']['mean']:.2f} "
          f"| AUROC(<120) {r['auroc_hb_lt_120']['mean']:.3f} ± {r['auroc_hb_lt_120']['sd']:.3f}")

    out = DOCS_DIR / "anemia"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
