"""Train the diabetic-retinopathy grading model on DeepDRiD.

Usage (from the repository root; needs a CUDA GPU for reasonable speed):
    python -m ml.retina.train                       # EfficientNet-B0
    python -m ml.retina.train --backbone efficientnet_b2.ra_in1k

Protocol
  - 5-fold cross-validation on the 1,600 development images, grouped by
    patient (both eyes of a person stay in the same fold)
  - the network regresses the DR grade (0-4); cut-points that turn the score
    into a grade are tuned on out-of-fold predictions to maximise quadratic
    weighted kappa (QWK)
  - probabilities for "referable DR" (grade >= 2) and "any DR" (grade >= 1)
    come from logistic calibration of the out-of-fold scores
  - the 5 fold models are averaged (ensemble) and scored ONCE on the 400
    held-out challenge-evaluation images from 100 different patients
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from ml.tabular.config import DOCS_DIR, REPO_ROOT

from .data import GRADE_NAMES, load
from .preprocess import INPUT_SIZE

SEED = 42
FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = REPO_ROOT / "ml" / "retina" / "runs"   # git-ignored checkpoints
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ data

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(INPUT_SIZE, scale=(0.80, 1.0), ratio=(0.95, 1.05)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class Fundus(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.grades, self.tf = list(df["cached"]), df["grade"].to_numpy(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(float(self.grades[i]))


# ------------------------------------------------------------------ model

def make_model(backbone: str) -> torch.nn.Module:
    return timm.create_model(backbone, pretrained=True, num_classes=1, drop_rate=0.3,
                             drop_path_rate=0.1)


def train_fold(backbone, train_df, epochs, batch, lr) -> torch.nn.Module:
    model = make_model(backbone).to(DEVICE)
    loader = DataLoader(Fundus(train_df, TRAIN_TF), batch_size=batch, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total = epochs * len(loader)
    warm = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    for ep in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = torch.nn.functional.smooth_l1_loss(model(x).squeeze(1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        print(f"    epoch {ep + 1:>2}/{epochs}  loss {running / len(loader):.4f}  ({time.time() - t0:.0f}s)",
              flush=True)
    return model


@torch.no_grad()
def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Test-time augmentation: original, horizontal flip, vertical flip."""
    model.eval()
    loader = DataLoader(Fundus(df, EVAL_TF), batch_size=16, num_workers=4)
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
            out.append(torch.stack([model(v).squeeze(1).float() for v in views]).mean(0).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ post-processing

def to_grade(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.digitize(score, np.sort(cuts))


def fit_cutpoints(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    def loss(c):
        return -cohen_kappa_score(y, to_grade(score, c), weights="quadratic")
    best = minimize(loss, x0=np.array([0.5, 1.5, 2.5, 3.5]), method="Nelder-Mead",
                    options={"xatol": 1e-3, "maxiter": 2000})
    return np.sort(best.x)


def fit_prob(score: np.ndarray, target: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e3).fit(score.reshape(-1, 1), target)


def threshold_for_sensitivity(y, p, target=0.90) -> float:
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[max(int(np.ceil(target * len(pos))) - 1, 0)])


def metrics(df: pd.DataFrame, score, cuts, p_ref, p_any, ref_thr) -> dict:
    y = df["grade"].to_numpy()
    grade = to_grade(score, cuts)
    ref, anydr = (y >= 2).astype(int), (y >= 1).astype(int)
    flag = p_ref >= ref_thr
    return {
        "n_images": int(len(y)),
        "qwk": float(cohen_kappa_score(y, grade, weights="quadratic")),
        "accuracy": float((grade == y).mean()),
        "auroc_referable": float(roc_auc_score(ref, p_ref)),
        "auroc_any_dr": float(roc_auc_score(anydr, p_any)),
        "referable_sensitivity": float(flag[ref == 1].mean()),
        "referable_specificity": float((~flag)[ref == 0].mean()),
    }


def patient_bootstrap(df, score, cuts, p_ref, p_any, ref_thr, rounds=1000) -> dict:
    """95% CIs resampling *patients* (images of one patient are correlated)."""
    rng = np.random.default_rng(SEED)
    pids = df["patient_id"].to_numpy()
    groups = {p: np.where(pids == p)[0] for p in np.unique(pids)}
    keys = list(groups)
    samples = []
    for _ in range(rounds):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        sub = df.iloc[idx]
        if sub["grade"].ge(2).nunique() < 2:
            continue
        samples.append(metrics(sub, score[idx], cuts, p_ref[idx], p_any[idx], ref_thr))
    return {k: [float(np.percentile([s[k] for s in samples], 2.5)),
                float(np.percentile([s[k] for s in samples], 97.5))]
            for k in samples[0] if k != "n_images"}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dev, test = load()
    run_dir = WORK_DIR / args.backbone.split(".")[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[retina] {args.backbone} | dev {len(dev)} images / {dev.patient_id.nunique()} patients "
          f"| test {len(test)} images / {test.patient_id.nunique()} patients | device {DEVICE}")

    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(dev))
    test_scores = []
    for fold, (tr, va) in enumerate(cv.split(dev, dev["grade"], groups=dev["patient_id"])):
        print(f"  fold {fold + 1}/{FOLDS}: train {len(tr)} | val {len(va)}", flush=True)
        model = train_fold(args.backbone, dev.iloc[tr], args.epochs, args.batch, args.lr)
        oof[va] = predict(model, dev.iloc[va])
        test_scores.append(predict(model, test))
        torch.save(model.state_dict(), run_dir / f"fold{fold}.pt")
        qwk = cohen_kappa_score(dev["grade"].iloc[va], to_grade(oof[va], np.array([.5, 1.5, 2.5, 3.5])),
                                weights="quadratic")
        print(f"  fold {fold + 1} val QWK (default cut-points) {qwk:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # post-processing fitted on out-of-fold predictions only
    y = dev["grade"].to_numpy()
    cuts = fit_cutpoints(oof, y)
    cal_ref, cal_any = fit_prob(oof, (y >= 2).astype(int)), fit_prob(oof, (y >= 1).astype(int))
    p_ref_oof = cal_ref.predict_proba(oof.reshape(-1, 1))[:, 1]
    p_any_oof = cal_any.predict_proba(oof.reshape(-1, 1))[:, 1]
    ref_thr = threshold_for_sensitivity((y >= 2).astype(int), p_ref_oof, 0.90)

    test_score = np.mean(test_scores, axis=0)
    p_ref_test = cal_ref.predict_proba(test_score.reshape(-1, 1))[:, 1]
    p_any_test = cal_any.predict_proba(test_score.reshape(-1, 1))[:, 1]

    results = {
        "backbone": args.backbone, "epochs": args.epochs, "input_size": INPUT_SIZE,
        "cutpoints": cuts.tolist(), "referable_threshold": ref_thr,
        "calibration": {"referable": [float(cal_ref.coef_[0, 0]), float(cal_ref.intercept_[0])],
                        "any_dr": [float(cal_any.coef_[0, 0]), float(cal_any.intercept_[0])]},
        "cross_validation": metrics(dev, oof, cuts, p_ref_oof, p_any_oof, ref_thr),
        "test": metrics(test, test_score, cuts, p_ref_test, p_any_test, ref_thr),
        "test_ci95": patient_bootstrap(test, test_score, cuts, p_ref_test, p_any_test, ref_thr),
        "test_confusion": confusion_matrix(test["grade"], to_grade(test_score, cuts),
                                           labels=range(5)).tolist(),
        "grade_names": GRADE_NAMES,
    }
    np.save(run_dir / "oof.npy", oof)
    np.save(run_dir / "test_score.npy", test_score)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    out = DOCS_DIR / "retina"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{args.backbone.split('.')[0]}.json").write_text(json.dumps(results, indent=2),
                                                                    encoding="utf-8")
    cvm, tm, ci = results["cross_validation"], results["test"], results["test_ci95"]
    print(f"\n[retina] CV   QWK {cvm['qwk']:.3f} | AUROC referable {cvm['auroc_referable']:.3f} "
          f"| any DR {cvm['auroc_any_dr']:.3f}")
    print(f"[retina] TEST QWK {tm['qwk']:.3f} ({ci['qwk'][0]:.3f}-{ci['qwk'][1]:.3f}) | AUROC referable "
          f"{tm['auroc_referable']:.3f} ({ci['auroc_referable'][0]:.3f}-{ci['auroc_referable'][1]:.3f}) "
          f"| any DR {tm['auroc_any_dr']:.3f} | sens {tm['referable_sensitivity']:.2f} "
          f"spec {tm['referable_specificity']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
