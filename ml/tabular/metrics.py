"""Evaluation metrics: discrimination, calibration, screening performance, CIs."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is still >= target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)])


def screening_stats(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "flagged_fraction": float(pred.mean()),
    }


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def evaluate(y, p, threshold: float, rounds: int, seed: int) -> dict:
    """Point estimates plus bootstrap 95% confidence intervals."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    point = {**core_metrics(y, p), **screening_stats(y, p, threshold)}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    n = len(y)
    for _ in range(rounds):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        boot = {**core_metrics(y[i], p[i]), **screening_stats(y[i], p[i], threshold)}
        for k, v in boot.items():
            samples[k].append(v)
    result = {"n": int(n), "prevalence": float(y.mean())}
    for k, v in point.items():
        lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
        result[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return result
