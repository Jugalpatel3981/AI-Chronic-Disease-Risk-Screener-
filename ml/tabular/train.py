"""Train, calibrate, evaluate and document the questionnaire-based risk models.

Usage (from the repository root, after building the NHANES dataset):
    python -m ml.tabular.train                 # all three diseases
    python -m ml.tabular.train diabetes ckd    # selected diseases

For each disease this:
  1. compares Logistic Regression, XGBoost and LightGBM with 5-fold CV
     (Optuna hyper-parameter search, AUROC objective)
  2. calibrates the best model's probabilities with isotonic regression fitted
     on out-of-fold predictions
  3. picks a screening threshold that catches >= 80% of cases
  4. evaluates on an internal held-out test set (2011-2018) and a temporal
     test set (2021-2023) with bootstrap 95% confidence intervals
  5. audits performance by sex, age group and ethnicity
  6. explains the model with SHAP
  7. saves the model bundle to backend/models/tabular/ and writes a model
     card with charts to docs/models/
"""

import datetime as dt
import json
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import plots
from .config import (ARTIFACT_DIR, BOOTSTRAP_ROUNDS, CLIP_RANGES, CV_FOLDS, DATASET_PATH, DISEASES,
                     DOCS_DIR, FEATURE_LABELS, OPTUNA_TRIALS, SEED, TARGET_SENSITIVITY,
                     TEST_FRACTION, features_for)
from .metrics import core_metrics, evaluate, threshold_for_sensitivity

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ data

def clip_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in X:
            X[col] = X[col].clip(lo, hi)
    return X


def load(disease: str):
    df = pd.read_parquet(DATASET_PATH)
    label = DISEASES[disease]["label"]
    df = df[df[label].notna()].reset_index(drop=True)
    feats = features_for(disease)
    df[feats] = clip_features(df[feats])
    dev = df[df["split"] == "train"].reset_index(drop=True)
    temporal = df[df["split"] == "temporal_test"].reset_index(drop=True)
    train, internal = train_test_split(dev, test_size=TEST_FRACTION, random_state=SEED,
                                       stratify=dev[label])
    return feats, label, train.reset_index(drop=True), internal.reset_index(drop=True), temporal


# ------------------------------------------------------------------ candidate models

def make_logreg(p):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=p["C"], max_iter=5000)),
    ])


def make_xgb(p):
    return xgb.XGBClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"], tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=8,
    )


def make_lgbm(p):
    return lgb.LGBMClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
        subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"], random_state=SEED, n_jobs=8, verbose=-1,
    )


def space_logreg(t):
    return {"C": t.suggest_float("C", 1e-3, 10, log=True)}


def space_xgb(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 20, log=True),
    }


def space_lgbm(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 48),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 300, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 20, log=True),
    }


CANDIDATES = {
    "Logistic Regression": (make_logreg, space_logreg, 10),
    "XGBoost": (make_xgb, space_xgb, OPTUNA_TRIALS),
    "LightGBM": (make_lgbm, space_lgbm, OPTUNA_TRIALS),
}


def tune(name, X, y, cv):
    make, space, trials = CANDIDATES[name]

    def objective(trial):
        return cross_val_score(make(space(trial)), X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = study.best_params
    folds = cross_val_score(make(best), X, y, cv=cv, scoring="roc_auc")
    return best, folds


# ------------------------------------------------------------------ evaluation helpers

def subgroup_rows(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> list[dict]:
    overall = roc_auc_score(y, p)
    age_band = pd.cut(frame["age"], [17, 39, 59, 80], labels=["Age 18–39", "Age 40–59", "Age 60+"])
    groups = {
        "Sex": frame["sex_male"].map({1: "Male", 0: "Female"}),
        "Age": age_band.astype(str),
        "Ethnicity": frame["ethnicity"].fillna("Unknown"),
    }
    rows, rng = [], np.random.default_rng(SEED)
    for group, labels in groups.items():
        for sub in sorted(labels.dropna().unique()):
            m = (labels == sub).to_numpy()
            ys, ps = y[m], p[m]
            if m.sum() < 100 or ys.min() == ys.max():
                continue
            boots = []
            for _ in range(300):
                i = rng.integers(0, len(ys), len(ys))
                if ys[i].min() != ys[i].max():
                    boots.append(roc_auc_score(ys[i], ps[i]))
            rows.append({
                "group": group, "subgroup": sub, "n": int(m.sum()),
                "auroc": float(roc_auc_score(ys, ps)),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "observed_rate": float(ys.mean()), "mean_predicted": float(ps.mean()),
                "overall": float(overall),
            })
    return rows


def shap_importance(model, X: pd.DataFrame, is_tree: bool) -> np.ndarray:
    sample = X.sample(min(2000, len(X)), random_state=SEED)
    if is_tree:
        values = shap.TreeExplainer(model).shap_values(sample)
        if isinstance(values, list):
            values = values[1]
    else:
        background = shap.sample(sample, 200, random_state=SEED)
        explainer = shap.Explainer(
            lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1],
            background)
        values = explainer(sample.iloc[:500], silent=True).values
    return np.abs(np.asarray(values)).mean(axis=0)


def fmt(m: dict, key: str, pct: bool = False) -> str:
    v = m[key]
    if pct:
        return f"{v['value'] * 100:.1f}% ({v['ci_low'] * 100:.1f}–{v['ci_high'] * 100:.1f})"
    return f"{v['value']:.3f} ({v['ci_low']:.3f}–{v['ci_high']:.3f})"


# ------------------------------------------------------------------ main pipeline

def train_disease(disease: str) -> dict:
    cfg = DISEASES[disease]
    feats, label, train, internal, temporal = load(disease)
    X, y = train[feats], train[label].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n=== {cfg['title']}: train {len(train):,} | internal test {len(internal):,} "
          f"| temporal test {len(temporal):,} | prevalence {y.mean():.1%}")

    # 1. model comparison
    comparison, params = {}, {}
    for name in CANDIDATES:
        best, folds = tune(name, X, y, cv)
        comparison[name], params[name] = folds, best
        print(f"  {name:<20} CV AUROC {folds.mean():.4f} ± {folds.std():.4f}")
    best_name = max(comparison, key=lambda n: comparison[n].mean())
    make = CANDIDATES[best_name][0]
    print(f"  -> selected {best_name}")

    # 2. calibration on out-of-fold predictions, then refit on all training data
    oof = cross_val_predict(make(params[best_name]), X, y, cv=cv, method="predict_proba")[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
    model = make(params[best_name]).fit(X, y)
    oof_cal = calibrator.predict(oof)

    # 3. screening threshold
    threshold = threshold_for_sensitivity(y, oof_cal, TARGET_SENSITIVITY)

    # 4. evaluation
    def predict(frame):
        raw = model.predict_proba(frame[feats])[:, 1]
        return raw, calibrator.predict(raw)

    results, curves_roc, curves_cal = {}, {}, {}
    for set_name, frame in (("Internal test (2011–2018)", internal),
                            ("Temporal test (2021–2023)", temporal)):
        yt = frame[label].astype(int).to_numpy()
        raw, cal = predict(frame)
        m = evaluate(yt, cal, threshold, BOOTSTRAP_ROUNDS, SEED)
        m["uncalibrated"] = core_metrics(yt, raw)
        results[set_name] = m
        curves_roc[set_name] = (yt, cal, m["auroc"]["value"])
        curves_cal[set_name] = (yt, cal)
        print(f"  {set_name:<27} AUROC {fmt(m, 'auroc')} | Brier {m['brier']['value']:.4f} "
              f"| sens {m['sensitivity']['value']:.2f} spec {m['specificity']['value']:.2f}")

    # 5. fairness (internal + temporal test combined for larger subgroups)
    test_all = pd.concat([internal, temporal], ignore_index=True)
    fair = subgroup_rows(test_all, test_all[label].astype(int).to_numpy(), predict(test_all)[1])

    # 6. explanation
    is_tree = best_name != "Logistic Regression"
    importance = shap_importance(model, internal[feats], is_tree)

    # 7. save artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "disease": disease,
        "title": cfg["title"],
        "definition": cfg["definition"],
        "model_type": best_name,
        "model": model,
        "calibrator": calibrator,
        "features": feats,
        "feature_labels": {f: FEATURE_LABELS[f] for f in feats},
        "clip_ranges": {k: v for k, v in CLIP_RANGES.items() if k in feats},
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "train_prevalence": float(y.mean()),
        "shap_background": internal[feats].sample(200, random_state=SEED).reset_index(drop=True),
        "trained_on": dt.date.today().isoformat(),
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgb.__version__,
                     "lightgbm": lgb.__version__},
    }
    joblib.dump(bundle, ARTIFACT_DIR / f"{disease}.joblib", compress=3)

    out_dir = DOCS_DIR / disease
    summary = {
        "disease": disease, "title": cfg["title"], "definition": cfg["definition"],
        "selected_model": best_name, "best_params": params[best_name],
        "cv_auroc": {n: {"mean": float(f.mean()), "sd": float(f.std()), "folds": f.tolist()}
                     for n, f in comparison.items()},
        "threshold": threshold, "results": results, "fairness": fair,
        "shap_importance": dict(sorted(
            {FEATURE_LABELS[f]: float(v) for f, v in zip(feats, importance)}.items(),
            key=lambda kv: -kv[1])),
        "n_train": int(len(train)), "train_prevalence": float(y.mean()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots.model_comparison_plot(comparison, f"{cfg['title']}: model comparison",
                                out_dir / "model_comparison.png")
    plots.roc_plot(curves_roc, f"{cfg['title']}: ROC curve", out_dir / "roc.png")
    plots.calibration_plot(curves_cal, f"{cfg['title']}: calibration", out_dir / "calibration.png")
    plots.importance_plot([FEATURE_LABELS[f] for f in feats], importance,
                          f"{cfg['title']}: what drives the prediction",
                          out_dir / "shap_importance.png",
                          unit="log-odds" if is_tree else "probability")
    plots.fairness_plot(fair, f"{cfg['title']}: performance by subgroup", out_dir / "fairness.png")
    write_model_card(summary, out_dir.parent / f"{disease}.md")
    return summary


def write_model_card(s: dict, path):
    internal = s["results"]["Internal test (2011–2018)"]
    temporal = s["results"]["Temporal test (2021–2023)"]
    d = s["disease"]
    lines = [
        f"# Model card: {s['title']} risk (questionnaire)",
        "",
        f"**Predicts:** {s['definition']}.",
        "",
        f"**Inputs:** age, sex, height, weight, BMI, waist (optional), exercise, sitting time, "
        f"sleep, alcohol, smoking, daily calories (optional)"
        + {"hypertension": ", and whether the person has diagnosed diabetes.",
           "diabetes": ", and whether the person has diagnosed high blood pressure.",
           "ckd": ", and whether the person has diagnosed high blood pressure or diabetes."}[d],
        "No lab tests are needed.",
        "",
        f"**Training data:** {s['n_train']:,} US adults from NHANES 2011–2018 "
        f"({s['train_prevalence']:.1%} prevalence). See [the dataset summary](../data/nhanes_dataset.md).",
        "",
        f"**Model:** {s['selected_model']}, picked from three candidates by 5-fold cross-validated AUROC. "
        "Probabilities are calibrated with isotonic regression on out-of-fold predictions.",
        "",
        "## Model comparison",
        "",
        "| Model | CV AUROC (mean ± SD) |",
        "|---|---|",
    ]
    for name, c in s["cv_auroc"].items():
        mark = " ✅" if name == s["selected_model"] else ""
        lines.append(f"| {name}{mark} | {c['mean']:.3f} ± {c['sd']:.3f} |")
    lines += [
        "",
        f"![Model comparison]({d}/model_comparison.png)",
        "",
        "## Performance",
        "",
        "95% confidence intervals from 1,000 bootstrap resamples. The temporal test set "
        "(2021–2023) was collected after the training period and was never used for training or tuning.",
        "",
        "| Metric | Internal test (2011–2018) | Temporal test (2021–2023) |",
        "|---|---|---|",
        f"| People | {internal['n']:,} | {temporal['n']:,} |",
        f"| Prevalence | {internal['prevalence']:.1%} | {temporal['prevalence']:.1%} |",
        f"| AUROC | {fmt(internal, 'auroc')} | {fmt(temporal, 'auroc')} |",
        f"| AUPRC | {fmt(internal, 'auprc')} | {fmt(temporal, 'auprc')} |",
        f"| Brier score (lower is better) | {fmt(internal, 'brier')} | {fmt(temporal, 'brier')} |",
        f"| Calibration error (ECE) | {fmt(internal, 'ece')} | {fmt(temporal, 'ece')} |",
        "",
        f"**Screening threshold:** risk ≥ {s['threshold']:.1%}, chosen on training data to catch "
        "at least 80% of cases.",
        "",
        "| At the screening threshold | Internal test | Temporal test |",
        "|---|---|---|",
        f"| Sensitivity (cases caught) | {fmt(internal, 'sensitivity', True)} | {fmt(temporal, 'sensitivity', True)} |",
        f"| Specificity | {fmt(internal, 'specificity', True)} | {fmt(temporal, 'specificity', True)} |",
        f"| PPV (flagged people who have it) | {fmt(internal, 'ppv', True)} | {fmt(temporal, 'ppv', True)} |",
        f"| NPV (cleared people who don't) | {fmt(internal, 'npv', True)} | {fmt(temporal, 'npv', True)} |",
        f"| People flagged | {internal['flagged_fraction']['value']:.1%} | {temporal['flagged_fraction']['value']:.1%} |",
        "",
        f"![ROC curve]({d}/roc.png) ![Calibration]({d}/calibration.png)",
        "",
        "## What drives the prediction (SHAP)",
        "",
        f"![SHAP importance]({d}/shap_importance.png)",
        "",
        "## Fairness audit",
        "",
        "AUROC by subgroup on the combined internal and temporal test sets. Ethnicity is "
        "**not** a model input; it is used only for this audit.",
        "",
        "AUROC *within* an age band is naturally lower than overall: age is the strongest "
        "predictor, and inside a narrow band it can no longer separate people. The useful "
        "fairness check is the comparison between sex and ethnicity groups, and whether mean "
        "predicted risk tracks the observed rate in each group (calibration).",
        "",
        "| Group | Subgroup | n | AUROC (95% CI) | Observed rate | Mean predicted |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["fairness"]:
        lines.append(f"| {r['group']} | {r['subgroup']} | {r['n']:,} | {r['auroc']:.3f} "
                     f"({r['ci_low']:.3f}–{r['ci_high']:.3f}) | {r['observed_rate']:.1%} | "
                     f"{r['mean_predicted']:.1%} |")
    gaps = [r for r in s["fairness"] if abs(r["observed_rate"] - r["mean_predicted"]) >= 0.03
            and r["group"] != "Age"]
    lines += ["", f"![Fairness]({d}/fairness.png)", ""]
    if gaps:
        lines += ["**Calibration gaps (≥3 percentage points between observed and predicted):**", ""]
        for r in gaps:
            direction = "under-estimates" if r["observed_rate"] > r["mean_predicted"] else "over-estimates"
            lines.append(f"- The model {direction} risk for **{r['subgroup']}** adults "
                         f"(observed {r['observed_rate']:.1%}, predicted {r['mean_predicted']:.1%}).")
        lines += [
            "",
            "Ranking (AUROC) is similar across groups, but absolute risk is off for these groups. "
            "Possible fixes are group-specific recalibration or adding ethnicity as an input; both "
            "involve trade-offs and are left as future work.",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- This is a screening estimate, not a diagnosis. Labels come from a single NHANES visit.",
        "- It is trained on US adults; performance in other populations hasn't been validated.",
        "- Lifestyle inputs are self-reported, so reporting errors reduce accuracy.",
        "- It estimates whether the condition is present *now* (prevalence), not the risk of "
        "developing it in the future.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overview(summaries: list[dict]):
    lines = [
        "# Questionnaire risk models: overview",
        "",
        "Generated by `python -m ml.tabular.train`. Each model has its own card with full "
        "metrics, charts and a fairness audit.",
        "",
        "| Condition | Model | AUROC internal test | AUROC temporal test (2021–2023) | Sensitivity / specificity (temporal) |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        i = s["results"]["Internal test (2011–2018)"]
        t = s["results"]["Temporal test (2021–2023)"]
        lines.append(
            f"| [{s['title']}]({s['disease']}.md) | {s['selected_model']} | {fmt(i, 'auroc')} | "
            f"{fmt(t, 'auroc')} | {t['sensitivity']['value']:.0%} / {t['specificity']['value']:.0%} |")
    (DOCS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    diseases = argv or list(DISEASES)
    unknown = set(diseases) - set(DISEASES)
    if unknown:
        print(f"unknown disease(s): {', '.join(unknown)}; choose from {', '.join(DISEASES)}")
        return 1
    summaries = [train_disease(d) for d in diseases]
    if set(diseases) == set(DISEASES):
        write_overview(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
