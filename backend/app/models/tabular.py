"""Questionnaire risk models (hypertension, diabetes, CKD)."""

import ml.tabular  # noqa: F401  (imports LightGBM before XGBoost; see ml/tabular/__init__.py)

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from ..schemas import DiseaseRisk, Factor, Questionnaire

DISEASES = ["hypertension", "diabetes", "ckd"]
SMOKING = {"never": 0, "former": 1, "current": 2}
TOP_FACTORS = 5


def to_features(q: Questionnaire) -> dict:
    bmi = q.weight_kg / (q.height_cm / 100) ** 2
    return {
        "age": q.age, "sex_male": 1 if q.sex == "male" else 0,
        "height_cm": q.height_cm, "weight_kg": q.weight_kg, "bmi": bmi,
        "waist_cm": q.waist_cm if q.waist_cm is not None else np.nan,
        "exercise_min_week": q.exercise_min_week,
        "sedentary_min_day": q.sitting_hours_day * 60,
        "sleep_hours": q.sleep_hours, "alcohol_drinks_week": q.alcohol_drinks_week,
        "smoking_status": SMOKING[q.smoking],
        "calories_kcal": q.calories_kcal if q.calories_kcal is not None else np.nan,
        "told_high_bp": int(q.diagnosed_high_bp), "told_diabetes": int(q.diagnosed_diabetes),
    }


def risk_level(p: float, threshold: float) -> str:
    """Below the screening threshold = low; the upper half of the remaining
    range = high; in between = moderate."""
    if p < threshold:
        return "low"
    return "high" if p >= (threshold + 1) / 2 else "moderate"


class TabularModel:
    def __init__(self, path: Path):
        b = joblib.load(path)
        self.disease = b["disease"]
        self.title, self.definition, self.model_type = b["title"], b["definition"], b["model_type"]
        self.model, self.calibrator = b["model"], b["calibrator"]
        self.features, self.labels = b["features"], b["feature_labels"]
        self.clip, self.threshold = b["clip_ranges"], float(b["threshold"])

    def frame(self, feats: dict) -> pd.DataFrame:
        row = {f: feats[f] for f in self.features}
        for f, (lo, hi) in self.clip.items():
            if not pd.isna(row[f]):
                row[f] = float(np.clip(row[f], lo, hi))
        return pd.DataFrame([row], dtype=float)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values (log-odds) computed natively by the tree library,
        so the API does not need the `shap` package. The last column is the bias."""
        if isinstance(self.model, xgb.XGBClassifier):
            out = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        elif isinstance(self.model, lgb.LGBMClassifier):
            out = self.model.predict(X, pred_contrib=True)
        else:
            raise TypeError(f"unsupported model type {type(self.model).__name__}")
        return np.asarray(out)[0, :-1]

    def predict(self, feats: dict, already_diagnosed: bool) -> DiseaseRisk:
        X = self.frame(feats)
        p = float(self.calibrator.predict(self.model.predict_proba(X)[:, 1])[0])
        values = self.contributions(X)
        order = np.argsort(-np.abs(values))[:TOP_FACTORS]
        factors = [Factor(feature=self.features[i], label=self.labels[self.features[i]],
                          value=None if pd.isna(X.iloc[0, i]) else float(X.iloc[0, i]),
                          impact=float(values[i]),
                          direction="raises" if values[i] > 0 else "lowers") for i in order]
        return DiseaseRisk(
            disease=self.disease, title=self.title, probability=p,
            risk_level=risk_level(p, self.threshold), screen_positive=p >= self.threshold,
            threshold=self.threshold, already_diagnosed=already_diagnosed, top_factors=factors,
            definition=self.definition, model=f"{self.model_type} (calibrated)",
        )


class TabularService:
    def __init__(self, model_dir: Path):
        self.models = {d: TabularModel(model_dir / "tabular" / f"{d}.joblib") for d in DISEASES}

    def predict(self, q: Questionnaire) -> tuple[float, list[DiseaseRisk]]:
        feats = to_features(q)
        diagnosed = {"hypertension": q.diagnosed_high_bp, "diabetes": q.diagnosed_diabetes, "ckd": False}
        return feats["bmi"], [m.predict(feats, diagnosed[d]) for d, m in self.models.items()]
