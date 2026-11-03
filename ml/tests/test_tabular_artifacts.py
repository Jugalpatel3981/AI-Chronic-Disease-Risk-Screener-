"""Sanity checks on the trained model bundles: they load, return valid
probabilities, and react to risk factors in the medically expected direction."""

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.tabular.config import ARTIFACT_DIR, DISEASES

REFERENCE_PERSON = {
    "age": 50, "sex_male": 1, "height_cm": 175, "weight_kg": 80, "bmi": 80 / 1.75**2,
    "waist_cm": 95, "exercise_min_week": 120, "sedentary_min_day": 360, "sleep_hours": 7,
    "alcohol_drinks_week": 2, "smoking_status": 0, "calories_kcal": 2200,
    "told_high_bp": 0, "told_diabetes": 0,
}


def risk(bundle, **changes):
    person = {**REFERENCE_PERSON, **changes}
    X = pd.DataFrame([{f: person[f] for f in bundle["features"]}])
    raw = bundle["model"].predict_proba(X)[:, 1]
    return float(bundle["calibrator"].predict(raw)[0])


@pytest.fixture(params=list(DISEASES))
def bundle(request):
    path = ARTIFACT_DIR / f"{request.param}.joblib"
    if not path.exists():
        pytest.skip(f"{path.name} not trained yet")
    return joblib.load(path)


def test_bundle_has_required_fields(bundle):
    for key in ("model", "calibrator", "features", "threshold", "clip_ranges", "feature_labels"):
        assert key in bundle
    assert 0 < bundle["threshold"] < 1


def test_probability_is_valid(bundle):
    assert 0.0 <= risk(bundle) <= 1.0


def test_missing_optional_inputs_are_handled(bundle):
    p = risk(bundle, waist_cm=np.nan, calories_kcal=np.nan)
    assert 0.0 <= p <= 1.0


def test_risk_rises_with_age(bundle):
    assert risk(bundle, age=70) > risk(bundle, age=30)


def test_risk_rises_with_obesity(bundle):
    heavy = {"weight_kg": 115, "bmi": 115 / 1.75**2, "waist_cm": 120}
    assert risk(bundle, **heavy) >= risk(bundle)


def test_ckd_risk_rises_with_diabetes_and_high_bp():
    path = ARTIFACT_DIR / "ckd.joblib"
    if not path.exists():
        pytest.skip("ckd model not trained yet")
    b = joblib.load(path)
    assert risk(b, told_diabetes=1, told_high_bp=1) > risk(b)
