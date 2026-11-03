import numpy as np
import pandas as pd
import pytest

from ml.nhanes import features as F


def s(*values):
    return pd.Series(values, dtype="float")


# ---------------------------------------------------------------- eGFR

@pytest.mark.parametrize(
    "scr, age, male, expected",
    [
        (0.8, 50, 0, 89.7),   # reference values from the CKD-EPI 2021 equation
        (1.0, 60, 1, 86.2),
        (2.0, 70, 1, 35.2),
        (0.6, 30, 0, 123.8),
    ],
)
def test_egfr_ckd_epi_2021(scr, age, male, expected):
    got = F.egfr_ckd_epi_2021(s(scr), s(age), s(male)).iloc[0]
    assert got == pytest.approx(expected, abs=0.1)


def test_egfr_missing_creatinine_is_nan():
    assert np.isnan(F.egfr_ckd_epi_2021(s(np.nan), s(50), s(1)).iloc[0])


# ---------------------------------------------------------------- codes

def test_valid_drops_refused_and_dont_know_codes():
    out = F.valid(s(3, 7, 77, 9999), 1, 7)
    assert out.tolist()[:2] == [3, 7]
    assert out.iloc[2:].isna().all()


def test_clean_xpt_zeros():
    df = F.clean_xpt_zeros(pd.DataFrame({"x": [5.397605e-79, 2.0]}))
    assert df["x"].tolist() == [0.0, 2.0]


# ---------------------------------------------------------------- alcohol

def test_drinks_per_week_2011_2016():
    alq = pd.DataFrame({
        "ALQ101": [1, 1, 2, 9],
        "ALQ110": [np.nan, np.nan, 2, np.nan],
        "ALQ120Q": [2, 0, np.nan, np.nan],   # 2 times per week / never
        "ALQ120U": [1, np.nan, np.nan, np.nan],
        "ALQ130": [3, np.nan, np.nan, np.nan],
    })
    out = F.drinks_per_week_2011_2016(alq)
    assert out.iloc[0] == pytest.approx(6.0)   # 2 days x 3 drinks
    assert out.iloc[1] == 0.0                  # did not drink in past year
    assert out.iloc[2] == 0.0                  # never a drinker
    assert np.isnan(out.iloc[3])               # unknown


def test_drinks_per_week_2017_plus():
    alq = pd.DataFrame({
        "ALQ111": [1, 1, 2],
        "ALQ121": [5, 0, np.nan],   # once a week / never in past year / skipped
        "ALQ130": [4, np.nan, np.nan],
    })
    assert F.drinks_per_week_2017_plus(alq).tolist() == [4.0, 0.0, 0.0]


# ---------------------------------------------------------------- activity

def test_leisure_activity_2011_2018():
    paq = pd.DataFrame({
        "PAQ650": [1, 2], "PAQ655": [3, np.nan], "PAD660": [30, np.nan],
        "PAQ665": [2, 1], "PAQ670": [np.nan, 5], "PAD675": [np.nan, 9999],
    })
    out = F.leisure_activity_2011_2018(paq)
    assert out["vigorous_min_week"].tolist() == [90.0, 0.0]
    assert out["moderate_min_week"].iloc[0] == 0.0
    assert np.isnan(out["moderate_min_week"].iloc[1])  # minutes = don't know


def test_leisure_activity_2021_2023():
    paq = pd.DataFrame({
        "PAD790Q": [3, 0, 1], "PAD790U": [b"W", b"", b"D"], "PAD800": [40, np.nan, 20],
        "PAD810Q": [0, 2, 9999], "PAD810U": [b"", b"M", b""], "PAD820": [np.nan, 60, np.nan],
    })
    out = F.leisure_activity_2021_2023(paq)
    assert out["moderate_min_week"].tolist() == [120.0, 0.0, 140.0]
    assert out["vigorous_min_week"].iloc[0] == 0.0
    assert out["vigorous_min_week"].iloc[1] == pytest.approx(2 * 12 / 52 * 60)
    assert np.isnan(out["vigorous_min_week"].iloc[2])


# ---------------------------------------------------------------- smoking / BP

def test_smoking_status():
    smq = pd.DataFrame({"SMQ020": [2, 1, 1, 1, 9], "SMQ040": [np.nan, 3, 1, 2, np.nan]})
    out = F.smoking_status(smq)
    assert out.iloc[:4].tolist() == [0, 1, 2, 2]
    assert np.isnan(out.iloc[4])


def test_mean_bp_ignores_zero_diastolic():
    bp = pd.DataFrame({"S1": [120, 140], "S2": [130, np.nan], "D1": [0, 90], "D2": [80, 70]})
    out = F.mean_bp(bp, ["S1", "S2"], ["D1", "D2"])
    assert out["sbp_mean"].tolist() == [125.0, 140.0]
    assert out["dbp_mean"].tolist() == [80.0, 80.0]
