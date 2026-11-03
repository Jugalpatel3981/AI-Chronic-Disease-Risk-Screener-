import numpy as np
import pytest

from ml.tabular.metrics import (evaluate, expected_calibration_error, screening_stats,
                                threshold_for_sensitivity)


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    t = threshold_for_sensitivity(y, p, 0.8)
    sens = screening_stats(y, p, t)["sensitivity"]
    assert 0.8 <= sens < 0.81


def test_screening_stats_known_confusion_matrix():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    s = screening_stats(y, p, 0.5)
    assert s["sensitivity"] == 0.5        # 1 of 2 cases caught
    assert s["specificity"] == pytest.approx(2 / 3)
    assert s["ppv"] == 0.5
    assert s["flagged_fraction"] == 0.4


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)],
                        np.r_[np.ones(500), np.zeros(500)],
                        np.r_[np.ones(900), np.zeros(100)]])
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_evaluate_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    m = evaluate(y, p, 0.5, rounds=200, seed=0)
    for key in ("auroc", "brier", "sensitivity"):
        assert m[key]["ci_low"] <= m[key]["value"] <= m[key]["ci_high"]
    assert m["n"] == 500
