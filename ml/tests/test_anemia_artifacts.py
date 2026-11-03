"""Checks on the exported anaemia model: it loads, produces plausible Hb
estimates, and paler nails give lower Hb estimates."""

import json

import numpy as np
import pytest

from ml.anemia.preprocess import batch
from ml.tabular.config import REPO_ROOT

MODEL_DIR = REPO_ROOT / "backend" / "models" / "anemia"


@pytest.fixture(scope="module")
def model():
    if not (MODEL_DIR / "head.json").exists():
        pytest.skip("anemia model not built yet")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_DIR / "nail_backbone.onnx"),
                                providers=["CPUExecutionProvider"])
    head = json.loads((MODEL_DIR / "head.json").read_text(encoding="utf-8"))
    return sess, head


def estimate_hb(model, nails, skins):
    sess, head = model
    n = sess.run(None, {"image": batch(nails)})[0].mean(axis=(2, 3)).mean(0)
    s = sess.run(None, {"image": batch(skins)})[0].mean(axis=(2, 3)).mean(0)
    return float(np.dot(head["weights"], np.concatenate([n, n - s])) + head["bias"])


def make_paler(crop, amount=0.5):
    """Blend the nail towards a pale, whitish colour (simulates pallor)."""
    return (crop * (1 - amount) + np.array([235, 225, 220]) * amount).astype(np.uint8)


@pytest.fixture(scope="module")
def people():
    from ml.anemia.data import RAW_DIR, load
    if not (RAW_DIR / "metadata.csv").exists():
        pytest.skip("anemia dataset not downloaded")
    return load().sample(20, random_state=0)


def test_head_matches_backbone(model):
    _, head = model
    assert len(head["weights"]) == 2 * 1280
    assert head["residual_sd_gl"] > 0


def test_estimates_are_plausible(model, people):
    for p in people.itertuples():
        assert 20 < estimate_hb(model, p.nails, p.skins) < 220


def test_paler_nails_give_lower_hb(model, people):
    for p in people.itertuples():
        normal = estimate_hb(model, p.nails, p.skins)
        pale = estimate_hb(model, [make_paler(c) for c in p.nails], p.skins)
        assert pale < normal
