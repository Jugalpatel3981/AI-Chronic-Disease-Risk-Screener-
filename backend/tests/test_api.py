"""End-to-end API tests with FastAPI's TestClient (real models, no mocks)."""

import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

PERSON = {
    "age": 55, "sex": "male", "height_cm": 175, "weight_kg": 92, "waist_cm": 104,
    "exercise_min_week": 30, "sitting_hours_day": 9, "sleep_hours": 6, "alcohol_drinks_week": 8,
    "smoking": "current", "calories_kcal": 2600, "diagnosed_high_bp": False, "diagnosed_diabetes": False,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models"]["questionnaire"] is True


# ---------------------------------------------------------------- questionnaire

def test_questionnaire_returns_three_diseases(client):
    r = client.post("/api/v1/predict/questionnaire", json=PERSON)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {x["disease"] for x in body["results"]} == {"hypertension", "diabetes", "ckd"}
    for res in body["results"]:
        assert 0 <= res["probability"] <= 1
        assert res["risk_level"] in {"low", "moderate", "high"}
        assert len(res["top_factors"]) == 5
    assert body["bmi"] == pytest.approx(30.0, abs=0.1)
    assert "not a medical diagnosis" in body["disclaimer"]


def test_optional_fields_can_be_omitted(client):
    person = {k: v for k, v in PERSON.items() if k not in ("waist_cm", "calories_kcal")}
    assert client.post("/api/v1/predict/questionnaire", json=person).status_code == 200


def test_healthier_profile_has_lower_risk(client):
    healthy = {**PERSON, "age": 30, "weight_kg": 70, "waist_cm": 82, "exercise_min_week": 300,
               "sitting_hours_day": 4, "sleep_hours": 7.5, "alcohol_drinks_week": 1, "smoking": "never"}
    risky = {r["disease"]: r["probability"]
             for r in client.post("/api/v1/predict/questionnaire", json=PERSON).json()["results"]}
    safe = {r["disease"]: r["probability"]
            for r in client.post("/api/v1/predict/questionnaire", json=healthy).json()["results"]}
    assert all(safe[d] < risky[d] for d in risky)


def test_diagnosed_condition_is_flagged(client):
    r = client.post("/api/v1/predict/questionnaire", json={**PERSON, "diagnosed_diabetes": True}).json()
    diabetes = next(x for x in r["results"] if x["disease"] == "diabetes")
    assert diabetes["already_diagnosed"] is True


@pytest.mark.parametrize("bad", [{"age": 12}, {"sex": "other"}, {"height_cm": 20},
                                 {"smoking": "sometimes"}, {"height_cm": 120, "weight_kg": 290}])
def test_invalid_questionnaire_is_rejected(client, bad):
    assert client.post("/api/v1/predict/questionnaire", json={**PERSON, **bad}).status_code == 422


# ---------------------------------------------------------------- anemia

@pytest.fixture(scope="module")
def nail_photo():
    from ml.anemia.data import RAW_DIR
    path = RAW_DIR / "photo" / "1.jpg"
    if not path.exists():
        pytest.skip("anemia dataset not downloaded")
    meta = next(l for l in (RAW_DIR / "metadata.csv").read_text().splitlines()[1:] if l.startswith("1,"))
    boxes = json.loads(meta.split(',"')[1].rstrip('"'))
    skins = json.loads(meta.split(',"')[2].rstrip('"'))
    to_xyxy = lambda bs: [[x1, y1, x2, y2] for y1, x1, y2, x2 in bs]  # noqa: E731
    return path.read_bytes(), to_xyxy(boxes), to_xyxy(skins)


def test_anemia_prediction(client, nail_photo):
    data, nails, skins = nail_photo
    r = client.post("/api/v1/predict/anemia", files={"image": ("hand.jpg", data, "image/jpeg")},
                    data={"nail_boxes": json.dumps(nails), "skin_boxes": json.dumps(skins), "sex": "female"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 3 < body["hemoglobin_g_dl"] < 22
    assert body["nails_used"] == 3
    assert body["heatmap_png"].startswith("data:image/png;base64,")


def _detector_available():
    from backend.app.config import MODEL_DIR
    return (MODEL_DIR / "anemia" / "nail_detector.onnx").exists()


def _jpeg(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.mark.skipif(not _detector_available(), reason="nail detector not exported")
def test_anemia_finds_nails_automatically(client, nail_photo):
    data, nails, skins = nail_photo
    auto = client.post("/api/v1/predict/anemia", files={"image": ("hand.jpg", data, "image/jpeg")},
                       data={"sex": "female"})
    manual = client.post("/api/v1/predict/anemia", files={"image": ("hand.jpg", data, "image/jpeg")},
                         data={"nail_boxes": json.dumps(nails), "skin_boxes": json.dumps(skins), "sex": "female"})
    assert auto.status_code == 200, auto.text
    assert auto.json()["nails_used"] == 3
    # detected boxes should give nearly the same estimate as the hand-drawn ones
    assert abs(auto.json()["hemoglobin_g_dl"] - manual.json()["hemoglobin_g_dl"]) < 1.0


@pytest.mark.skipif(not _detector_available(), reason="nail detector not exported")
def test_anemia_handles_large_portrait_phone_photo(client, nail_photo):
    rgb = np.asarray(Image.open(io.BytesIO(nail_photo[0])).convert("RGB"))
    phone = np.asarray(Image.fromarray(rgb).resize((3200, 2400), Image.LANCZOS).rotate(90, expand=True))
    landscape = client.post("/api/v1/predict/anemia", files={"image": ("a.jpg", nail_photo[0], "image/jpeg")})
    portrait = client.post("/api/v1/predict/anemia", files={"image": ("b.jpg", _jpeg(phone), "image/jpeg")})
    assert portrait.status_code == 200, portrait.text
    assert abs(portrait.json()["hemoglobin_g_dl"] - landscape.json()["hemoglobin_g_dl"]) < 1.0


@pytest.mark.skipif(not _detector_available(), reason="nail detector not exported")
def test_anemia_says_when_no_nails_are_found(client):
    noise = np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    r = client.post("/api/v1/predict/anemia", files={"image": ("n.jpg", _jpeg(noise), "image/jpeg")})
    assert r.status_code == 422
    assert "couldn't find your nails" in r.json()["detail"]


def test_anemia_rejects_bad_boxes(client, nail_photo):
    data, _, skins = nail_photo
    r = client.post("/api/v1/predict/anemia", files={"image": ("hand.jpg", data, "image/jpeg")},
                    data={"nail_boxes": "not json", "skin_boxes": json.dumps(skins)})
    assert r.status_code == 422


# ---------------------------------------------------------------- uploads

def test_rejects_non_image_upload(client):
    r = client.post("/api/v1/predict/anemia", files={"image": ("a.txt", b"hello", "text/plain")},
                    data={"nail_boxes": "[[0,0,50,50]]", "skin_boxes": "[[0,0,50,50]]"})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/v1/predict/anemia", files={"image": ("a.png", b"\x89PNG broken", "image/png")},
                    data={"nail_boxes": "[[0,0,50,50]]", "skin_boxes": "[[0,0,50,50]]"})
    assert r.status_code == 400


# ---------------------------------------------------------------- retina

def test_retina_rejects_non_fundus_photo(client):
    if not client.get("/health").json()["models"]["retina"]:
        pytest.skip("retina model not exported yet")
    selfie = np.full((400, 400, 3), 200, dtype=np.uint8)
    r = client.post("/api/v1/predict/retina", files={"image": ("x.png", png_bytes(selfie), "image/png")})
    assert r.status_code == 422
