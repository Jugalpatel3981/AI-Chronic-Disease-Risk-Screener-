"""AI Chronic Disease Risk Screener API.

Run locally (from the repository root):
    uvicorn backend.app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import ALLOWED_ORIGINS, API_VERSION, DISCLAIMER, MODEL_DIR
from .images import read_image
from .schemas import AnemiaResult, Health, Questionnaire, QuestionnaireResult, RetinaResult, Sex

log = logging.getLogger("screener")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

services: dict[str, object] = {}


def _load(name: str, factory):
    try:
        t = time.perf_counter()
        services[name] = factory(MODEL_DIR)
        log.info("loaded %s model in %.1fs", name, time.perf_counter() - t)
    except FileNotFoundError as exc:
        log.warning("%s model not available: %s", name, exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .models.anemia import AnemiaService
    from .models.retina import RetinaService
    from .models.tabular import TabularService
    _load("questionnaire", TabularService)
    _load("anemia", AnemiaService)
    _load("retina", RetinaService)
    yield
    services.clear()


app = FastAPI(
    title="AI Chronic Disease Risk Screener API",
    version=API_VERSION,
    description="Screening estimates for hypertension, type 2 diabetes, chronic kidney disease, "
                "anemia (fingernail photo) and diabetic retinopathy (retina photo). " + DISCLAIMER,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.0f}"
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, ms)
    return response


def service(name: str):
    if name not in services:
        raise HTTPException(503, f"The {name} model is not available on this server.")
    return services[name]


def parse_boxes(raw: str, name: str) -> list[tuple[int, int, int, int]]:
    try:
        boxes = json.loads(raw)
        assert isinstance(boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in boxes)
        return [tuple(float(v) for v in b) for b in boxes]
    except (ValueError, AssertionError, TypeError):
        raise HTTPException(422, f"{name} must be a JSON list of [x1, y1, x2, y2] boxes.") from None


# ------------------------------------------------------------------ routes

@app.get("/health", response_model=Health, tags=["system"])
def health():
    loaded = {k: k in services for k in ("questionnaire", "anemia", "retina")}
    return Health(status="ok" if all(loaded.values()) else "degraded", version=API_VERSION, models=loaded)


@app.post("/api/v1/predict/questionnaire", response_model=QuestionnaireResult, tags=["predict"],
          summary="Hypertension, diabetes and CKD risk from a short questionnaire")
async def predict_questionnaire(q: Questionnaire):
    bmi, results = await run_in_threadpool(service("questionnaire").predict, q)
    return QuestionnaireResult(bmi=round(bmi, 1), results=results, disclaimer=DISCLAIMER)


@app.post("/api/v1/predict/anemia", response_model=AnemiaResult, tags=["predict"],
          summary="Anemia screening from a fingernail photo (nails are found automatically)")
async def predict_anemia(
    image: Annotated[UploadFile, File(description="Photo of fingers with clearly visible nails")],
    sex: Annotated[Sex | None, Form()] = None,
    nail_boxes: Annotated[str | None, Form(description="Optional JSON list of 1-3 nail boxes "
                                                       "[[x1,y1,x2,y2], ...]; found automatically if omitted")] = None,
    skin_boxes: Annotated[str | None, Form(description="Optional JSON list of 1-3 skin boxes next to the nails")] = None,
):
    svc = service("anemia")
    rgb = await read_image(image)
    nails = parse_boxes(nail_boxes, "nail_boxes") if nail_boxes else None
    skins = parse_boxes(skin_boxes, "skin_boxes") if skin_boxes else None
    out = await run_in_threadpool(svc.predict, rgb, nails, skins, sex)
    return AnemiaResult(**out, disclaimer=DISCLAIMER)


@app.post("/api/v1/predict/retina", response_model=RetinaResult, tags=["predict"],
          summary="Diabetic retinopathy grading from a retina (fundus) photo")
async def predict_retina(image: Annotated[UploadFile, File(description="Fundus photograph")]):
    svc = service("retina")
    rgb = await read_image(image)
    out = await run_in_threadpool(svc.predict, rgb)
    return RetinaResult(**out, disclaimer=DISCLAIMER)


@app.exception_handler(ValidationError)
async def validation_error(_: Request, exc: ValidationError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
