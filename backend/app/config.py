"""Runtime settings, read from environment variables."""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BACKEND_DIR / "models"))

# Comma-separated list of frontend origins allowed to call the API.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

API_VERSION = "1.0.0"
DISCLAIMER = ("This is an AI screening estimate for education and research. It is not a medical "
              "diagnosis. Please consult a healthcare professional about your health.")
