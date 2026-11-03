"""Model inputs, labels and settings for the questionnaire-based risk models."""

from ml.nhanes.config import PROCESSED_DIR, REPO_ROOT

DATASET_PATH = PROCESSED_DIR / "nhanes_adults.parquet"
ARTIFACT_DIR = REPO_ROOT / "backend" / "models" / "tabular"
DOCS_DIR = REPO_ROOT / "docs" / "models"

SEED = 42
CV_FOLDS = 5
OPTUNA_TRIALS = 25
TEST_FRACTION = 0.2          # internal held-out test set (from 2011-2018)
TARGET_SENSITIVITY = 0.80    # screening threshold: catch at least 80% of cases
BOOTSTRAP_ROUNDS = 1000

# Inputs a person can answer without any lab test. Ethnicity is not a model
# input; it is only used to audit fairness.
BASE_FEATURES = [
    "age",
    "sex_male",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",              # optional in the app (NaN allowed)
    "exercise_min_week",
    "sedentary_min_day",
    "sleep_hours",
    "alcohol_drinks_week",
    "smoking_status",
    "calories_kcal",         # optional in the app (NaN allowed)
]

# Extreme self-reported values are clipped to plausible ranges, both in
# training and in the API, so a typo cannot push the model off its data.
CLIP_RANGES = {
    "age": (18, 80),
    "height_cm": (130, 210),
    "weight_kg": (30, 250),
    "bmi": (12, 70),
    "waist_cm": (50, 180),
    "exercise_min_week": (0, 2520),   # up to 6 hours a day
    "sedentary_min_day": (0, 1200),
    "sleep_hours": (2, 14),
    "alcohol_drinks_week": (0, 70),
    "calories_kcal": (500, 6000),
}

# Each disease may also use answers to "has a doctor ever told you that you
# have ...?" for *other* conditions. A model never sees a diagnosis of its
# own target disease.
DISEASES = {
    "hypertension": {
        "label": "label_hypertension",
        "title": "Hypertension",
        "definition": "Mean measured blood pressure ≥130/80 mmHg or taking blood-pressure "
                      "medication (2017 ACC/AHA guideline)",
        "extra_features": ["told_diabetes"],
    },
    "diabetes": {
        "label": "label_diabetes",
        "title": "Type 2 Diabetes",
        "definition": "HbA1c ≥6.5%, diagnosed by a doctor, or taking insulin / diabetes pills "
                      "(ADA criteria)",
        "extra_features": ["told_high_bp"],
    },
    "ckd": {
        "label": "label_ckd",
        "title": "Chronic Kidney Disease",
        "definition": "eGFR <60 mL/min/1.73m² (CKD-EPI 2021) or urine albumin-to-creatinine "
                      "ratio ≥30 mg/g (KDIGO)",
        "extra_features": ["told_high_bp", "told_diabetes"],
    },
}

FEATURE_LABELS = {
    "age": "Age",
    "sex_male": "Sex (male)",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "BMI",
    "waist_cm": "Waist size",
    "exercise_min_week": "Exercise per week",
    "sedentary_min_day": "Sitting time per day",
    "sleep_hours": "Sleep",
    "alcohol_drinks_week": "Alcohol per week",
    "smoking_status": "Smoking",
    "calories_kcal": "Daily calories",
    "told_high_bp": "Diagnosed high blood pressure",
    "told_diabetes": "Diagnosed diabetes",
}


def features_for(disease: str) -> list[str]:
    return BASE_FEATURES + DISEASES[disease]["extra_features"]
