"""Request and response models (validated by Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Sex = Literal["male", "female"]
RiskLevel = Literal["low", "moderate", "high"]


class Questionnaire(BaseModel):
    """Answers a person can give without any lab test. Ranges reject typos;
    values outside the training data are clipped inside the model."""

    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: Sex
    height_cm: float = Field(..., ge=100, le=230)
    weight_kg: float = Field(..., ge=25, le=300)
    waist_cm: float | None = Field(None, ge=40, le=200, description="Optional")
    exercise_min_week: float = Field(..., ge=0, le=10080,
                                     description="Moderate or vigorous leisure exercise, minutes per week")
    sitting_hours_day: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    alcohol_drinks_week: float = Field(..., ge=0, le=200)
    smoking: Literal["never", "former", "current"]
    calories_kcal: float | None = Field(None, ge=300, le=10000, description="Optional")
    diagnosed_high_bp: bool = False
    diagnosed_diabetes: bool = False

    @model_validator(mode="after")
    def plausible_bmi(self):
        bmi = self.weight_kg / (self.height_cm / 100) ** 2
        if not 10 <= bmi <= 90:
            raise ValueError(f"height and weight give an implausible BMI of {bmi:.1f}")
        return self


class Factor(BaseModel):
    feature: str
    label: str
    value: float | None
    impact: float = Field(..., description="SHAP value in log-odds; positive raises risk")
    direction: Literal["raises", "lowers"]


class DiseaseRisk(BaseModel):
    disease: str
    title: str
    probability: float
    risk_level: RiskLevel
    screen_positive: bool
    threshold: float
    already_diagnosed: bool
    top_factors: list[Factor]
    definition: str
    model: str


class QuestionnaireResult(BaseModel):
    bmi: float
    results: list[DiseaseRisk]
    disclaimer: str


class AnemiaResult(BaseModel):
    hemoglobin_g_dl: float
    hemoglobin_range_g_dl: tuple[float, float]
    probability: float
    risk_level: RiskLevel
    threshold_g_dl: float
    sex_used: Sex | None
    nails_used: int
    heatmap_png: str = Field(..., description="Base64 PNG: nail crop with pallor heatmap")
    note: str
    disclaimer: str


class RetinaResult(BaseModel):
    grade: int
    grade_name: str
    probability_referable: float
    probability_any_dr: float
    risk_level: RiskLevel
    referable: bool
    heatmap_png: str = Field(..., description="Base64 PNG: fundus with attention heatmap")
    note: str
    disclaimer: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    models: dict[str, bool]
