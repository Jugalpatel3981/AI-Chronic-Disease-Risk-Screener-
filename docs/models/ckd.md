# Model card: Chronic Kidney Disease risk (questionnaire)

**Predicts:** eGFR <60 mL/min/1.73m² (CKD-EPI 2021) or urine albumin-to-creatinine ratio ≥30 mg/g (KDIGO).

**Inputs:** age, sex, height, weight, BMI, waist (optional), exercise, sitting time, sleep, alcohol, smoking, daily calories (optional), and whether the person has diagnosed high blood pressure or diabetes.
No lab tests are needed.

**Training data:** 17,650 US adults from NHANES 2011–2018 (17.1% prevalence). See [the dataset summary](../data/nhanes_dataset.md).

**Model:** LightGBM, picked from three candidates by 5-fold cross-validated AUROC. Probabilities are calibrated with isotonic regression on out-of-fold predictions.

## Model comparison

| Model | CV AUROC (mean ± SD) |
|---|---|
| Logistic Regression | 0.771 ± 0.010 |
| XGBoost | 0.783 ± 0.010 |
| LightGBM ✅ | 0.783 ± 0.010 |

![Model comparison](ckd/model_comparison.png)

## Performance

95% confidence intervals from 1,000 bootstrap resamples. The temporal test set (2021–2023) was collected after the training period and was never used for training or tuning.

| Metric | Internal test (2011–2018) | Temporal test (2021–2023) |
|---|---|---|
| People | 4,413 | 6,161 |
| Prevalence | 17.1% | 17.2% |
| AUROC | 0.789 (0.771–0.808) | 0.786 (0.771–0.802) |
| AUPRC | 0.454 (0.419–0.495) | 0.438 (0.410–0.468) |
| Brier score (lower is better) | 0.115 (0.108–0.121) | 0.117 (0.112–0.122) |
| Calibration error (ECE) | 0.008 (0.008–0.021) | 0.017 (0.012–0.027) |

**Screening threshold:** risk ≥ 11.8%, chosen on training data to catch at least 80% of cases.

| At the screening threshold | Internal test | Temporal test |
|---|---|---|
| Sensitivity (cases caught) | 80.5% (77.6–83.3) | 84.1% (81.9–86.4) |
| Specificity | 61.5% (60.0–63.1) | 54.1% (52.7–55.5) |
| PPV (flagged people who have it) | 30.1% (28.2–32.1) | 27.6% (26.1–29.2) |
| NPV (cleared people who don't) | 93.9% (92.9–94.8) | 94.3% (93.4–95.1) |
| People flagged | 45.6% | 52.4% |

![ROC curve](ckd/roc.png) ![Calibration](ckd/calibration.png)

## What drives the prediction (SHAP)

![SHAP importance](ckd/shap_importance.png)

## Fairness audit

AUROC by subgroup on the combined internal and temporal test sets. Ethnicity is **not** a model input; it is used only for this audit.

AUROC *within* an age band is naturally lower than overall: age is the strongest predictor, and inside a narrow band it can no longer separate people. The useful fairness check is the comparison between sex and ethnicity groups, and whether mean predicted risk tracks the observed rate in each group (calibration).

| Group | Subgroup | n | AUROC (95% CI) | Observed rate | Mean predicted |
|---|---|---|---|---|---|
| Sex | Female | 5,607 | 0.758 (0.741–0.777) | 16.6% | 18.1% |
| Sex | Male | 4,967 | 0.815 (0.800–0.831) | 17.8% | 16.9% |
| Age | Age 18–39 | 3,351 | 0.634 (0.592–0.668) | 6.2% | 6.7% |
| Age | Age 40–59 | 3,110 | 0.714 (0.687–0.741) | 12.0% | 12.4% |
| Age | Age 60+ | 4,113 | 0.743 (0.726–0.761) | 30.0% | 30.3% |
| Ethnicity | Mexican American | 1,003 | 0.755 (0.703–0.796) | 12.9% | 13.9% |
| Ethnicity | Non-Hispanic Asian | 919 | 0.753 (0.704–0.803) | 12.8% | 14.2% |
| Ethnicity | Non-Hispanic Black | 1,747 | 0.781 (0.752–0.805) | 21.8% | 18.6% |
| Ethnicity | Non-Hispanic White | 5,213 | 0.804 (0.790–0.818) | 17.8% | 18.9% |
| Ethnicity | Other / Multiracial | 584 | 0.786 (0.738–0.834) | 17.5% | 17.3% |
| Ethnicity | Other Hispanic | 1,108 | 0.737 (0.690–0.777) | 14.1% | 16.0% |

![Fairness](ckd/fairness.png)

**Calibration gaps (≥3 percentage points between observed and predicted):**

- The model under-estimates risk for **Non-Hispanic Black** adults (observed 21.8%, predicted 18.6%).

Ranking (AUROC) is similar across groups, but absolute risk is off for these groups. Possible fixes are group-specific recalibration or adding ethnicity as an input; both involve trade-offs and are left as future work.

## Limitations

- This is a screening estimate, not a diagnosis. Labels come from a single NHANES visit.
- It is trained on US adults; performance in other populations hasn't been validated.
- Lifestyle inputs are self-reported, so reporting errors reduce accuracy.
- It estimates whether the condition is present *now* (prevalence), not the risk of developing it in the future.
