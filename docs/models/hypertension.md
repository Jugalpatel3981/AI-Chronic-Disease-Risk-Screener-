# Model card: Hypertension risk (questionnaire)

**Predicts:** Mean measured blood pressure ≥130/80 mmHg or taking blood-pressure medication (2017 ACC/AHA guideline).

**Inputs:** age, sex, height, weight, BMI, waist (optional), exercise, sitting time, sleep, alcohol, smoking, daily calories (optional), and whether the person has diagnosed diabetes.
No lab tests are needed.

**Training data:** 17,296 US adults from NHANES 2011–2018 (50.3% prevalence). See [the dataset summary](../data/nhanes_dataset.md).

**Model:** XGBoost, picked from three candidates by 5-fold cross-validated AUROC. Probabilities are calibrated with isotonic regression on out-of-fold predictions.

## Model comparison

| Model | CV AUROC (mean ± SD) |
|---|---|
| Logistic Regression | 0.824 ± 0.005 |
| XGBoost ✅ | 0.826 ± 0.006 |
| LightGBM | 0.826 ± 0.006 |

![Model comparison](hypertension/model_comparison.png)

## Performance

95% confidence intervals from 1,000 bootstrap resamples. The temporal test set (2021–2023) was collected after the training period and was never used for training or tuning.

| Metric | Internal test (2011–2018) | Temporal test (2021–2023) |
|---|---|---|
| People | 4,325 | 6,094 |
| Prevalence | 50.3% | 53.6% |
| AUROC | 0.822 (0.809–0.835) | 0.789 (0.779–0.801) |
| AUPRC | 0.798 (0.780–0.815) | 0.776 (0.760–0.791) |
| Brier score (lower is better) | 0.171 (0.165–0.178) | 0.185 (0.180–0.191) |
| Calibration error (ECE) | 0.013 (0.012–0.030) | 0.034 (0.025–0.046) |

**Screening threshold:** risk ≥ 47.5%, chosen on training data to catch at least 80% of cases.

| At the screening threshold | Internal test | Temporal test |
|---|---|---|
| Sensitivity (cases caught) | 83.4% (81.8–85.0) | 86.9% (85.7–88.1) |
| Specificity | 66.1% (63.8–68.1) | 55.7% (54.0–57.6) |
| PPV (flagged people who have it) | 71.3% (69.6–73.1) | 69.4% (67.9–70.9) |
| NPV (cleared people who don't) | 79.7% (77.8–81.7) | 78.6% (76.9–80.4) |
| People flagged | 58.8% | 67.1% |

![ROC curve](hypertension/roc.png) ![Calibration](hypertension/calibration.png)

## What drives the prediction (SHAP)

![SHAP importance](hypertension/shap_importance.png)

## Fairness audit

AUROC by subgroup on the combined internal and temporal test sets. Ethnicity is **not** a model input; it is used only for this audit.

AUROC *within* an age band is naturally lower than overall: age is the strongest predictor, and inside a narrow band it can no longer separate people. The useful fairness check is the comparison between sex and ethnicity groups, and whether mean predicted risk tracks the observed rate in each group (calibration).

| Group | Subgroup | n | AUROC (95% CI) | Observed rate | Mean predicted |
|---|---|---|---|---|---|
| Sex | Female | 5,526 | 0.820 (0.810–0.830) | 49.3% | 51.6% |
| Sex | Male | 4,893 | 0.782 (0.770–0.794) | 55.6% | 56.2% |
| Age | Age 18–39 | 3,234 | 0.760 (0.741–0.779) | 22.1% | 21.0% |
| Age | Age 40–59 | 3,063 | 0.707 (0.689–0.726) | 53.9% | 55.3% |
| Age | Age 60+ | 4,122 | 0.652 (0.633–0.670) | 74.7% | 78.3% |
| Ethnicity | Mexican American | 1,017 | 0.822 (0.798–0.847) | 42.2% | 47.8% |
| Ethnicity | Non-Hispanic Asian | 902 | 0.805 (0.777–0.834) | 43.7% | 41.9% |
| Ethnicity | Non-Hispanic Black | 1,710 | 0.833 (0.816–0.853) | 62.2% | 54.2% |
| Ethnicity | Non-Hispanic White | 5,146 | 0.784 (0.771–0.798) | 53.8% | 57.9% |
| Ethnicity | Other / Multiracial | 553 | 0.814 (0.775–0.850) | 49.4% | 51.5% |
| Ethnicity | Other Hispanic | 1,091 | 0.832 (0.808–0.858) | 47.4% | 50.0% |

![Fairness](hypertension/fairness.png)

**Calibration gaps (≥3 percentage points between observed and predicted):**

- The model over-estimates risk for **Mexican American** adults (observed 42.2%, predicted 47.8%).
- The model under-estimates risk for **Non-Hispanic Black** adults (observed 62.2%, predicted 54.2%).
- The model over-estimates risk for **Non-Hispanic White** adults (observed 53.8%, predicted 57.9%).

Ranking (AUROC) is similar across groups, but absolute risk is off for these groups. Possible fixes are group-specific recalibration or adding ethnicity as an input; both involve trade-offs and are left as future work.

## Limitations

- This is a screening estimate, not a diagnosis. Labels come from a single NHANES visit.
- It is trained on US adults; performance in other populations hasn't been validated.
- Lifestyle inputs are self-reported, so reporting errors reduce accuracy.
- It estimates whether the condition is present *now* (prevalence), not the risk of developing it in the future.
