# Data

Raw and processed data are **not committed** to git. Re-create them with the commands below, which download the official public files from the CDC.

## NHANES (tabular models)

The [National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/) is run by the US CDC. It combines interviews with physical exams and lab tests for a nationally representative sample of people.

Run these from the repository root:

```bash
python -m pip install -r ml/requirements.txt
python -m ml.nhanes.download        # ~190 MB of .xpt files -> data/raw/nhanes/
python -m ml.nhanes.build_dataset   # -> data/processed/nhanes_adults.parquet
python -m ml.nhanes.summarize       # -> docs/data/nhanes_dataset.md
```

- **Cycles used:** 2011–2012, 2013–2014, 2015–2016 and 2017–2018 for training; 2021–2023 as a held-out temporal test set.
- **Files:** demographics, body measures, blood pressure, HbA1c, serum creatinine, urine albumin/creatinine, and the activity, sleep, alcohol, smoking, diet, diabetes, kidney and medical-conditions questionnaires.
- **Labels:** derived from measured values using current clinical definitions (see [ml/nhanes/build_dataset.py](../ml/nhanes/build_dataset.py)).
- A summary with prevalence and missingness is in [docs/data/nhanes_dataset.md](../docs/data/nhanes_dataset.md).

## Layout

```
data/raw/        downloaded source files      (git-ignored)
data/processed/  cleaned, model-ready tables  (git-ignored)
```
