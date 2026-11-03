"""Build the per-person adult NHANES dataset with screening features and labels.

Usage (from the repository root, after `python -m ml.nhanes.download`):
    python -m ml.nhanes.build_dataset

Output: data/processed/nhanes_adults.parquet, one row per examined adult.

Labels (current clinical definitions, applied to measured values):
    hypertension  mean SBP >= 130 or DBP >= 80 mmHg, or taking BP medication
                  (2017 ACC/AHA guideline)
    hypertension_stage2  mean SBP >= 140 or DBP >= 90, or taking BP medication
    diabetes      HbA1c >= 6.5%, or told by a doctor, or on insulin / pills (ADA)
    prediabetes   HbA1c 5.7-6.4% or told prediabetes, and not diabetic
    ckd           eGFR < 60 (CKD-EPI 2021) or urine albumin/creatinine >= 30 mg/g
                  (KDIGO)
"""

import sys

import pandas as pd

from . import features as F
from .config import BP_FILE, CYCLES, PROCESSED_DIR, RAW_DIR

TEMPORAL_TEST_CYCLE = "2021-2023"


def read_xpt(cycle: str, name: str, suffix: str) -> pd.DataFrame:
    path = RAW_DIR / cycle / f"{name}_{suffix}.xpt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m ml.nhanes.download` first")
    df = pd.read_sas(path, format="xport")
    df["SEQN"] = df["SEQN"].astype(int)
    return F.clean_xpt_zeros(df).set_index("SEQN")


def build_cycle(cycle: str, suffix: str) -> pd.DataFrame:
    r = lambda name: read_xpt(cycle, name, suffix)  # noqa: E731

    demo, bmx = r("DEMO"), r("BMX")
    out = pd.DataFrame(index=demo.index)
    out["cycle"] = cycle
    out["survey_weight"] = demo["WTMEC2YR"]
    out["age"] = F.valid(demo["RIDAGEYR"], 0, 80)  # 80 = "80 or older"
    out["sex_male"] = pd.to_numeric(demo["RIAGENDR"]).map({1: 1, 2: 0})
    out["ethnicity"] = pd.to_numeric(demo["RIDRETH3"]).map({
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other / Multiracial",
    })
    out["pregnant"] = pd.to_numeric(demo.get("RIDEXPRG"), errors="coerce") == 1

    # ---- body measures
    out = out.join(bmx[["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]].rename(columns={
        "BMXWT": "weight_kg", "BMXHT": "height_cm", "BMXBMI": "bmi", "BMXWAIST": "waist_cm",
    }))

    # ---- lifestyle
    paq = r("PAQ")
    act = (F.leisure_activity_2021_2023(paq) if suffix == "L"
           else F.leisure_activity_2011_2018(paq))
    out = out.join(act)
    out["exercise_min_week"] = out["moderate_min_week"] + out["vigorous_min_week"]
    out["sedentary_min_day"] = F.valid(paq["PAD680"], 0, 1440)

    slq = r("SLQ")
    sleep_col = "SLD010H" if suffix in ("G", "H") else "SLD012"
    out["sleep_hours"] = F.valid(slq[sleep_col], 2, 14)

    alq = r("ALQ")
    out["alcohol_drinks_week"] = (F.drinks_per_week_2017_plus(alq) if suffix in ("J", "L")
                                  else F.drinks_per_week_2011_2016(alq))
    out["smoking_status"] = F.smoking_status(r("SMQ"))
    out["calories_kcal"] = F.valid(r("DR1TOT")["DR1TKCAL"], 300, 15000)

    mcq = r("MCQ")
    out["family_history_diabetes"] = (F.yes_no(mcq["MCQ300C"]) if "MCQ300C" in mcq
                                      else pd.Series(float("nan"), index=mcq.index))

    # ---- self-reported diagnoses (known to the user; optional model inputs)
    bpq, diq, kiq = r("BPQ"), r("DIQ"), r("KIQ_U")
    out["told_high_bp"] = F.yes_no(bpq["BPQ020"])
    out["told_diabetes"] = F.yes_no(diq["DIQ010"])  # 3 = borderline -> NaN
    out["told_prediabetes"] = F.yes_no(diq["DIQ160"])
    out["told_weak_kidneys"] = F.yes_no(kiq["KIQ022"])

    # ---- clinical measurements (used for labels only, never as model inputs)
    bp = r(BP_FILE[suffix])
    if suffix == "L":
        out = out.join(F.mean_bp(bp, [f"BPXOSY{i}" for i in (1, 2, 3)],
                                 [f"BPXODI{i}" for i in (1, 2, 3)]))
        bp_med = F.yes_no(bpq["BPQ150"])
    else:
        out = out.join(F.mean_bp(bp, [f"BPXSY{i}" for i in (1, 2, 3, 4)],
                                 [f"BPXDI{i}" for i in (1, 2, 3, 4)]))
        bp_med = F.yes_no(bpq["BPQ050A"])
    out["bp_medication"] = bp_med.reindex(out.index).fillna(0)  # not asked -> not taking

    out["hba1c"] = F.valid(r("GHB")["LBXGH"], 3, 20)
    out["insulin"] = F.yes_no(diq["DIQ050"])
    out["diabetes_pills"] = F.yes_no(diq["DIQ070"])
    out["creatinine_mg_dl"] = F.valid(r("BIOPRO")["LBXSCR"], 0.1, 20)
    out["uacr_mg_g"] = F.valid(r("ALB_CR")["URDACT"], 0, 100000)
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    sbp, dbp, med = df["sbp_mean"], df["dbp_mean"], df["bp_medication"] == 1
    have_bp = sbp.notna() & dbp.notna()

    df["label_hypertension"] = ((sbp >= 130) | (dbp >= 80) | med).astype(float)
    df.loc[~have_bp & ~med, "label_hypertension"] = float("nan")
    df["label_hypertension_stage2"] = ((sbp >= 140) | (dbp >= 90) | med).astype(float)
    df.loc[~have_bp & ~med, "label_hypertension_stage2"] = float("nan")

    diag = (df["told_diabetes"] == 1) | (df["insulin"] == 1) | (df["diabetes_pills"] == 1)
    df["label_diabetes"] = ((df["hba1c"] >= 6.5) | diag).astype(float)
    df.loc[df["hba1c"].isna() & ~diag, "label_diabetes"] = float("nan")
    df["undiagnosed_diabetes"] = ((df["hba1c"] >= 6.5) & ~diag).astype(float)
    pre = (df["hba1c"].between(5.7, 6.45)) | (df["told_prediabetes"] == 1)
    df["label_prediabetes"] = (pre & (df["label_diabetes"] == 0)).astype(float)

    df["egfr"] = F.egfr_ckd_epi_2021(df["creatinine_mg_dl"], df["age"], df["sex_male"])
    have_kidney = df["egfr"].notna() | df["uacr_mg_g"].notna()
    df["label_ckd"] = ((df["egfr"] < 60) | (df["uacr_mg_g"] >= 30)).astype(float)
    df.loc[~have_kidney, "label_ckd"] = float("nan")
    return df


def build() -> pd.DataFrame:
    parts = []
    for cycle, (_, suffix) in CYCLES.items():
        print(f"[build] {cycle}")
        parts.append(build_cycle(cycle, suffix))
    df = pd.concat(parts)
    df.index.name = "seqn"

    # Study population: examined, non-pregnant adults.
    n0 = len(df)
    df = df[(df["age"] >= 18) & df["weight_kg"].notna() & ~df["pregnant"]].drop(columns="pregnant")
    print(f"[build] kept {len(df):,} examined non-pregnant adults out of {n0:,} participants")

    df = add_labels(df)
    df["split"] = (df["cycle"] == TEMPORAL_TEST_CYCLE).map({True: "temporal_test", False: "train"})

    # Pooled survey weights: divide by the number of pooled cycles per split.
    n_train_cycles = len(CYCLES) - 1
    df.loc[df["split"] == "train", "survey_weight"] /= n_train_cycles
    return df.reset_index()


def main() -> int:
    df = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "nhanes_adults.parquet"
    df.to_parquet(path, index=False)
    print(f"[build] saved {len(df):,} rows x {df.shape[1]} columns -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
