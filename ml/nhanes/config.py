"""NHANES download configuration: which survey cycles and data files to use."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nhanes"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.xpt"

# NHANES survey cycles: label -> (start year used in the URL, file suffix).
# 2011-2018 are used for training/validation; 2021-2023 is held out as a
# temporal test set (data collected after the training period).
CYCLES = {
    "2011-2012": (2011, "G"),
    "2013-2014": (2013, "H"),
    "2015-2016": (2015, "I"),
    "2017-2018": (2017, "J"),
    "2021-2023": (2021, "L"),
}

# Data files needed from each cycle.
FILES = [
    "DEMO",    # demographics: age, sex, ethnicity, survey weights
    "BMX",     # body measures: weight, height, BMI, waist
    "BPQ",     # blood pressure questionnaire: diagnosis, medication
    "DIQ",     # diabetes questionnaire: diagnosis, insulin, pills
    "GHB",     # HbA1c lab result
    "BIOPRO",  # serum biochemistry: creatinine
    "ALB_CR",  # urine albumin-to-creatinine ratio
    "KIQ_U",   # kidney conditions questionnaire
    "PAQ",     # physical activity questionnaire
    "SLQ",     # sleep questionnaire
    "ALQ",     # alcohol questionnaire
    "SMQ",     # smoking questionnaire
    "DR1TOT",  # 24-hour dietary recall totals: calories
    "MCQ",     # medical conditions: family history of diabetes
]

# Blood pressure exam file differs by cycle: auscultatory (BPX) up to
# 2017-2018, oscillometric (BPXO) from 2021-2023.
BP_FILE = {"G": "BPX", "H": "BPX", "I": "BPX", "J": "BPX", "L": "BPXO"}
