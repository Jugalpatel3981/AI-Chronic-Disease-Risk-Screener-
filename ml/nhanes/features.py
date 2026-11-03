"""Pure functions that turn raw NHANES variables into clean features and labels.

Every function takes pandas Series/DataFrames of raw NHANES codes and returns
values in plain units (minutes per week, drinks per week, mL/min/1.73m2, ...).
Refused / don't-know codes become NaN. Kept free of file I/O so they can be
unit-tested.
"""

import numpy as np
import pandas as pd

WEEKS_PER_MONTH = 52.0 / 12.0


def clean_xpt_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """SAS XPT files store 0 as ~5.4e-79 when read by pandas; snap those to 0."""
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-10, 0.0)
    return df


def valid(series: pd.Series, low: float, high: float) -> pd.Series:
    """Keep values in [low, high]; everything else (7/9/77/99/777/... codes) -> NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= low) & (s <= high))


def yes_no(series: pd.Series) -> pd.Series:
    """NHANES yes/no coding: 1 = yes -> 1.0, 2 = no -> 0.0, anything else -> NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.map({1: 1.0, 2: 0.0})


# ---------------------------------------------------------------- activity

def _domain_minutes_week(did: pd.Series, days: pd.Series, minutes: pd.Series) -> pd.Series:
    """Minutes/week for one activity domain (2011-2018 PAQ layout).

    did: 1 = yes, 2 = no. days: 1-7 per week. minutes: per day (10-1440).
    """
    did_ = yes_no(did)
    total = valid(days, 1, 7) * valid(minutes, 1, 1440)
    return total.where(did_ == 1, np.where(did_ == 0, 0.0, np.nan))


def leisure_activity_2011_2018(paq: pd.DataFrame) -> pd.DataFrame:
    vig = _domain_minutes_week(paq["PAQ650"], paq["PAQ655"], paq["PAD660"])
    mod = _domain_minutes_week(paq["PAQ665"], paq["PAQ670"], paq["PAD675"])
    return pd.DataFrame({"vigorous_min_week": vig, "moderate_min_week": mod})


_FREQ_UNIT_PER_WEEK = {"D": 7.0, "W": 1.0, "M": 1.0 / WEEKS_PER_MONTH, "Y": 1.0 / 52.0}


def _decode(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v.decode() if isinstance(v, bytes) else v)


def _sessions_per_week(freq: pd.Series, unit: pd.Series) -> pd.Series:
    f = valid(freq, 0, 7000)  # 7777 / 9999 = refused / don't know
    u = _decode(unit).map(_FREQ_UNIT_PER_WEEK)
    out = f * u
    return out.where(f != 0, 0.0)  # 0 sessions needs no unit


def leisure_activity_2021_2023(paq: pd.DataFrame) -> pd.DataFrame:
    mod = _sessions_per_week(paq["PAD790Q"], paq["PAD790U"]) * valid(paq["PAD800"], 1, 1440)
    vig = _sessions_per_week(paq["PAD810Q"], paq["PAD810U"]) * valid(paq["PAD820"], 1, 1440)
    mod = mod.where(valid(paq["PAD790Q"], 0, 7000) != 0, 0.0)
    vig = vig.where(valid(paq["PAD810Q"], 0, 7000) != 0, 0.0)
    return pd.DataFrame({"vigorous_min_week": vig, "moderate_min_week": mod})


# ---------------------------------------------------------------- alcohol

# ALQ121 (2017+) frequency categories -> drinking days per week.
_ALQ121_DAYS_PER_WEEK = {
    0: 0.0,                       # never in the last year
    1: 7.0,                       # every day
    2: 5.5,                       # nearly every day
    3: 3.5,                       # 3-4 times a week
    4: 2.0,                       # 2 times a week
    5: 1.0,                       # once a week
    6: 2.5 / WEEKS_PER_MONTH,     # 2-3 times a month
    7: 1.0 / WEEKS_PER_MONTH,     # once a month
    8: 9.0 / 52.0,                # 7-11 times a year
    9: 4.5 / 52.0,                # 3-6 times a year
    10: 1.5 / 52.0,               # 1-2 times a year
}


def drinks_per_week_2011_2016(alq: pd.DataFrame) -> pd.Series:
    """ALQ120Q/U = how often drank in past year, ALQ130 = drinks per drinking day."""
    freq = valid(alq["ALQ120Q"], 0, 365)
    unit = pd.to_numeric(alq["ALQ120U"], errors="coerce").map(
        {1: 1.0, 2: 1.0 / WEEKS_PER_MONTH, 3: 1.0 / 52.0}
    )
    per_day = valid(alq["ALQ130"], 1, 95)
    drinks = freq * unit * per_day
    drinks = drinks.where(freq != 0, 0.0)
    # Fewer than 12 drinks in a year AND in a lifetime: questionnaire skips -> 0.
    never = (yes_no(alq["ALQ101"]) == 0) & (yes_no(alq["ALQ110"]) == 0)
    return drinks.where(~never, 0.0)


def drinks_per_week_2017_plus(alq: pd.DataFrame) -> pd.Series:
    days = pd.to_numeric(alq["ALQ121"], errors="coerce").map(_ALQ121_DAYS_PER_WEEK)
    per_day = valid(alq["ALQ130"], 1, 95)
    drinks = days * per_day
    drinks = drinks.where(days != 0, 0.0)
    never = yes_no(alq["ALQ111"]) == 0  # never had a drink of alcohol
    return drinks.where(~never, 0.0)


# ---------------------------------------------------------------- smoking

def smoking_status(smq: pd.DataFrame) -> pd.Series:
    """0 = never (<100 cigarettes in life), 1 = former, 2 = current."""
    ever = yes_no(smq["SMQ020"])
    now = pd.to_numeric(smq["SMQ040"], errors="coerce")
    status = pd.Series(np.nan, index=smq.index)
    status[ever == 0] = 0
    status[(ever == 1) & (now == 3)] = 1
    status[(ever == 1) & now.isin([1, 2])] = 2
    return status


# ---------------------------------------------------------------- blood pressure

def mean_bp(df: pd.DataFrame, sys_cols: list[str], dia_cols: list[str]) -> pd.DataFrame:
    sbp = df[sys_cols].apply(lambda c: valid(c, 60, 300)).mean(axis=1)
    # A diastolic reading of 0 is a measurement artefact, not a real value.
    dbp = df[dia_cols].apply(lambda c: valid(c, 30, 200)).mean(axis=1)
    return pd.DataFrame({"sbp_mean": sbp, "dbp_mean": dbp})


# ---------------------------------------------------------------- kidney

def egfr_ckd_epi_2021(creatinine_mg_dl: pd.Series, age: pd.Series, male: pd.Series) -> pd.Series:
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73m2)."""
    scr = pd.to_numeric(creatinine_mg_dl, errors="coerce")
    is_male = male.astype(float) == 1
    kappa = np.where(is_male, 0.9, 0.7)
    alpha = np.where(is_male, -0.302, -0.241)
    ratio = scr / kappa
    egfr = (
        142
        * np.minimum(ratio, 1.0) ** alpha
        * np.maximum(ratio, 1.0) ** -1.200
        * 0.9938 ** age.astype(float)
        * np.where(is_male, 1.0, 1.012)
    )
    return pd.Series(egfr, index=creatinine_mg_dl.index).where(scr > 0)
