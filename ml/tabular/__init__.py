"""Questionnaire-based risk models.

LightGBM must be imported before XGBoost. On Windows the two libraries ship
different OpenMP runtimes, and loading XGBoost first makes LightGBM crash with
an access violation when both models run in one process (as they do in the
API). Importing this package guarantees the safe order.
"""

import lightgbm  # noqa: F401  (must come before xgboost)
import xgboost  # noqa: F401
