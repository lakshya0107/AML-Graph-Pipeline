"""Project-wide configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", ROOT_DIR / "data"))
ARTIFACT_DIR = Path(os.environ.get("AML_ARTIFACT_DIR", ROOT_DIR / "artifacts"))

LEDGER_PATH = DATA_DIR / "ledger.csv"
MODEL_PATH = ARTIFACT_DIR / "aml_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Currencies the pipeline understands, with static FX rates into USD.
CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "INR")
FX_TO_USD: dict[str, float] = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "INR": 0.012}

# Cash-transaction style reporting threshold used for structuring detection.
REPORTING_THRESHOLD_USD = 10_000.0
STRUCTURING_BAND = 0.90  # amounts in [90%, 100%) of the threshold look structured

# Graph features are computed over rolling windows of this many days.
GRAPH_WINDOW_DAYS = 7

RANDOM_SEED = 42
