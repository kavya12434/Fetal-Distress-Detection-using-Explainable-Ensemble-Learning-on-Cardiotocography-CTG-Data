"""Cached loaders for models, training stats, and precomputed artifacts.

Everything here is memoised with lru_cache so files are read from disk once.
The project's src/ directory is placed on sys.path so that pickled objects
(e.g. the ensemble's BalancedXGBClassifier from src/ensemble.py) resolve on
unpickling.
"""

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ROOT / "data" / "processed"

# Make src importable so pickles (ensemble.BalancedXGBClassifier) deserialize,
# and so constants.FEATURES / CLASS_NAMES are available.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from constants import CLASS_NAMES, FEATURES  # noqa: E402

INDEX_COL = "orig_index"


@lru_cache(maxsize=None)
def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_x_train() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "X_train.csv").set_index(INDEX_COL)
    df.index.name = None
    return df[FEATURES]


@lru_cache(maxsize=1)
def feature_stats() -> dict:
    """Per-feature min / max / median from the TRAINING set (programmatic bounds)."""
    X = load_x_train()
    stats = {}
    for feat in FEATURES:
        col = X[feat]
        stats[feat] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
        }
    return stats


@lru_cache(maxsize=1)
def test_results() -> list[dict]:
    df = pd.read_csv(ARTIFACTS_DIR / "test_results.csv")
    return df.where(pd.notnull(df), None).to_dict(orient="records")


@lru_cache(maxsize=1)
def metrics_json() -> dict:
    import json

    with open(ARTIFACTS_DIR / "metrics.json") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def shap_ranking() -> list[dict]:
    df = pd.read_csv(ARTIFACTS_DIR / "shap_feature_ranking.csv")
    return df.to_dict(orient="records")


@lru_cache(maxsize=1)
def tree_ranking() -> list[dict]:
    df = pd.read_csv(ARTIFACTS_DIR / "shap_treeexplainer_ranking.csv")
    return df.to_dict(orient="records")


@lru_cache(maxsize=1)
def example_cases() -> dict:
    return joblib.load(ARTIFACTS_DIR / "example_cases.pkl")


# ---- FC4: calibration + abstention artifacts ----

@lru_cache(maxsize=1)
def calibration_json() -> dict:
    """Test-set calibration report written by src/evaluate_calibration.py."""
    import json

    with open(ARTIFACTS_DIR / "calibration.json") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def abstention_json() -> dict:
    """Chosen abstention threshold and its stats, from src/abstention.py."""
    import json

    with open(ARTIFACTS_DIR / "abstention.json") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def abstention_sweep() -> list[dict]:
    """Full tau sweep (one row per threshold) from src/abstention.py."""
    df = pd.read_csv(ARTIFACTS_DIR / "abstention_sweep.csv")
    return df.where(pd.notnull(df), None).to_dict(orient="records")
