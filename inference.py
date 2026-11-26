"""Inference + SHAP logic, initialised once at startup.

The InferenceEngine holds the primary model (ensemble_softmax) and a
KernelExplainer built over a kmeans summary of the training set. It is created
in main.py's lifespan and reused for every request -- models and the explainer
are never rebuilt per call.
"""

import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message="The `probability` parameter was deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

import numpy as np
import pandas as pd
import shap

from artifacts import (
    abstention_json,
    example_cases,
    load_model,
    load_x_train,
)
from constants import CLASS_NAMES, FEATURES

PRIMARY_MODEL = "ensemble_softmax"
N_BACKGROUND = 25
NSAMPLES = 200
SEED = 42
# Disable KernelExplainer's default AIC-based L1 feature selection. At
# nsamples=200 (>> the 21 features) the system is over-determined, so l1_reg=0
# yields the exact weighted least-squares Shapley solution. With the default
# l1_reg="auto", ~11 of 21 features are shrunk to exactly 0.0 (incl.
# histogram_mean) regardless of nsamples -- raising nsamples alone does NOT fix
# it; the L1 selection is the actual cause.
L1_REG = 0


class CounterfactualService:
    """Wraps src/counterfactual.py's engine for request-time use.

    The underlying model is only read from, but the search mutates numpy buffers
    it allocates per call and the engine object is shared across the threadpool,
    so the same lock discipline used for SHAP is applied here.
    """

    def __init__(self):
        from counterfactual import CounterfactualEngine

        self._engine = CounterfactualEngine()
        self._lock = threading.Lock()
        self.tau = self._engine.tau
        self.variant = self._engine.variant
        self.model_file = self._engine.model_file

    def warmup(self) -> float:
        """One throwaway search so the first request does not pay cold cost."""
        patho = example_cases()["Pathological"]["feature_values"]
        row = {f: float(patho[f]) for f in FEATURES}
        t0 = time.perf_counter()
        self.generate(row, target_class=None, k=1)
        return (time.perf_counter() - t0) * 1000.0

    def generate(self, row: dict, target_class=None, k: int = 3):
        import numpy as np

        with self._lock:
            x0 = np.array([float(row[f]) for f in FEATURES], dtype=float)
            p0 = self._engine._proba(x0.reshape(1, -1))[0]
            orig = int(np.argmax(p0))
            cfs, message = self._engine.generate_counterfactual(
                row, target_class=target_class, k=k
            )

        if target_class is None:
            resolved = orig - 1 if orig > 0 else None
        else:
            resolved = int(target_class)

        payload = {
            "original_class": orig,
            "original_label": CLASS_NAMES[orig],
            "original_probability": float(p0[orig]),
            "target_class": resolved,
            "target_label": CLASS_NAMES[resolved] if resolved is not None else None,
            "counterfactuals": [c.to_dict() for c in cfs],
        }
        return payload, message


class InferenceEngine:
    def __init__(self, model_name: str = PRIMARY_MODEL):
        self.model_name = model_name
        self.model = load_model(model_name)
        # SHAP 0.52's KernelExplainer has no `seed` parameter; it draws coalitions
        # from the global NumPy RNG. Seeding it here is a best-effort for
        # reproducibility of the first (warm-up) explanation.
        np.random.seed(SEED)
        background = shap.kmeans(load_x_train(), N_BACKGROUND)
        self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        self.expected_value = np.array(self.explainer.expected_value, dtype=float)
        self.class_names = CLASS_NAMES
        self.features = FEATURES
        # SHAP's KernelExplainer is NOT thread-safe: shap_values() mutates
        # instance state (self.y, self.nsamplesRun, self.nsamplesAdded) as it
        # runs. FastAPI executes these sync endpoints in Starlette's threadpool,
        # so two overlapping requests on this one shared explainer corrupt each
        # other's buffers and raise
        #   ValueError: could not broadcast input array from shape (4725,3)
        #   into shape (5000,3)
        # (5000 = NSAMPLES * N_BACKGROUND). Serialising the call is the minimal
        # correct fix; at ~122ms per explanation that caps throughput at about
        # 8 req/s, which is fine for this workload.
        self._shap_lock = threading.Lock()

        # ---- FC4: calibrated model + abstention threshold ----
        # Which calibrated variant and which tau were chosen is a decision made
        # offline (src/evaluate_calibration.py picks the variant, src/abstention.py
        # picks tau); the API reads that decision rather than re-deriving it.
        abst = abstention_json()
        self.calibration_variant = abst["model_variant"]
        self.calibrated_model_file = abst["model_file"]
        self.tau = float(abst["chosen"]["tau"])
        self.margin_definition = abst["margin_definition"]
        # model_file is repo-relative, e.g. "models/ensemble_softmax_isotonic.pkl"
        self.calibrated_model = load_model(Path(self.calibrated_model_file).stem)

    def warmup(self) -> float:
        """Run one throwaway explanation on the cached Pathological example so the
        first real request does not pay the warm-up cost.

        Returns the warm-up latency in ms (for logging).
        """
        patho = example_cases()["Pathological"]["feature_values"]
        warm_row = {f: float(patho[f]) for f in FEATURES}
        t0 = time.perf_counter()
        self.predict_one(warm_row)
        return (time.perf_counter() - t0) * 1000.0

    # ---- helpers ----
    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=FEATURES)

    def _shap(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP array of shape (n, n_features, n_classes)."""
        with self._shap_lock:
            arr = np.array(
                self.explainer.shap_values(X, nsamples=NSAMPLES, l1_reg=L1_REG, silent=True)
            )
        return arr

    # ---- public API ----
    def predict_one(self, row: dict) -> dict:
        t0 = time.perf_counter()
        X = self._frame([row])
        proba = self.model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        shap_arr = self._shap(X)[0]  # (n_features, n_classes)
        elapsed = (time.perf_counter() - t0) * 1000.0

        shap_items = [
            {"feature": f, "value": float(row[f]), "shap_value": float(shap_arr[i, pred])}
            for i, f in enumerate(FEATURES)
        ]
        return {
            "predicted_class": pred,
            "predicted_label": CLASS_NAMES[pred],
            "probabilities": {CLASS_NAMES[c]: float(proba[c]) for c in range(len(CLASS_NAMES))},
            "confidence": float(proba[pred]),
            "shap_values": shap_items,
            "base_value": float(self.expected_value[pred]),
            "inference_time_ms": round(elapsed, 3),
        }

    def predict_calibrated(self, row: dict) -> dict:
        """predict_one() plus calibrated probabilities and the abstention decision.

        The SHAP block is unchanged and still explains the UNCALIBRATED ensemble
        that produced `predicted_label` -- calibration is a monotone reshaping of
        the probability scale applied afterwards, and the explainer is built over
        that base model. The abstention decision uses the CALIBRATED margin,
        matching how tau was chosen in src/abstention.py.
        """
        base = self.predict_one(row)

        t0 = time.perf_counter()
        X = self._frame([row])
        cal_proba = np.asarray(self.calibrated_model.predict_proba(X)[0], dtype=float)
        cal_elapsed = (time.perf_counter() - t0) * 1000.0

        order = np.sort(cal_proba)
        margin = float(order[-1] - order[-2])
        cal_pred = int(np.argmax(cal_proba))
        abstain = bool(margin < self.tau)

        base.update({
            "calibrated_probabilities": {
                CLASS_NAMES[c]: float(cal_proba[c]) for c in range(len(CLASS_NAMES))
            },
            "calibrated_predicted_class": cal_pred,
            "calibrated_predicted_label": CLASS_NAMES[cal_pred],
            "calibrated_confidence": float(cal_proba[cal_pred]),
            "margin": margin,
            "abstain": abstain,
            "tau": self.tau,
            "recommended_action": (
                "refer to specialist" if abstain else "auto-classify"
            ),
            "calibration_variant": self.calibration_variant,
            "calibration_time_ms": round(cal_elapsed, 3),
        })
        return base

    def explain_one(self, row: dict) -> dict:
        t0 = time.perf_counter()
        X = self._frame([row])
        proba = self.model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        shap_arr = self._shap(X)[0]  # (n_features, n_classes)
        elapsed = (time.perf_counter() - t0) * 1000.0

        per_class = []
        for c, name in enumerate(CLASS_NAMES):
            items = [
                {"feature": f, "value": float(row[f]), "shap_value": float(shap_arr[i, c])}
                for i, f in enumerate(FEATURES)
            ]
            per_class.append({
                "class_label": name,
                "base_value": float(self.expected_value[c]),
                "shap_values": items,
            })
        return {
            "predicted_class": pred,
            "predicted_label": CLASS_NAMES[pred],
            "probabilities": {CLASS_NAMES[c]: float(proba[c]) for c in range(len(CLASS_NAMES))},
            "per_class": per_class,
            "inference_time_ms": round(elapsed, 3),
        }

    def predict_batch(self, rows: list[dict]) -> dict:
        t0 = time.perf_counter()
        X = self._frame(rows)
        proba = self.model.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        elapsed = (time.perf_counter() - t0) * 1000.0

        predictions = []
        summary = {name: 0 for name in CLASS_NAMES}
        for i, p in enumerate(preds):
            p = int(p)
            summary[CLASS_NAMES[p]] += 1
            predictions.append({
                "index": i,
                "predicted_class": p,
                "predicted_label": CLASS_NAMES[p],
                "probabilities": {CLASS_NAMES[c]: float(proba[i, c]) for c in range(len(CLASS_NAMES))},
                "confidence": float(proba[i, p]),
            })
        return {
            "n": len(rows),
            "predictions": predictions,
            "class_summary": summary,
            "inference_time_ms": round(elapsed, 3),
        }
