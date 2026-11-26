"""Pydantic v2 schemas.

The input model `CTGFeatures` is built programmatically with `create_model` so
its per-field `ge`/`le` bounds come directly from the training-set ranges
(feature_stats()), not hardcoded numbers. Field attribute names are Python-safe
aliases of the raw CSV column names (only "baseline value" differs); the JSON
API uses the raw feature names via aliases.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from artifacts import example_cases, feature_stats
from constants import CLASS_NAMES, FEATURES
from feature_meta import FEATURE_META, safe_name


def _build_ctg_features_model():
    stats = feature_stats()
    fields: dict[str, Any] = {}
    for feat in FEATURES:
        s = stats[feat]
        fields[safe_name(feat)] = (
            float,
            Field(
                ...,
                ge=s["min"],
                le=s["max"],
                alias=feat,
                description=f"{FEATURE_META[feat]['description']} "
                f"(unit: {FEATURE_META[feat]['unit']}; train range "
                f"{s['min']:.4g}..{s['max']:.4g})",
            ),
        )

    # A real, in-bounds example payload (the cached Pathological demo case),
    # keyed by the raw feature names so Swagger's "Try it out" works as-is.
    example = {f: float(v) for f, v in example_cases()["Pathological"]["feature_values"].items()}

    config = ConfigDict(populate_by_name=True, json_schema_extra={"example": example})
    return create_model("CTGFeatures", __config__=config, **fields)


CTGFeatures = _build_ctg_features_model()


def features_to_row(features: "CTGFeatures") -> dict:
    """Return a dict keyed by the raw feature names, ordered like FEATURES."""
    dumped = features.model_dump(by_alias=True)
    return {feat: dumped[feat] for feat in FEATURES}


# ---------- response models ----------

class ShapItem(BaseModel):
    feature: str
    value: float = Field(..., description="The input feature value.")
    shap_value: float = Field(..., description="SHAP contribution for the predicted class.")


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    confidence: float
    shap_values: list[ShapItem] = Field(..., description="Per-feature SHAP for the predicted class.")
    base_value: float = Field(..., description="Expected value (base) for the predicted class.")
    inference_time_ms: float


class CalibratedPredictResponse(PredictResponse):
    """PredictResponse (SHAP fields intact) + calibration and abstention."""

    calibrated_probabilities: dict[str, float] = Field(
        ..., description="Probabilities from the chosen calibrated model."
    )
    calibrated_predicted_class: int
    calibrated_predicted_label: str
    calibrated_confidence: float
    margin: float = Field(
        ..., description="Top calibrated probability minus the second-highest."
    )
    abstain: bool = Field(
        ..., description="True when margin < tau: refer for specialist review."
    )
    tau: float = Field(..., description="Recommended abstention threshold.")
    recommended_action: str
    calibration_variant: str
    calibration_time_ms: float


class CounterfactualChange(BaseModel):
    feature: str
    from_value: float = Field(..., description="Original value.")
    to_value: float = Field(..., description="Counterfactual value.")
    direction: str = Field(..., description='"increase" or "decrease".')
    delta: float
    normalised_delta: float = Field(
        ..., description="|delta| divided by the feature's 1st-99th percentile span."
    )


class CounterfactualItem(BaseModel):
    changes: list[CounterfactualChange]
    n_features_changed: int
    total_normalised_change: float
    predicted_class: int
    predicted_label: str
    calibrated_probability: float
    margin: float
    target_reached: bool


class CounterfactualRequest(BaseModel):
    features: CTGFeatures
    target_class: int | None = Field(
        None, description="0/1/2. Defaults to the next-less-severe class."
    )
    k: int = Field(3, ge=1, le=5, description="Maximum number of counterfactuals.")


class CounterfactualResponse(BaseModel):
    original_class: int
    original_label: str
    original_probability: float
    target_class: int | None
    target_label: str | None
    counterfactuals: list[CounterfactualItem]
    message: str = Field(
        ..., description="Explains the result, including why none was found."
    )
    tau: float
    calibration_variant: str
    elapsed_ms: float


class ChatMessage(BaseModel):
    role: str = Field(..., description='"user" or "assistant".')
    content: str


class ChatShapItem(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str = Field(..., description='"toward" or "against" the predicted class.')


class ChatCaseContext(BaseModel):
    """The model output the Predict page is ALREADY showing.

    The backend does not recompute any of this. It is echoed back from the
    frontend so the assistant can only ever describe what the user can see --
    the assistant and the UI can never disagree about a number.
    """

    predicted_class: int
    predicted_label: str
    calibrated_probabilities: dict[str, float]
    margin: float
    abstain: bool
    tau: float
    top_shap: list[ChatShapItem] = Field(default_factory=list)
    counterfactual: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    features: CTGFeatures
    case: ChatCaseContext
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    ollama_available: bool
    model: str
    elapsed_ms: float


class CalibrationResponse(BaseModel):
    """Everything the frontend needs to render the calibration story."""

    calibration: dict[str, Any] = Field(
        ..., description="artifacts/calibration.json -- ECE/Brier table and reliability bins."
    )
    abstention: dict[str, Any] = Field(
        ..., description="artifacts/abstention.json -- chosen tau and its stats."
    )
    abstention_sweep: list[dict[str, Any]] = Field(
        ..., description="artifacts/abstention_sweep.csv -- one row per tau."
    )
    figures: dict[str, str] = Field(
        ..., description="Repo-relative paths to the calibration/abstention PNGs."
    )


class BatchRequest(BaseModel):
    samples: list[CTGFeatures] = Field(..., max_length=500, min_length=1)


class BatchPrediction(BaseModel):
    index: int
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    confidence: float


class BatchResponse(BaseModel):
    n: int
    predictions: list[BatchPrediction]
    class_summary: dict[str, int]
    inference_time_ms: float


class ClassShap(BaseModel):
    class_label: str
    base_value: float
    shap_values: list[ShapItem]


class ExplainResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    per_class: list[ClassShap] = Field(..., description="SHAP for all three classes.")
    inference_time_ms: float


class MetricsResponse(BaseModel):
    n_test: int
    baseline: str
    ranked_table: list[dict[str, Any]]
    confusion_matrices: dict[str, list[list[int]]]
    classification_reports: dict[str, dict[str, Any]]


class GlobalExplainResponse(BaseModel):
    kernel_ranking: list[dict[str, Any]]
    tree_ranking: list[dict[str, Any]]
    top10_comparison: list[dict[str, str]]
    spearman_rho: float
    spearman_p_value: float


class ExampleCase(BaseModel):
    name: str
    feature_values: dict[str, float]
    true_label: str
    predicted_label: str
    probabilities: dict[str, float]


class ExamplesResponse(BaseModel):
    examples: list[ExampleCase]


class FeatureMeta(BaseModel):
    name: str
    label: str
    description: str
    unit: str
    min: float
    max: float
    median: float
    group: str


class FeaturesResponse(BaseModel):
    groups: list[str]
    features: list[FeatureMeta]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    classes: list[str]
    versions: dict[str, str]

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(BaseModel):
    error: str
    detail: str
