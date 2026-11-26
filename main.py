"""FastAPI inference backend for the fetal health CTG classifier.

Models and the SHAP explainer are loaded exactly once at startup via a lifespan
context manager and stored on app.state -- never per request.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is importable (constants, pickled ensemble classes).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from inference import CounterfactualService, InferenceEngine
from routers import calibration, chat, counterfactual, explain, metrics, predict

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the engine once (loads ensemble_softmax + KernelExplainer background)
    # and warm the explainer so the first request is fast, not cold.
    import logging

    engine = InferenceEngine()
    warm_ms = engine.warmup()
    logging.getLogger("uvicorn").info(
        "InferenceEngine ready (model=%s, warmup=%.1fms, calibrated=%s, tau=%.2f)",
        engine.model_name, warm_ms, engine.calibration_variant, engine.tau,
    )
    app.state.engine = engine

    # Counterfactual engine: loads the calibrated model + constraint table once.
    cf = CounterfactualService()
    cf_ms = cf.warmup()
    logging.getLogger("uvicorn").info(
        "CounterfactualService ready (model=%s, tau=%.2f, warmup=%.0fms)",
        cf.model_file, cf.tau, cf_ms,
    )
    app.state.cf_engine = cf

    yield
    app.state.engine = None
    app.state.cf_engine = None


app = FastAPI(
    title="Fetal Health CTG Inference API",
    description=(
        "Serves the weighted soft-voting ensemble (ensemble_softmax) that "
        "classifies cardiotocography (CTG) readings as Normal, Suspect, or "
        "Pathological. Provides single and batch prediction, per-class SHAP "
        "explanations, held-out test metrics, and demo cases for the frontend."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": str(exc.errors())},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(calibration.router, prefix="/api")
app.include_router(counterfactual.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
