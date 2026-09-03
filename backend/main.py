"""
Module: backend.main
Purpose: FastAPI application — lifespan event hooks, CORS, and all API endpoints.

Startup sequence:
  1. Verify all 4 .joblib model files exist on disk.
  2. Load models into ModelRegistry via model_service.initialize_models().
  3. Load ML-ready dataset into ComponentService via component_service.initialize_dataset().
  4. If either step fails the server exits immediately with a descriptive error.

CORS:
  Allowed origins are read from the ALLOWED_ORIGINS environment variable.
  Default (development): http://localhost:3000,http://localhost:5173
  For production, set ALLOWED_ORIGINS to the exact frontend domain.

Endpoints:
  GET  /api/health                   — Readiness / liveness probe
  GET  /api/components               — Paginated component list (no ground truth)
  GET  /api/components/{id}          — Single component measurements (no ground truth)
  POST /api/predict                  — Main screening prediction endpoint

OpenAPI docs:
  GET  /docs                         — Swagger UI
  GET  /redoc                        — ReDoc UI
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure workspace root is importable
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from backend.schemas import (  # noqa: E402
    HealthResponse,
    ComponentLookupResponse,
    ComponentListResponse,
    ComponentListItem,
    PredictRequest,
    PredictResponse,
    ScreeningStage,
    DatasetOverviewResponse,
)
from backend.services import model_service, component_service  # noqa: E402
from backend.services.prediction_service import build_predict_response  # noqa: E402
from backend.utils.preprocessing import build_inference_row  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — models + dataset loaded ONCE at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """FastAPI lifespan: initialise all services on startup, clean up on shutdown."""
    logger.info("=== SIH 2026 Burn-In Screening API — Startup ===")

    try:
        logger.info("Loading ML model artifacts from disk ...")
        model_service.initialize_models()
        logger.info(
            "Models loaded: %s",
            list(model_service.model_load_status().keys()),
        )
    except FileNotFoundError as exc:
        logger.critical("FATAL: %s", exc)
        raise SystemExit(1) from exc

    try:
        logger.info("Loading ML-ready dataset ...")
        component_service.initialize_dataset()
        logger.info("Dataset loaded successfully.")
    except Exception as exc:  # pragma: no cover
        logger.critical("FATAL: Could not load dataset: %s", exc)
        raise SystemExit(1) from exc

    logger.info("=== API ready. Visit /docs for interactive documentation. ===")
    yield
    logger.info("=== SIH 2026 API — Shutdown ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SIH 2026 — AI-Driven Component Burn-In Screening API",
    description=(
        "FastAPI backend for the Predictive Burn-In Screening System. "
        "Connects trained ML models (Module A: defect classification, "
        "Module B: 168h degradation forecasting) to the frontend dashboard.\n\n"
        "**Temporal isolation guarantee**: The API strictly enforces that "
        "96h sensor data cannot influence 24h-gate predictions, and that "
        "168h end-of-test measurements are never used as model inputs.\n\n"
        "**No ground truth leakage**: `module_a_label`, `iddq_drift_168h_true`, "
        "and `component_type` are never returned by any inference endpoint."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info("CORS configured for origins: %s", _allowed_origins)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
    description=(
        "Returns the current readiness status of the API. "
        "`status` is `'ok'` only when all four model artifacts are loaded. "
        "Does not expose filesystem paths or sensitive configuration."
    ),
)
def health_check() -> HealthResponse:
    status_map = model_service.model_load_status()
    all_ok = all(status_map.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        module_a_loaded=model_service.module_a_loaded(),
        module_b_loaded=model_service.module_b_loaded(),
        models_detail=status_map,
    )


@app.get(
    "/api/dataset-overview",
    response_model=DatasetOverviewResponse,
    summary="Get Dataset Overview",
    tags=["Dataset"],
    description=(
        "Returns aggregated dataset distribution metrics from the ground truth dataset "
        "(normal vs. drifting vs. anomalous proportions and burn-in gates). "
        "No individual component labels or sensitive ground-truth records are exposed."
    ),
)
def dataset_overview() -> DatasetOverviewResponse:
    return DatasetOverviewResponse(**component_service.get_dataset_overview())


@app.get(
    "/api/components",
    response_model=ComponentListResponse,
    summary="List Components",
    tags=["Components"],
    description=(
        "Returns a paginated list of component IDs with basic IDDQ readings. "
        "Use `split=test` to list only the locked evaluation set. "
        "Ground-truth labels are never included."
    ),
)
def list_components(
    split: str = Query(
        default="all",
        pattern="^(all|test)$",
        description="Dataset partition to list: 'all' or 'test' (locked evaluation set)",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Row offset for pagination"),
) -> ComponentListResponse:
    total, items = component_service.list_components(split=split, limit=limit, offset=offset)
    return ComponentListResponse(
        total_available=total,
        returned_count=len(items),
        split_filter=split,
        components=[ComponentListItem(**item) for item in items],
    )


@app.get(
    "/api/components/{component_id}",
    response_model=ComponentLookupResponse,
    summary="Get Component Measurements",
    tags=["Components"],
    description=(
        "Returns all available sensor measurements for the specified component. "
        "**Ground-truth labels (`module_a_label`, `iddq_drift_168h_true`) "
        "are never returned by this endpoint.**"
    ),
)
def get_component(component_id: str) -> ComponentLookupResponse:
    payload = component_service.get_component_lookup_payload(component_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Component '{component_id}' not found in the dataset.",
        )
    return ComponentLookupResponse(**payload)


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Run Burn-In Screening Prediction",
    tags=["Prediction"],
    description=(
        "Main screening endpoint. Accepts either:\n\n"
        "- **`component_id`**: Loads measurements from the dataset and runs prediction.\n"
        "- **Raw measurements**: `measurements_0h` + `measurements_24h` + "
        "  (optional) `measurements_96h`.\n\n"
        "The backend automatically determines which screening gate(s) to run:\n"
        "- Only 0h → `insufficient` (monitoring only, no model output)\n"
        "- 0h + 24h → A24 + B24 (24h early gate)\n"
        "- 0h + 24h + 96h → A24 + B24 + A96 + B96 (full dual-gate assessment)\n\n"
        "**Temporal integrity**: 96h measurements NEVER influence 24h gate predictions. "
        "168h data is never accepted as input.\n\n"
        "**No ground truth leakage**: Ground-truth labels are not used or returned."
    ),
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Main screening prediction endpoint.

    Determines available gates from the provided data and runs the appropriate
    Module A + Module B pipelines. Returns a fully structured response including
    observed parameter changes, feature importances, and a final PASS/REVIEW/REJECT
    decision.
    """
    # --- Mode 1: component_id lookup ---
    if request.component_id is not None:
        c_id = request.component_id

        if not component_service.component_exists(c_id):
            raise HTTPException(
                status_code=404,
                detail=f"Component '{c_id}' not found in the dataset.",
            )

        row = component_service.get_component_row(c_id)
        if row is None:  # pragma: no cover
            raise HTTPException(status_code=404, detail=f"Component '{c_id}' not found.")

        # Determine available gates from the row data
        has_24h = _has_valid_24h(row)
        has_96h = _has_valid_96h(row)

    # --- Mode 2: raw measurements ---
    else:
        m0h = request.measurements_0h
        m24h = request.measurements_24h
        m96h = request.measurements_96h
        c_id = "custom_input"

        row = build_inference_row(m0h, m24h, m96h)
        has_24h = True  # m24h is required when m0h is provided (validated in schema)
        has_96h = m96h is not None

    try:
        return build_predict_response(
            component_id=c_id,
            row=row,
            has_24h=has_24h,
            has_96h=has_96h,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during prediction for '%s': %s", c_id, exc)
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_valid_24h(row: dict) -> bool:
    """True if the row contains at least one non-None 24h sensor value."""
    _24h_cols = [
        "iddq_uA_24h", "leakage_current_uA_24h",
        "propagation_delay_ns_24h", "voltage_V_24h", "temperature_C_24h",
    ]
    return any(row.get(c) is not None for c in _24h_cols)


def _has_valid_96h(row: dict) -> bool:
    """True if the row contains at least one non-None 96h sensor value."""
    _96h_cols = [
        "iddq_uA_96h", "leakage_current_uA_96h",
        "propagation_delay_ns_96h", "voltage_V_96h", "temperature_C_96h",
    ]
    return any(row.get(c) is not None for c in _96h_cols)


# ---------------------------------------------------------------------------
# Serve Frontend Static Files & Dashboard Index
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(_WORKSPACE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    static_dir = os.path.join(FRONTEND_DIR, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(content=b"", status_code=204)


