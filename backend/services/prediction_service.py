"""
Module: backend.services.prediction_service
Purpose: Orchestrates Module A and Module B inference for each available screening gate,
         assembles GateResults, and builds the final PredictResponse.

Design principles:
  - Delegates ALL model inference to src.inference.predict (no re-implementation).
  - Delegates ALL screening decisions to src.decision.screening_decision.
  - Computes feature importance via backend.services.model_service.
  - Computes observed changes via backend.utils.preprocessing.
  - Never touches ground-truth columns.
  - Temporal isolation: 96h data is ONLY passed to A96/B96 when the user
    explicitly provides 96h measurements.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from src.inference.predict import prepare_inference_features, predict_24h, predict_96h  # noqa
from src.decision.screening_decision import make_screening_decision, DEFAULT_CONFIG  # noqa
from src.features.build_features import FEATURES_24H_GATE, FEATURES_96H_GATE  # noqa

from backend.schemas import (  # noqa
    ScreeningStage,
    ScreeningDecision,
    ConfidenceLevel,
    ModuleAGateResult,
    ModuleBGateResult,
    GateResults,
    FinalDecision,
    MeasurementsSnapshot,
    PredictResponse,
    ObservedChange,
)
from backend.services.model_service import get_models, get_feature_importances  # noqa
from backend.utils.preprocessing import compute_observed_changes  # noqa


# ---------------------------------------------------------------------------
# Internal: run one screening gate
# ---------------------------------------------------------------------------

def _run_gate(
    row: Dict[str, Any],
    gate: str,
    models: Dict[str, Any],
) -> GateResults:
    """
    Runs Module A + Module B for a single gate ('24h' or '96h') and assembles
    a GateResults object.

    Parameters
    ----------
    row   : flat dict of sensor readings (may contain NaNs — pipeline handles imputation)
    gate  : '24h' or '96h'
    models: loaded model registry from model_service
    """
    is_24h = "24" in gate

    # --- Module A ---
    if is_24h:
        inf_res = predict_24h(row, models=models)
        clf_pipeline = models["a24"]
        clf_name = "LogisticRegression"
        feature_list = FEATURES_24H_GATE
        imp_step = "classifier"
    else:
        inf_res = predict_96h(row, models=models)
        clf_pipeline = models["a96"]
        clf_name = "RandomForestClassifier"
        feature_list = FEATURES_96H_GATE
        imp_step = "classifier"

    defect_prob = inf_res["defect_probability"]
    pred_class = inf_res["predicted_class"]
    drift_raw = inf_res["predicted_168h_iddq_drift"]
    drift_pct = inf_res["predicted_168h_iddq_drift_pct"]
    n_features = inf_res["num_features_used"]
    model_b_name = inf_res["model_b_name"]

    importances = get_feature_importances(clf_pipeline, list(feature_list), step_name=imp_step)

    module_a = ModuleAGateResult(
        gate=gate,
        model_name=clf_name,
        prediction=pred_class,
        class_name="defective" if pred_class == 1 else "normal",
        risk_probability=defect_prob,
        features_used=n_features,
        feature_importances=importances,
    )

    # --- Module B ---
    module_b = ModuleBGateResult(
        gate=gate,
        model_name=model_b_name,
        predicted_iddq_drift_168h=round(drift_raw, 6),
        predicted_iddq_drift_168h_pct=round(drift_pct, 3),
    )

    # --- Gate-level Decision ---
    dec_raw = make_screening_decision(
        defect_probability=defect_prob,
        predicted_168h_drift=drift_raw,
        screening_gate=gate,
        config=DEFAULT_CONFIG,
    )
    gate_decision = FinalDecision(
        status=ScreeningDecision(dec_raw["decision"]),
        confidence_level=ConfidenceLevel(dec_raw["confidence_level"]),
        reason=dec_raw["reason"],
        recommendation=dec_raw["recommendation"],
    )

    return GateResults(module_a=module_a, module_b=module_b, gate_decision=gate_decision)


# ---------------------------------------------------------------------------
# Public: build_predict_response
# ---------------------------------------------------------------------------

def build_predict_response(
    component_id: str,
    row: Dict[str, Any],
    has_24h: bool,
    has_96h: bool,
) -> PredictResponse:
    """
    Builds the complete PredictResponse for a component.

    Parameters
    ----------
    component_id : identifier string
    row          : flat dict of all sensor readings (no ground-truth cols)
    has_24h      : whether valid 24h measurements are present
    has_96h      : whether valid 96h measurements are present
    """
    models = get_models()

    # Determine screening stage
    if has_96h:
        stage = ScreeningStage.GATE_96H
    elif has_24h:
        stage = ScreeningStage.GATE_24H
    else:
        stage = ScreeningStage.INSUFFICIENT

    # Build measurements snapshot (echo of inputs)
    snap_0h = _extract_time_block(row, suffix="_0h")
    snap_24h = _extract_time_block(row, suffix="_24h") if has_24h else None
    snap_96h = _extract_time_block(row, suffix="_96h") if has_96h else None

    measurements = MeasurementsSnapshot(**{"0h": snap_0h, "24h": snap_24h, "96h": snap_96h})

    # Observed changes
    observed_changes = compute_observed_changes(row, has_96h=has_96h) if has_24h else []

    # Run gate(s)
    gate_24h_result: Optional[GateResults] = None
    gate_96h_result: Optional[GateResults] = None
    final_decision: Optional[FinalDecision] = None

    if stage == ScreeningStage.INSUFFICIENT:
        # Cannot run any model — return monitoring-only response
        pass

    elif stage == ScreeningStage.GATE_24H:
        gate_24h_result = _run_gate(row, "24h", models)
        # At 24h-only stage, the gate decision IS the final decision
        final_decision = gate_24h_result.gate_decision

    elif stage == ScreeningStage.GATE_96H:
        gate_24h_result = _run_gate(row, "24h", models)
        gate_96h_result = _run_gate(row, "96h", models)
        # The final decision uses the more informative 96h gate result
        final_decision = gate_96h_result.gate_decision

    return PredictResponse(
        component_id=component_id,
        screening_stage=stage,
        measurements=measurements,
        observed_changes=observed_changes,
        gate_24h=gate_24h_result,
        gate_96h=gate_96h_result,
        final_decision=final_decision,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_time_block(row: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Returns a sub-dict of row keys that end with the given suffix."""
    return {k: v for k, v in row.items() if k.endswith(suffix)}
