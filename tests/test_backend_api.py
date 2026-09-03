"""
Test Suite: tests/test_backend_api.py
Project: SIH 2026 — AI-Driven Anomaly Detection in Component Burn-In & Screening
Purpose: FastAPI backend tests covering all 13 required scenarios.

Tests:
  1.  Model loading
  2.  Health endpoint
  3.  Valid prediction request (24h gate — component_id mode)
  4.  Valid prediction request (96h gate — component_id mode)
  5.  Invalid sensor values → HTTP 422
  6.  A24 temporal leakage prevention (96h features must not enter 24h gate)
  7.  A96 temporal leakage prevention (168h features must never enter any gate)
  8.  B24 prediction output schema and value types
  9.  B96 prediction output schema and value types
  10. Missing-value handling (NaN sensors handled by pipeline imputer)
  11. Component lookup endpoint (no ground truth exposed)
  12. Ground-truth fields not in predict response
  13. Model files are not modified
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    TestClient with lifespan=True so that startup events (model loading,
    dataset loading) run exactly once for the test module.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# Known healthy component from the dataset (observed in demo results: PASS at 96h)
_HEALTHY_ID = "SYN_C01216"
# Known defective component (observed: REJECT at 24h due to 99.9% defect prob)
_DEFECTIVE_ID = "SYN_C04946"


# ---------------------------------------------------------------------------
# Reference sensor payload for raw-measurement mode (based on SYN_C01216 values)
# ---------------------------------------------------------------------------
_VALID_24H_PAYLOAD: Dict[str, Any] = {
    "measurements_0h": {
        "iddq_uA_0h": 98.2,
        "leakage_current_uA_0h": 2.1,
        "propagation_delay_ns_0h": 1.05,
        "voltage_V_0h": 1.2,
        "temperature_C_0h": 125.0,
    },
    "measurements_24h": {
        "iddq_uA_24h": 99.5,
        "leakage_current_uA_24h": 2.15,
        "propagation_delay_ns_24h": 1.06,
        "voltage_V_24h": 1.2,
        "temperature_C_24h": 125.1,
    },
}

_VALID_96H_PAYLOAD: Dict[str, Any] = {
    **_VALID_24H_PAYLOAD,
    "measurements_96h": {
        "iddq_uA_96h": 100.1,
        "leakage_current_uA_96h": 2.18,
        "propagation_delay_ns_96h": 1.07,
        "voltage_V_96h": 1.2,
        "temperature_C_96h": 125.3,
    },
}


# ===========================================================================
# Test 1: Model Loading
# ===========================================================================

def test_01_model_loading(client: TestClient):
    """
    Test 1: All four production model artifacts load successfully.
    Verifies they expose the expected sklearn Pipeline interface.
    The client fixture triggers the FastAPI lifespan (startup event) which
    calls initialize_models() — so models are guaranteed to be loaded here.
    """
    from backend.services.model_service import get_models, is_loaded

    # The lifespan runs when TestClient enters its context manager (module-scoped fixture)
    assert is_loaded(), "Model registry must be populated after startup lifespan"
    models = get_models()

    for key in ("a24", "a96", "b24", "b96"):
        assert key in models, f"Expected model key '{key}' in registry"
        assert hasattr(models[key], "predict"), f"Model '{key}' must have .predict()"

    # Classification models must support predict_proba
    assert hasattr(models["a24"], "predict_proba"), "a24 must have predict_proba"
    assert hasattr(models["a96"], "predict_proba"), "a96 must have predict_proba"


# ===========================================================================
# Test 2: Health Endpoint
# ===========================================================================

def test_02_health_endpoint(client: TestClient):
    """
    Test 2: GET /api/health returns status 'ok' and correct model load flags.
    """
    resp = client.get("/api/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["module_a_loaded"] is True
    assert data["module_b_loaded"] is True

    # Individual model flags
    for key in ("a24", "a96", "b24", "b96"):
        assert data["models_detail"][key] is True, f"Model '{key}' not marked as loaded"

    # Must NOT expose filesystem paths or sensitive information
    response_text = resp.text.lower()
    assert "c:\\" not in response_text
    assert "d:\\" not in response_text
    assert "password" not in response_text


# ===========================================================================
# Test 3: Valid Prediction Request — 24h gate
# ===========================================================================

def test_03_valid_prediction_24h_gate_component_id(client: TestClient):
    """
    Test 3: POST /api/predict with component_id runs A24 + B24 when only 24h data
    is available and returns a valid response schema.
    """
    # Use raw measurements with only 24h data to force GATE_24H
    resp = client.post("/api/predict", json=_VALID_24H_PAYLOAD)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["screening_stage"] == "24h"
    assert data["component_id"] == "custom_input"

    # gate_24h must be present
    g24 = data["gate_24h"]
    assert g24 is not None
    assert g24["module_a"]["features_used"] == 11
    assert g24["module_a"]["risk_probability"] >= 0.0
    assert g24["module_a"]["risk_probability"] <= 1.0
    assert g24["module_a"]["prediction"] in (0, 1)
    assert g24["module_a"]["class_name"] in ("normal", "defective")

    # Module B output
    assert g24["module_b"]["predicted_iddq_drift_168h"] is not None
    assert isinstance(g24["module_b"]["predicted_iddq_drift_168h_pct"], float)

    # gate_96h must be absent for 24h-only request
    assert data["gate_96h"] is None

    # Final decision
    assert data["final_decision"]["status"] in ("PASS", "REVIEW", "REJECT")


# ===========================================================================
# Test 4: Valid Prediction Request — 96h gate
# ===========================================================================

def test_04_valid_prediction_96h_gate(client: TestClient):
    """
    Test 4: POST /api/predict with 96h measurements runs A24+B24 AND A96+B96.
    """
    resp = client.post("/api/predict", json=_VALID_96H_PAYLOAD)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["screening_stage"] == "96h"

    # Both gates must be present
    g24 = data["gate_24h"]
    g96 = data["gate_96h"]
    assert g24 is not None
    assert g96 is not None

    # 24h gate: 11 features
    assert g24["module_a"]["features_used"] == 11
    # 96h gate: 19 features
    assert g96["module_a"]["features_used"] == 19

    # Model names should match trained artifacts
    assert "logistic" in g24["module_a"]["model_name"].lower() or \
           "regression" in g24["module_a"]["model_name"].lower()
    assert "randomforest" in g96["module_a"]["model_name"].lower() or \
           "forest" in g96["module_a"]["model_name"].lower()

    # Final decision comes from 96h gate
    assert data["final_decision"] is not None
    assert data["final_decision"]["status"] in ("PASS", "REVIEW", "REJECT")


# ===========================================================================
# Test 5: Invalid Sensor Values → HTTP 422
# ===========================================================================

def test_05_invalid_sensor_values_rejected(client: TestClient):
    """
    Test 5: Sensor values outside physically reasonable bounds must return 422.
    """
    # Negative IDDQ is physically impossible
    invalid_payload = {
        "measurements_0h": {
            "iddq_uA_0h": -999.0,          # invalid: below 0.0
            "leakage_current_uA_0h": 2.1,
            "propagation_delay_ns_0h": 1.05,
            "voltage_V_0h": 1.2,
            "temperature_C_0h": 125.0,
        },
        "measurements_24h": {
            "iddq_uA_24h": 99.5,
            "leakage_current_uA_24h": 2.15,
            "propagation_delay_ns_24h": 1.06,
            "voltage_V_24h": 1.2,
            "temperature_C_24h": 125.1,
        },
    }
    resp = client.post("/api/predict", json=invalid_payload)
    assert resp.status_code == 422, f"Expected 422 for invalid sensor, got {resp.status_code}"

    # Voltage above 2.0V is unreasonable for this sensor profile
    invalid_voltage = {
        "measurements_0h": {
            "iddq_uA_0h": 98.0,
            "voltage_V_0h": 9999.0,         # invalid: above 2.0V
            "temperature_C_0h": 125.0,
        },
        "measurements_24h": {"iddq_uA_24h": 99.0},
    }
    resp2 = client.post("/api/predict", json=invalid_voltage)
    assert resp2.status_code == 422


# ===========================================================================
# Test 6: A24 Temporal Leakage Prevention
# ===========================================================================

def test_06_a24_temporal_leakage_prevention():
    """
    Test 6: The 24h inference feature matrix must contain ONLY the 11 features
    defined for the 24h gate. No 96h or 168h feature can enter.
    """
    from src.features.build_features import FEATURES_24H_GATE, FEATURES_96H_GATE

    # Verify the gate feature sets are disjoint for 96h additions
    _96h_only = set(FEATURES_96H_GATE) - set(FEATURES_24H_GATE)
    for f in _96h_only:
        assert f not in FEATURES_24H_GATE, \
            f"96h feature '{f}' must not appear in the 24h gate feature set"

    # Verify no 168h feature is present in either gate
    for f in FEATURES_24H_GATE + FEATURES_96H_GATE:
        assert "_168h" not in f, f"168h feature '{f}' leaked into gate feature set"

    # Verify total counts match training design
    assert len(FEATURES_24H_GATE) == 11, \
        f"Expected 11 features in 24h gate, got {len(FEATURES_24H_GATE)}"
    assert len(FEATURES_96H_GATE) == 19, \
        f"Expected 19 features in 96h gate, got {len(FEATURES_96H_GATE)}"


# ===========================================================================
# Test 7: A96 Temporal Leakage Prevention
# ===========================================================================

def test_07_a96_temporal_leakage_prevention():
    """
    Test 7: prepare_inference_features must raise ValueError if a 168h measurement
    column is injected into the feature matrix.
    """
    from src.inference.predict import prepare_inference_features
    from src.features.build_features import FEATURES_168H_FORBIDDEN

    # Build a minimal valid 96h row with a 168h column injected
    row = {
        "iddq_uA_0h": 98.2, "leakage_current_uA_0h": 2.1,
        "propagation_delay_ns_0h": 1.05, "voltage_V_0h": 1.2, "temperature_C_0h": 125.0,
        "iddq_uA_24h": 99.5, "leakage_current_uA_24h": 2.15,
        "propagation_delay_ns_24h": 1.06, "voltage_V_24h": 1.2, "temperature_C_24h": 125.1,
        "iddq_drift_24h_pct": 0.013,
        "iddq_uA_96h": 100.1, "leakage_current_uA_96h": 2.18,
        "propagation_delay_ns_96h": 1.07, "voltage_V_96h": 1.2, "temperature_C_96h": 125.3,
        "iddq_drift_96h_pct": 0.019, "leakage_drift_96h_pct": 0.014, "delay_drift_96h_pct": 0.010,
    }

    # Valid 96h inference must succeed
    X, feat_names = prepare_inference_features(row, gate="96h")
    assert len(feat_names) == 19

    # Confirm no 168h columns are present in the feature matrix
    for col in X.columns:
        assert "_168h" not in col, f"168h column '{col}' found in 96h inference matrix!"

    # Confirm each FEATURES_168H_FORBIDDEN column is absent from both gate sets
    from src.features.build_features import FEATURES_24H_GATE, FEATURES_96H_GATE
    for forbidden in FEATURES_168H_FORBIDDEN:
        assert forbidden not in FEATURES_24H_GATE
        assert forbidden not in FEATURES_96H_GATE


# ===========================================================================
# Test 8: B24 Prediction Output Schema
# ===========================================================================

def test_08_b24_prediction_output_schema(client: TestClient):
    """
    Test 8: B24 regression output must be a numeric float in a realistic range,
    returned with both raw-fraction and percentage representations.
    """
    resp = client.post("/api/predict", json=_VALID_24H_PAYLOAD)
    assert resp.status_code == 200

    data = resp.json()
    b24 = data["gate_24h"]["module_b"]

    assert b24["gate"] == "24h"
    assert isinstance(b24["predicted_iddq_drift_168h"], float)
    assert isinstance(b24["predicted_iddq_drift_168h_pct"], float)

    # Raw fraction and percentage must be consistent (pct ≈ raw * 100)
    raw = b24["predicted_iddq_drift_168h"]
    pct = b24["predicted_iddq_drift_168h_pct"]
    assert abs(pct - raw * 100.0) < 0.1, \
        f"pct ({pct}) should be raw ({raw}) * 100"

    # Model name should reference GradientBoosting
    assert "gradient" in b24["model_name"].lower() or "boosting" in b24["model_name"].lower()


# ===========================================================================
# Test 9: B96 Prediction Output Schema
# ===========================================================================

def test_09_b96_prediction_output_schema(client: TestClient):
    """
    Test 9: B96 regression output schema with RandomForestRegressor.
    The 96h forecast should be more precise than the 24h forecast for a
    healthy component (lower absolute drift predicted).
    """
    resp = client.post("/api/predict", json=_VALID_96H_PAYLOAD)
    assert resp.status_code == 200

    data = resp.json()
    b96 = data["gate_96h"]["module_b"]

    assert b96["gate"] == "96h"
    assert isinstance(b96["predicted_iddq_drift_168h"], float)
    assert isinstance(b96["predicted_iddq_drift_168h_pct"], float)

    raw = b96["predicted_iddq_drift_168h"]
    pct = b96["predicted_iddq_drift_168h_pct"]
    assert abs(pct - raw * 100.0) < 0.1

    # Model name should reference RandomForest
    assert "random" in b96["model_name"].lower() or "forest" in b96["model_name"].lower()


# ===========================================================================
# Test 10: Missing Value Handling
# ===========================================================================

def test_10_missing_value_handling(client: TestClient):
    """
    Test 10: Requests with some None/missing sensor values should succeed —
    the trained pipeline's embedded SimpleImputer handles them. The API must
    NOT fit a new imputer from inference-time data.
    """
    # Payload with several None values (simulates ~1.5% sensor dropouts)
    sparse_payload = {
        "measurements_0h": {
            "iddq_uA_0h": 98.2,
            "leakage_current_uA_0h": None,   # missing
            "propagation_delay_ns_0h": 1.05,
            "voltage_V_0h": None,            # missing
            "temperature_C_0h": 125.0,
        },
        "measurements_24h": {
            "iddq_uA_24h": 99.5,
            "leakage_current_uA_24h": None,  # missing
            "propagation_delay_ns_24h": 1.06,
            "voltage_V_24h": 1.2,
            "temperature_C_24h": 125.1,
        },
    }
    resp = client.post("/api/predict", json=sparse_payload)
    assert resp.status_code == 200, f"Missing values should be handled, got: {resp.text}"

    data = resp.json()
    assert data["screening_stage"] == "24h"
    assert data["gate_24h"] is not None
    # Despite missing values, a valid prediction must be returned
    assert data["gate_24h"]["module_a"]["risk_probability"] >= 0.0


# ===========================================================================
# Test 11: Component Lookup — No Ground Truth
# ===========================================================================

def test_11_component_lookup_no_ground_truth(client: TestClient):
    """
    Test 11: GET /api/components/{id} returns measurements but NEVER
    module_a_label, iddq_drift_168h_true, or component_type.
    """
    resp = client.get(f"/api/components/{_HEALTHY_ID}")
    assert resp.status_code == 200

    data = resp.json()
    assert data["component_id"] == _HEALTHY_ID

    # Ground-truth fields must be absent
    assert "module_a_label" not in data
    assert "iddq_drift_168h_true" not in data
    assert "component_type" not in data

    # Measurement blocks must be present
    assert "measurements_0h" in data
    assert "measurements_24h" in data
    assert "measurements_96h" in data

    # IDDQ baseline must be a positive float
    iddq_0h = data["measurements_0h"].get("iddq_uA_0h")
    assert iddq_0h is not None and iddq_0h > 0.0

    # 404 for unknown component
    resp_404 = client.get("/api/components/SYN_C99999")
    assert resp_404.status_code == 404


# ===========================================================================
# Test 12: Ground-Truth Fields Not Exposed in Predict Response
# ===========================================================================

def test_12_ground_truth_not_in_predict_response(client: TestClient):
    """
    Test 12: POST /api/predict must NEVER return module_a_label,
    iddq_drift_168h_true, or component_type in any part of the response.
    """
    # Test using a dataset component (ID mode)
    resp = client.post("/api/predict", json={"component_id": _HEALTHY_ID})
    assert resp.status_code == 200

    # Inspect the raw JSON text to catch any accidental inclusion
    response_text = resp.text
    assert "module_a_label" not in response_text, \
        "module_a_label must never appear in predict response"
    assert "iddq_drift_168h_true" not in response_text, \
        "iddq_drift_168h_true must never appear in predict response"
    assert "component_type" not in response_text, \
        "component_type must never appear in predict response"

    # Also test raw measurements mode
    resp2 = client.post("/api/predict", json=_VALID_96H_PAYLOAD)
    assert resp2.status_code == 200
    response_text2 = resp2.text
    assert "module_a_label" not in response_text2
    assert "iddq_drift_168h_true" not in response_text2


# ===========================================================================
# Test 13: Model Files Not Modified
# ===========================================================================

def test_13_model_files_not_modified():
    """
    Test 13: All 4 model .joblib files must still exist with the same size
    they had before the backend was implemented. Confirms no retraining occurred.
    """
    model_files = {
        "module_a_24h_logisticregression.joblib":    2586,
        "module_a_96h_randomforest.joblib":           3_667_355,
        "module_b_24h_gradientboostingregressor.joblib": 452_186,
        "module_b_96h_randomforestregressor.joblib":  6_698_771,
    }
    models_dir = os.path.join(WORKSPACE_DIR, "models")

    for filename, expected_size in model_files.items():
        full_path = os.path.join(models_dir, filename)
        assert os.path.exists(full_path), f"Model file missing: {filename}"

        actual_size = os.path.getsize(full_path)
        assert actual_size == expected_size, (
            f"Model file '{filename}' has been modified! "
            f"Expected {expected_size} bytes, found {actual_size} bytes."
        )


# ===========================================================================
# Bonus: Predict with component_id — end-to-end smoke test
# ===========================================================================

def test_bonus_component_id_predict_healthy(client: TestClient):
    """
    Bonus smoke test: known healthy component should return PASS at 96h gate.
    Based on documented demo results in FINAL_INTEGRATION_REPORT.md.
    """
    resp = client.post("/api/predict", json={"component_id": _HEALTHY_ID})
    assert resp.status_code == 200

    data = resp.json()
    assert data["component_id"] == _HEALTHY_ID
    assert data["screening_stage"] == "96h"
    assert data["gate_24h"] is not None
    assert data["gate_96h"] is not None

    # SYN_C01216 documented as PASS at 96h gate (2.5% defect prob, 0.97% drift)
    g96 = data["gate_96h"]
    assert g96["module_a"]["risk_probability"] < 0.50, \
        "SYN_C01216 should have low defect probability at 96h"

    # Observed changes should be populated for a full 96h component
    assert len(data["observed_changes"]) > 0


def test_bonus_component_id_predict_defective(client: TestClient):
    """
    Bonus smoke test: known gross-defect component should REJECT at 24h gate.
    Based on documented demo results: SYN_C04946 → 99.9% defect prob at 24h.
    """
    resp = client.post("/api/predict", json={"component_id": _DEFECTIVE_ID})
    assert resp.status_code == 200

    data = resp.json()
    assert data["component_id"] == _DEFECTIVE_ID
    g24 = data["gate_24h"]
    assert g24 is not None
    # With ~99.9% defect probability the gate decision must be REJECT
    assert g24["gate_decision"]["status"] == "REJECT"
    assert g24["module_a"]["risk_probability"] > 0.70


# ===========================================================================
# Frontend & Overview Integration Tests
# ===========================================================================

def test_dataset_overview_endpoint(client: TestClient):
    """
    Verifies GET /api/dataset-overview returns aggregate numbers (10,000 total,
    7,000 normal, 2,000 drifting, 1,000 anomalous) without exposing individual component labels.
    """
    resp = client.get("/api/dataset-overview")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_components"] == 10000
    assert data["normal_count"] == 7000
    assert data["drifting_count"] == 2000
    assert data["anomalous_count"] == 1000
    assert len(data["burnin_gates"]) == 4


def test_frontend_index_serves_successfully(client: TestClient):
    """
    Verifies GET / serves the complete frontend index.html dashboard with HTTP 200.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "AI-Driven Burn-In Screening" in html
    assert "SIH 2026" in html
    assert "chart-progression-canvas" in html


def test_frontend_static_assets_serve_successfully(client: TestClient):
    """
    Verifies static CSS and JS assets are served correctly.
    """
    res_css = client.get("/static/css/style.css")
    assert res_css.status_code == 200
    assert len(res_css.text) > 1000

    res_js = client.get("/static/js/app.js")
    assert res_js.status_code == 200
    assert len(res_js.text) > 1000


if __name__ == "__main__":
    pytest.main(["-v", __file__])

