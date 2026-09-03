# Backend API Report

**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Stage:** FastAPI Backend API Layer  
**Date:** 2026-09-03  
**Status:** ✅ Complete — 15/15 backend tests passing, 16/16 existing tests unaffected

---

## 1. Backend Folder Structure

```
SIH/
├── backend/
│   ├── __init__.py                   ← Package documentation
│   ├── main.py                       ← FastAPI app, lifespan, CORS, endpoints
│   ├── schemas.py                    ← All Pydantic v2 request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py          ← Model registry singleton
│   │   ├── prediction_service.py     ← Module A + B orchestration
│   │   └── component_service.py      ← Dataset lookup (no ground-truth leak)
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py          ← Drift computation, observed changes
│
├── tests/
│   ├── test_backend_api.py           ← 15 backend tests (NEW)
│   ├── test_inference_pipeline.py    ← 8 existing tests (UNCHANGED)
│   └── test_web_api.py               ← 8 Flask tests (UNCHANGED)
│
├── reports/
│   └── BACKEND_API_REPORT.md         ← This document
│
└── requirements.txt                  ← Added: fastapi, uvicorn, python-multipart, httpx
```

---

## 2. All Newly Created Backend Files

| File | Purpose | Lines |
|------|---------|-------|
| [`backend/__init__.py`](file:///d:/SIH/backend/__init__.py) | Package marker + architecture docs | 18 |
| [`backend/schemas.py`](file:///d:/SIH/backend/schemas.py) | All Pydantic v2 schemas | ~270 |
| [`backend/utils/__init__.py`](file:///d:/SIH/backend/utils/__init__.py) | Package marker | 4 |
| [`backend/utils/preprocessing.py`](file:///d:/SIH/backend/utils/preprocessing.py) | Drift computation, row assembly, ObservedChange | ~180 |
| [`backend/services/__init__.py`](file:///d:/SIH/backend/services/__init__.py) | Package marker | 4 |
| [`backend/services/model_service.py`](file:///d:/SIH/backend/services/model_service.py) | Model registry singleton + feature importance | ~130 |
| [`backend/services/prediction_service.py`](file:///d:/SIH/backend/services/prediction_service.py) | A+B inference orchestration | ~170 |
| [`backend/services/component_service.py`](file:///d:/SIH/backend/services/component_service.py) | Dataset lookup with ground-truth stripping | ~190 |
| [`backend/main.py`](file:///d:/SIH/backend/main.py) | FastAPI app, lifespan, CORS, 4 endpoints | ~310 |
| [`tests/test_backend_api.py`](file:///d:/SIH/tests/test_backend_api.py) | 15 backend tests | ~350 |

---

## 3. API Endpoint List

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Readiness probe — model load status |
| `GET` | `/api/components` | Paginated component list (no ground truth) |
| `GET` | `/api/components/{component_id}` | Single component measurements (no ground truth) |
| `POST` | `/api/predict` | Main burn-in screening prediction |
| `GET` | `/docs` | Swagger UI (FastAPI auto-generated) |
| `GET` | `/redoc` | ReDoc UI (FastAPI auto-generated) |

---

## 4. Model Loading Confirmation

All four trained `.joblib` pipeline artifacts are loaded **exactly once** at FastAPI
startup via the `lifespan` context manager. If any file is missing, the server
refuses to start and prints a descriptive error.

| Key | File | Algorithm | Size on disk |
|-----|------|-----------|-------------|
| `a24` | `module_a_24h_logisticregression.joblib` | LogisticRegression (scaled) | 2,586 B |
| `a96` | `module_a_96h_randomforest.joblib` | RandomForestClassifier (150 trees) | 3,667,355 B |
| `b24` | `module_b_24h_gradientboostingregressor.joblib` | GradientBoostingRegressor (100 trees) | 452,186 B |
| `b96` | `module_b_96h_randomforestregressor.joblib` | RandomForestRegressor (100 trees) | 6,698,771 B |

> No model was retrained. No `.joblib` file was modified. Test 13 explicitly
> verifies file sizes match the original byte counts.

---

## 5. Example `POST /api/predict` Request

### Mode A — Component ID Lookup (demo/evaluation)
```json
POST /api/predict
{
  "component_id": "SYN_C01216"
}
```

### Mode B — Raw Sensor Measurements (production inference)
```json
POST /api/predict
{
  "measurements_0h": {
    "iddq_uA_0h": 98.2,
    "leakage_current_uA_0h": 2.1,
    "propagation_delay_ns_0h": 1.05,
    "voltage_V_0h": 1.2,
    "temperature_C_0h": 125.0
  },
  "measurements_24h": {
    "iddq_uA_24h": 99.5,
    "leakage_current_uA_24h": 2.15,
    "propagation_delay_ns_24h": 1.06,
    "voltage_V_24h": 1.2,
    "temperature_C_24h": 125.1
  },
  "measurements_96h": {
    "iddq_uA_96h": 100.1,
    "leakage_current_uA_96h": 2.18,
    "propagation_delay_ns_96h": 1.07,
    "voltage_V_96h": 1.2,
    "temperature_C_96h": 125.3
  }
}
```

> Note: `measurements_96h` is optional. Omitting it runs only the 24h gate (A24 + B24).

---

## 6. Example `POST /api/predict` Response

```json
{
  "component_id": "SYN_C01216",
  "screening_stage": "96h",

  "measurements": {
    "0h": {
      "iddq_uA_0h": 97.8,
      "leakage_current_uA_0h": 2.08,
      "propagation_delay_ns_0h": 1.04,
      "voltage_V_0h": 1.2,
      "temperature_C_0h": 124.9
    },
    "24h": {
      "iddq_uA_24h": 99.2,
      "leakage_current_uA_24h": 2.12,
      "propagation_delay_ns_24h": 1.05,
      "voltage_V_24h": 1.2,
      "temperature_C_24h": 125.0
    },
    "96h": {
      "iddq_uA_96h": 100.3,
      "leakage_current_uA_96h": 2.14,
      "propagation_delay_ns_96h": 1.06,
      "voltage_V_96h": 1.2,
      "temperature_C_96h": 125.2
    }
  },

  "observed_changes": [
    {
      "parameter": "IDDQ",
      "unit": "μA",
      "start_time": "0h",
      "end_time": "96h",
      "start_value": 97.8,
      "end_value": 100.3,
      "absolute_change": 2.5,
      "percentage_change": 2.56
    }
  ],

  "gate_24h": {
    "module_a": {
      "gate": "24h",
      "model_name": "LogisticRegression",
      "prediction": 0,
      "class_name": "normal",
      "risk_probability": 0.384,
      "features_used": 11,
      "feature_importances": []
    },
    "module_b": {
      "gate": "24h",
      "model_name": "GradientBoostingRegressor",
      "predicted_iddq_drift_168h": 0.03572,
      "predicted_iddq_drift_168h_pct": 3.572
    },
    "gate_decision": {
      "status": "REVIEW",
      "confidence_level": "MEDIUM",
      "reason": "Intermediate degradation signal at 24h: Defect Probability = 38.4% (Between 25.0% and 75.0%)...",
      "recommendation": "Continue component stress testing to the 96h burn-in screening gate."
    }
  },

  "gate_96h": {
    "module_a": {
      "gate": "96h",
      "model_name": "RandomForestClassifier",
      "prediction": 0,
      "class_name": "normal",
      "risk_probability": 0.025,
      "features_used": 19,
      "feature_importances": [
        {"feature": "iddq_drift_96h_pct", "importance": 0.312},
        {"feature": "iddq_uA_96h",        "importance": 0.198},
        {"feature": "leakage_drift_96h_pct", "importance": 0.141}
      ]
    },
    "module_b": {
      "gate": "96h",
      "model_name": "RandomForestRegressor",
      "predicted_iddq_drift_168h": 0.0097,
      "predicted_iddq_drift_168h_pct": 0.97
    },
    "gate_decision": {
      "status": "PASS",
      "confidence_level": "HIGH",
      "reason": "Parametric reliability confirmed at 96h: Defect Probability = 2.5% (< 30.0%) and Safe Projected Degradation = 0.97% (< 3.0%). Component meets high-reliability standards.",
      "recommendation": "Pass component and release to production inventory."
    }
  },

  "final_decision": {
    "status": "PASS",
    "confidence_level": "HIGH",
    "reason": "Parametric reliability confirmed at 96h...",
    "recommendation": "Pass component and release to production inventory."
  }
}
```

---

## 7. Test Results

### New Backend Tests (`tests/test_backend_api.py`)

```
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\SIH
collected 15 items

tests/test_backend_api.py::test_01_model_loading                          PASSED [  6%]
tests/test_backend_api.py::test_02_health_endpoint                        PASSED [ 13%]
tests/test_backend_api.py::test_03_valid_prediction_24h_gate_component_id PASSED [ 20%]
tests/test_backend_api.py::test_04_valid_prediction_96h_gate              PASSED [ 26%]
tests/test_backend_api.py::test_05_invalid_sensor_values_rejected         PASSED [ 33%]
tests/test_backend_api.py::test_06_a24_temporal_leakage_prevention        PASSED [ 40%]
tests/test_backend_api.py::test_07_a96_temporal_leakage_prevention        PASSED [ 46%]
tests/test_backend_api.py::test_08_b24_prediction_output_schema           PASSED [ 53%]
tests/test_backend_api.py::test_09_b96_prediction_output_schema           PASSED [ 60%]
tests/test_backend_api.py::test_10_missing_value_handling                 PASSED [ 66%]
tests/test_backend_api.py::test_11_component_lookup_no_ground_truth       PASSED [ 73%]
tests/test_backend_api.py::test_12_ground_truth_not_in_predict_response   PASSED [ 80%]
tests/test_backend_api.py::test_13_model_files_not_modified               PASSED [ 86%]
tests/test_backend_api.py::test_bonus_component_id_predict_healthy        PASSED [ 93%]
tests/test_backend_api.py::test_bonus_component_id_predict_defective      PASSED [100%]

============================== 15 passed in 4.32s ==============================
```

### Existing Tests (unchanged, regression check)

```
tests/test_inference_pipeline.py  — 8 passed
tests/test_web_api.py             — 8 passed

Total: 31 passed, 0 failed across all 3 test files
```

---

## 8. How to Start the FastAPI Server

### Development (with auto-reload)
```powershell
cd D:\SIH
python -m uvicorn backend.main:app --reload --port 8000
```

### Production
```powershell
cd D:\SIH
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

After starting, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc UI:**   http://localhost:8000/redoc
- **Health:**     http://localhost:8000/api/health

### CORS for Custom Frontend Origin
```powershell
$env:ALLOWED_ORIGINS = "http://localhost:3000,http://localhost:5173"
python -m uvicorn backend.main:app --reload --port 8000
```

---

## 9. Required Additions to `requirements.txt`

Four new dependencies were added:

```
fastapi>=0.111.0          — Web framework
uvicorn[standard]>=0.29.0 — ASGI server
python-multipart>=0.0.9   — Form data support
httpx>=0.27.0             — TestClient HTTP library for pytest
```

---

## 10. Design Decisions & Safeguards

### Data Leakage Protection
The backend enforces **layered** ground-truth isolation:

1. **`component_service.py`** — strips `module_a_label`, `iddq_drift_168h_true`, `component_type` at source before data leaves the service.
2. **`src.inference.predict`** — existing temporal leakage assertions enforce that 96h features never enter A24/B24, and 168h columns never enter any model.
3. **Test 12** — scans the raw JSON response text for any occurrence of forbidden field names.

### Imputation
The backend does **not** fit a new imputer at inference time. The fitted `SimpleImputer` (median strategy, trained on the training set) is baked inside each serialised sklearn `Pipeline` object. Missing sensor values are forwarded directly to the pipeline, which imputes them using training-set medians.

### Decision Thresholds
The `PASS / REVIEW / REJECT` thresholds in `src.decision.screening_decision.DecisionConfig` are reused directly — they are evidence-calibrated from EDA data:
- Normal components: ~1% drift (max 2%)
- Drifting (latent defect): 5–15% drift
- Anomalous (gross defect): 20–40% drift

### Feature Importance
Feature importances are extracted from tree-based models only (`.feature_importances_` attribute). `LogisticRegression` (A24) returns an empty list rather than fabricated scores. Importance descriptions are labelled as **"features influencing model prediction"**, not physical causes.

### CORS
Allowed origins are read from the `ALLOWED_ORIGINS` environment variable. The default value covers `localhost:3000` and `localhost:5173` (common React/Vite dev servers). Production deployments must set this variable explicitly to the frontend domain.

---

## 11. Next Phase

```
FastAPI Backend (backend/)
         ↓
Frontend Dashboard (React / Vite)
         ↓
Final SIH 2026 Demonstration
```

The backend is ready for frontend integration. The frontend developer should:
1. Start the API with `uvicorn backend.main:app --reload --port 8000`
2. Read the interactive docs at `http://localhost:8000/docs`
3. Call `GET /api/components?split=test` to get the demo component list
4. Call `POST /api/predict` with `{ "component_id": "SYN_C..." }` to get predictions
