# Web Application Integration Report

**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Report Stage:** Final Web Application & Judge Demonstration Layer  
**Server Address:** http://127.0.0.1:5000 (local demo server)

---

## 1. Architecture Overview

The final SIH 2026 demonstration system consists of three integrated layers:

```
┌───────────────────────────────────────────────────────────────────────┐
│                       JUDGE DEMONSTRATION LAYER                       │
│         Browser-based Dashboard (4 Tabbed Views, Light Theme)         │
│  web/templates/index.html + web/static/css/style.css + js/app.js      │
└────────────────────────────┬──────────────────────────────────────────┘
                             │ REST API (JSON, HTTP)
┌────────────────────────────▼──────────────────────────────────────────┐
│                      FLASK BACKEND / API LAYER                        │
│               web/app.py  (Flask 3.1.3, Python 3.13)                  │
│  GET /api/health                                                       │
│  GET /api/test-components              (locked test partition only)    │
│  GET /api/component/<id>               (genuine sensor measurements)  │
│  POST /api/screen/24h                  (A24 + B24 + decision)         │
│  POST /api/screen/96h                  (A96 + B96 + decision)         │
│  POST /api/screen/sequential           (2-stage early-exit workflow)  │
│  GET /api/model-performance            (documented benchmark metrics)  │
└────────────────────────────┬──────────────────────────────────────────┘
                             │ Direct function calls (no subprocess)
┌────────────────────────────▼──────────────────────────────────────────┐
│                 PRODUCTION ML INFERENCE & DECISION LAYER               │
│  src/inference/predict.py   →  load_models(), predict_24h()/96h()     │
│  src/decision/screening_decision.py  →  run_screening_pipeline()       │
│  src/features/build_features.py      →  FEATURES_24H/96H_GATE         │
│  models/*.joblib  →  4 serialized sklearn Pipeline artifacts           │
│  data/ml_ready/ml_features.csv  →  10,000 genuine components          │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Dashboard (4-View Single Page App)

| View | Description |
| :--- | :--- |
| **Overview Dashboard** | Section 1: Burn-In workflow flowchart (0h → 24h → 96h). Section 2: Module A metrics card (Recall 73.78%→94.00%). Section 3: Module B metrics card (RMSE 4.033%→1.415%). Section 4: Why These Models (selection rationale). Section 5: Temporal Leakage Protection. |
| **Live Component Tester** | Left sidebar: full locked test-set list (N=1,500) with filter and search. Quick judge presets (Normal, Drifting, Anomalous). Right panel: genuine sensor readout table (0h/24h/96h). 3 action buttons: Run 24h, Run 96h, Sequential workflow. Live PASS/REVIEW/REJECT card with defect probability and drift forecast. |
| **Model Performance & Judge Q&A** | Full benchmark tables for Module A and Module B. Delta improvement row. 5-item Q&A accordion with grounded technical answers. |
| **Architecture & Leakage Protection** | Dataset inventory, model artifact table, feature availability matrix. |

**Design system:** Inter/JetBrains Mono, light white/slate background, brand blue (#2563eb) + teal (#0d9488), status colors (green/amber/red), rounded cards, whitespace-heavy layout.

---

## 3. REST API Endpoints

| Endpoint | Method | Purpose | Data Source |
| :--- | :--- | :--- | :--- |
| `/api/health` | GET | Server liveness + model status | N/A |
| `/api/test-components` | GET | Locked test component IDs + metadata | `data/ml_ready/ml_features.csv` |
| `/api/component/<id>` | GET | Genuine 0h/24h/96h sensor measurements | `data/ml_ready/ml_features.csv` |
| `/api/screen/24h` | POST | A24 + B24 inference + 24h decision | `models/module_a_24h_logisticregression.joblib` + `module_b_24h_gradientboostingregressor.joblib` |
| `/api/screen/96h` | POST | A96 + B96 inference + 96h decision | `models/module_a_96h_randomforest.joblib` + `module_b_96h_randomforestregressor.joblib` |
| `/api/screen/sequential` | POST | 2-stage early-exit workflow | All 4 models |
| `/api/model-performance` | GET | Documented locked test metrics | `reports/MODULE_A_TRAINING_REPORT.md` + `MODULE_B_TRAINING_REPORT.md` |

---

## 4. Data Flow — Genuine Data Guarantee

```
User selects test component (e.g. SYN_C04946)
   ↓
GET /api/component/SYN_C04946
   → Reads from ml_features.csv (genuine, never modified)
   → Returns actual sensor readings at 0h, 24h, 96h
   → Returns ground truth label + true 168h drift

POST /api/screen/24h  {component_id: "SYN_C04946"}
   → prepare_inference_features(row, gate="24h")
   → Extracts FEATURES_24H_GATE (11 columns: 0h + 24h only)
   → clf_a24.predict_proba(X) → Defect Probability = 99.9%
   → reg_b24.predict(X) → Forecast 168h Drift = 38.08%
   → make_screening_decision(0.999, 0.3808, "24h", DEFAULT_CONFIG)
   → Decision = REJECT | Reason = "Severe early defect signature..."
```

- **Zero 96h data enters 24h feature matrix** (enforced in `prepare_inference_features`)
- **Zero 168h sensor data enters any feature matrix** (hardcoded exclusion + validated in tests)
- **No random or fabricated data anywhere in the pipeline**
- **All component IDs come from the actual dataset**

---

## 5. Model Integration Verification

| Model File | Role | Input Dimensionality | Verified End-to-End |
| :--- | :--- | :--- | :--- |
| `module_a_24h_logisticregression.joblib` | A24 Classification | 11 features | ✅ |
| `module_a_96h_randomforest.joblib` | A96 Classification | 19 features | ✅ |
| `module_b_24h_gradientboostingregressor.joblib` | B24 Regression | 11 features | ✅ |
| `module_b_96h_randomforestregressor.joblib` | B96 Regression | 19 features | ✅ |

---

## 6. Judge Demonstration Workflow

A judge can demonstrate the full system in ~3 minutes:

1. **Open** http://127.0.0.1:5000
2. **Overview tab:** Walk through the Burn-In workflow flowchart → Module A card → Module B card → show the FNR drop (26.22% → 6.00%) and RMSE improvement (4.033% → 1.415%).
3. **Live Component Tester tab:**
   - Click **Gross Anomalous preset (SYN_C04946)** → observe `True 168h Drift = +35.62%`
   - Click **Run 24h Early Screening** → model outputs `REJECT`, Defect Prob = 99.9%, Forecast Drift = 38.08%
   - Click **Latent Drifting preset (SYN_C01252)** → observe moderate values
   - Click **Execute 2-Stage Sequential Workflow** → see REVIEW at 24h, REJECT at 96h
   - Click **Healthy preset (SYN_C01216)** → run sequential → PASS at 96h
4. **Model Performance tab:** Answer any metric question from the comparison tables and Q&A section.
5. **Architecture tab:** Show dataset inventory and model file table.

---

## 7. Test Verification Summary

```
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1
rootdir: D:\SIH
collected 16 items

tests/test_inference_pipeline.py::test_models_load_successfully PASSED
tests/test_inference_pipeline.py::test_24h_feature_subset_strictly_bounded PASSED
tests/test_inference_pipeline.py::test_96h_feature_subset_strictly_bounded PASSED
tests/test_inference_pipeline.py::test_forbidden_168h_measurements_rejected PASSED
tests/test_inference_pipeline.py::test_prediction_output_schema_and_types PASSED
tests/test_inference_pipeline.py::test_screening_decision_rules PASSED
tests/test_inference_pipeline.py::test_sequential_screening_workflow PASSED
tests/test_inference_pipeline.py::test_serialized_models_and_datasets_untouched PASSED
tests/test_web_api.py::test_health_endpoint PASSED
tests/test_web_api.py::test_get_test_components PASSED
tests/test_web_api.py::test_get_component_measurements PASSED
tests/test_web_api.py::test_screen_24h_endpoint PASSED
tests/test_web_api.py::test_screen_96h_endpoint PASSED
tests/test_web_api.py::test_sequential_screening_endpoint PASSED
tests/test_web_api.py::test_model_performance_endpoint PASSED
tests/test_web_api.py::test_index_page_serves_successfully PASSED

============================== 16 passed in 3.61s ==============================
```

Live API smoke test (against running server):

```
[PASS] GET /api/health
[PASS] GET /api/test-components?limit=5
[PASS] GET /api/component/SYN_C01216
[PASS] GET /api/model-performance
[PASS] POST /api/screen/24h  -> REJECT | Prob=99.9% | Drift=+38.08%
[PASS] POST /api/screen/96h  -> REJECT | Prob=100.0% | Drift=+36.85%
[PASS] POST /api/screen/sequential  -> PASS @ 96h | EarlyExit=False

API Smoke Test: 7 passed / 0 failed
```

---

## 8. Integrity & Scope Confirmations

| Constraint | Status |
| :--- | :--- |
| No Module C created | ✅ Confirmed |
| No model retraining or regeneration | ✅ Confirmed |
| No random/fabricated predictions | ✅ Confirmed |
| No fake component IDs | ✅ Confirmed |
| No invented metrics or thresholds | ✅ Confirmed |
| All 4 serialized model artifacts preserved, unmodified | ✅ Confirmed |
| All 3 dataset files preserved, unmodified | ✅ Confirmed |
| 168h sensor measurements excluded from all feature matrices | ✅ Confirmed |
| 96h data excluded from 24h gate feature matrix | ✅ Confirmed |
| All predictions from genuine serialized sklearn Pipeline artifacts | ✅ Confirmed |

---

## 9. Files Created (Web Layer Only)

| File | Purpose |
| :--- | :--- |
| [`web/app.py`](file:///d:/SIH/web/app.py) | Flask backend with all 7 REST API endpoints |
| [`web/templates/index.html`](file:///d:/SIH/web/templates/index.html) | 4-view judge demonstration dashboard |
| [`web/static/css/style.css`](file:///d:/SIH/web/static/css/style.css) | Light theme design system (CSS variables, cards, status badges) |
| [`web/static/js/app.js`](file:///d:/SIH/web/static/js/app.js) | Frontend client: navigation, component selector, API calls, result rendering |
| [`tests/test_web_api.py`](file:///d:/SIH/tests/test_web_api.py) | 8 Flask test-client API tests |

To start the dashboard: `python web/app.py` from `d:\SIH\`
