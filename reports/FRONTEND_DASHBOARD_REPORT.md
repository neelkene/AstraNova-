# Frontend Dashboard Report: AI-Driven Burn-In Screening

**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Stage:** Production Frontend Dashboard Implementation & Integration  
**Date:** 2026-09-03  
**Status:** ✅ Complete — All 34 automated tests passing; verified live on browser  

---

## 1. Executive Summary & Design Philosophy

The frontend for our SIH 2026 semiconductor burn-in screening platform has been built to meet the rigorous presentation standards of industrial IC reliability screening (AEC-Q100 / MIL-STD-883).

### Key Architectural & Design Principles:
1. **Strict Data Authenticity (Zero Fake Data Guarantee):**
   - Every single number, measurement, probability, prediction, and chart point originates directly from genuine project data (`data/raw/raw_burnin_data.csv`, `data/ml_ready/ml_features.csv`, `data/ground_truth/component_ground_truth.csv`) and the actual serialized models (`module_a_24h_logisticregression.joblib`, `module_a_96h_randomforest.joblib`, `module_b_24h_gradientboostingregressor.joblib`, `module_b_96h_randomforestregressor.joblib`).
   - No mock numbers, no random number generators, no hardcoded demo responses. Missing entries are shown as `N/A`.
2. **Professional Semiconductor Reliability Aesthetics:**
   - Crisp white background (`#ffffff` / `#f8fafc`) with subtle borders and shadows.
   - Restrained, high-contrast semiconductor palette: Slate 900 primary text, deep engineering navy branding, emerald for PASS, amber for REVIEW, and crimson for REJECT.
   - Clean monospace typography for component IDs, register addresses, and physical sensor readings.
   - Zero excessive gradients, zero neon colors, zero gaming effects, zero distracting animations.
3. **Physical Changes as the Primary Message:**
   - As mandated, the UI prioritizes **"How much did the parameter actually change in physical units?"** (e.g. `+2.50 μA`, `+0.12 ns`), with percentages presented only as secondary context.
4. **Transparent Decision Explainability:**
   - Distinguishes model correlation/feature influence from physical causation. Wording strictly uses: *"X is one of the most influential parameters in the model"* rather than claiming physical causation.

---

## 2. Frontend Structure & Integration

The frontend is served directly by our high-performance FastAPI backend (`backend/main.py`), allowing a single-command startup without cross-origin friction:

```text
SIH/
├── frontend/
│   ├── index.html                    ← Semantic HTML5 SPA dashboard layout
│   └── static/
│       ├── css/
│       │   └── style.css             ← Industrial clean design system (17 KB)
│       └── js/
│           ├── chart.umd.min.js      ← Local Chart.js bundle (205 KB, 100% offline)
│           └── app.js                ← Reactive API client & chart controllers (41 KB)
│
├── backend/
│   ├── main.py                       ← Mounts /static, serves index.html at /, handles API
│   ├── schemas.py                    ← Pydantic schemas (added 168h + dataset overview)
│   └── services/
│       └── component_service.py      ← Safe dataset lookup (no ground truth leakage)
│
├── tests/
│   ├── test_backend_api.py           ← 18 comprehensive backend & frontend tests
│   ├── test_inference_pipeline.py    ← 8 inference tests
│   └── test_web_api.py               ← 8 API tests
│
└── reports/
    └── FRONTEND_DASHBOARD_REPORT.md  ← This deliverable report
```

---

## 3. Detailed Dashboard Features & Layout

### A. Header & System Status
- **Title:** AI-Driven Burn-In Screening
- **Subtitle:** Predictive component reliability analysis using 24h and 96h burn-in data
- **Status Indicators:**
  - `System Online: All 4 Models Loaded` (live health probe connected to `/api/health`)
  - Active component ID (`SYN_C01216`)
  - Active test gate badge (`24h Early Warning` vs `96h Qualification` vs `Dual-Gate`)
  - Last analysis timestamp

### B. Component Selection & Presets
- **Real Component Search:** Interactive search box filtering all 10,000 components in real time.
- **Curated Demonstration Presets:**
  1. **Normal (`SYN_C01216`):** Pristine normal component from locked test set. Demonstrates nominal baseline, passing 96h qualification with minimal projected drift ($+0.97\%$).
  2. **Drifting (`SYN_C01252`):** Latent marginal defect. Appears borderline at 24h (REVIEW), continues stress testing, and is decisively rejected at 96h gate ($+9.09\%$ drift).
  3. **Gross Defect (`SYN_C04946`):** Severe early defect. Triaged and rejected immediately at the 24h early gate ($99.9\%$ defect probability, $+38.08\%$ drift), saving 144 hours of chamber stress.
- **Partition Filter:** Toggle between "All Components" and "Locked Test Set ($N=1,500$)".
- **Custom ID Loader:** Input any valid component ID from the dataset.

### C. Burn-In Progression Stepper & Controls
- Visual stage progression stepper:
  `0h Baseline → 24h Early Gate → 96h Qualification → 168h Forecast`
- One-click screening controllers:
  - `Run 24h Early Gate (Gate 1)`
  - `Run 96h Qualification (Gate 2)`
  - `Complete Dual-Gate Assessment`

### D. Final Screening Decision Hero Banner
- Prominently displays the combined verdict: **`PASS`**, **`REVIEW`**, or **`REJECT`**.
- Explains the exact evidence-based rationale synthesized from Module A (classification risk) and Module B (continuous degradation forecast).
- Provides actionable engineering recommendation (e.g. *"Release component to production inventory"* or *"Eject component from burn-in chamber immediately to save test resources"*).
- Displays model confidence level (`HIGH`, `MEDIUM`, `LOW`).

### E. KPI Metric Summary Cards
1. **Observed IDDQ Shift:** Actual physical difference (e.g. `+2.50 μA`, secondary `+2.56%`).
2. **Module A Defect Risk:** Calibrated defect probability ($p \in [0.0, 1.0]$) and discrete class.
3. **Module B 168h Forecast:** Estimated physical shift at 168h in $\mu\text{A}$ and projected drift $\%$.
4. **Chamber Time Optimization:** Quantifies operational hours saved ($72\text{h}$ or $144\text{h}$) compared to traditional full $168\text{h}$ burn-in.

### F. Actual Physical Measurements Table
Displays physical sensor readings across all burn-in checkpoints:
- **Parameters:**
  - $I_{DDQ}$ (Quiescent Current) with tooltip: *"IDDQ = quiescent supply current measured while the component is not actively switching."*
  - Leakage Current ($\mu\text{A}$)
  - Propagation Delay ($\text{ns}$)
  - Regulated Supply Voltage ($\text{V}$)
  - Junction Temperature ($^\circ\text{C}$)
- **Columns:** Parameter, 0h Baseline, 24h Early, 96h Mid, 168h Benchmark, 0h $\to$ 24h Physical Delta, 0h $\to$ 96h Physical Delta.
- Shows physical unit changes (e.g. `+2.50 μA`, `+0.12 ns`) as the primary visual element, with percentages in secondary muted text.

### G. Module A vs Module B Dual Comparison
- **Module A (Early Anomaly / Drift Screening):**
  - Displays A24 (Logistic Regression, 11 features) vs A96 (Random Forest, 19 features).
  - Shows prediction class, defect probability, algorithm, and feature counts.
  - Clearly labels temporal gates: `24h = Early Warning`, `96h = Higher-Confidence Screening`.
- **Module B (168h Continuous Degradation Forecast):**
  - Displays B24 (Gradient Boosting Regressor) vs B96 (Random Forest Regressor).
  - Displays predicted 168h IDDQ drift percentage and estimated physical current rise in $\mu\text{A}$.
  - Explains the forecast in plain language.

### H. Professional Charts (Real Data Only)
1. **Chart 1 — Parameter Progression Line Chart:**
   - Shows actual sensor readings progression across $0\text{h} \to 24\text{h} \to 96\text{h} \to 168\text{h}$.
   - Includes parameter switch tabs for IDDQ, Leakage Current, and Propagation Delay.
2. **Chart 2 — Parameter Physical Shift Bar Chart:**
   - Compares physical deltas across IDDQ ($\mu\text{A}$), Leakage Current ($\mu\text{A}$), and Delay ($\text{ns}$).
3. **Chart 3 — 168h Forecast Comparison:**
   - Compares measured 0h, 24h, and 96h readings against the Module B predicted 168h target and the true 168h measured benchmark.
4. **Chart 4 — Model Feature Importance (Degradation Drivers):**
   - Horizontal bar chart displaying normalized Gini feature importance from the trained tree models.
   - Displays cautious text: *"Parameters with highest Gini split importance in the trained model. Describes model feature influence, not physical causation."*
   - Transparently handles the 24h gate (linear model) by displaying that linear coefficients do not represent Gini importance, rather than fabricating a fake chart.
5. **Chart 5 — Dataset Population Overview Donut:**
   - Donut chart depicting the authentic distribution from `component_ground_truth.csv`:
     - **Normal Components:** $7,000$ ($70.0\%$)
     - **Latent Drifting:** $2,000$ ($20.0\%$)
     - **Gross Anomalous:** $1,000$ ($10.0\%$)

---

## 4. Section 18 Compliance & Validation Checklist

| # | Validation Item | Implementation / Verification Proof | Status |
|---|---|---|---|
| 1 | All displayed numbers come from real backend/API responses | Verified via network logs: `/api/components/{id}`, `/api/predict`, `/api/dataset-overview` | ✅ Verified |
| 2 | Search frontend code for hardcoded fake values | Grep search confirmed zero fake numbers or mock sensor readings in `app.js` and `index.html` | ✅ Verified |
| 3 | Remove all dummy data | All placeholder arrays removed; missing values render strictly as `N/A` | ✅ Verified |
| 4 | Actual trained Module A models used | Backend loads `module_a_24h_logisticregression.joblib` and `module_a_96h_randomforest.joblib` | ✅ Verified |
| 5 | Actual trained Module B models used | Backend loads `module_b_24h_gradientboostingregressor.joblib` and `module_b_96h_randomforestregressor.joblib` | ✅ Verified |
| 6 | 24h inference uses only 0h and 24h info | Tested via `test_06_a24_temporal_leakage_prevention`; strictly 11 features | ✅ Verified |
| 7 | 96h inference uses only 0h, 24h, and 96h info | Tested via `test_07_a96_temporal_leakage_prevention`; strictly 19 features | ✅ Verified |
| 8 | 168h measurements never used as input for early prediction | Assertions in `build_features.py` and `predict.py` enforce exclusion | ✅ Verified |
| 9 | Locked test set not modified | File sizes and MD5s of datasets and model artifacts intact (Test 13) | ✅ Verified |
| 10 | Frontend does not retrain models | Pure inference client calling existing backend endpoints | ✅ Verified |
| 11 | All charts use real data | Charts 1, 2, 3, 4, 5 directly bind to actual API payloads and component readings | ✅ Verified |
| 12 | Missing values handled honestly | Null sensor entries display `N/A` with neutral badges | ✅ Verified |
| 13 | Units displayed correctly | $\mu\text{A}$, $\text{ns}$, $\text{V}$, $^\circ\text{C}$, $\%$ displayed consistently with tooltips | ✅ Verified |
| 14 | Final decision comes from backend logic | `final_decision` synthesized directly from `screening_decision.py:DecisionConfig` | ✅ Verified |
| 15 | Demonstration flow works start to finish | Verified in live browser across normal, drifting, and defect components | ✅ Verified |

---

## 5. Automated Verification Results

All 34 automated unit and integration tests passed in 5.35 seconds:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\SIH
collected 34 items

tests/test_backend_api.py::test_01_model_loading PASSED                  [  2%]
tests/test_backend_api.py::test_02_health_endpoint PASSED                [  5%]
tests/test_backend_api.py::test_03_valid_prediction_24h_gate_component_id PASSED [  8%]
tests/test_backend_api.py::test_04_valid_prediction_96h_gate PASSED      [ 11%]
tests/test_backend_api.py::test_05_invalid_sensor_values_rejected PASSED [ 14%]
tests/test_backend_api.py::test_06_a24_temporal_leakage_prevention PASSED [ 17%]
tests/test_backend_api.py::test_07_a96_temporal_leakage_prevention PASSED [ 20%]
tests/test_backend_api.py::test_08_b24_prediction_output_schema PASSED   [ 23%]
tests/test_backend_api.py::test_09_b96_prediction_output_schema PASSED   [ 26%]
tests/test_backend_api.py::test_10_missing_value_handling PASSED         [ 29%]
tests/test_backend_api.py::test_11_component_lookup_no_ground_truth PASSED [ 32%]
tests/test_backend_api.py::test_12_ground_truth_not_in_predict_response PASSED [ 35%]
tests/test_backend_api.py::test_13_model_files_not_modified PASSED       [ 38%]
tests/test_backend_api.py::test_bonus_component_id_predict_healthy PASSED [ 41%]
tests/test_backend_api.py::test_bonus_component_id_predict_defective PASSED [ 44%]
tests/test_backend_api.py::test_dataset_overview_endpoint PASSED         [ 47%]
tests/test_backend_api.py::test_frontend_index_serves_successfully PASSED [ 50%]
tests/test_backend_api.py::test_frontend_static_assets_serve_successfully PASSED [ 52%]
tests/test_inference_pipeline.py::test_models_load_successfully PASSED   [ 55%]
tests/test_inference_pipeline.py::test_24h_feature_subset_strictly_bounded PASSED [ 58%]
tests/test_inference_pipeline.py::test_96h_feature_subset_strictly_bounded PASSED [ 61%]
tests/test_inference_pipeline.py::test_forbidden_168h_measurements_rejected PASSED [ 64%]
tests/test_inference_pipeline.py::test_prediction_output_schema_and_types PASSED [ 67%]
tests/test_inference_pipeline.py::test_screening_decision_rules PASSED   [ 70%]
tests/test_inference_pipeline.py::test_sequential_screening_workflow PASSED [ 73%]
tests/test_inference_pipeline.py::test_serialized_models_and_datasets_untouched PASSED [ 76%]
tests/test_web_api.py::test_health_endpoint PASSED                       [ 79%]
tests/test_web_api.py::test_get_test_components PASSED                   [ 82%]
tests/test_web_api.py::test_get_component_measurements PASSED            [ 85%]
tests/test_web_api.py::test_screen_24h_endpoint PASSED                   [ 88%]
tests/test_web_api.py::test_screen_96h_endpoint PASSED                   [ 91%]
tests/test_web_api.py::test_sequential_screening_endpoint PASSED         [ 94%]
tests/test_web_api.py::test_model_performance_endpoint PASSED            [ 97%]
tests/test_web_api.py::test_index_page_serves_successfully PASSED        [100%]

============================= 34 passed in 5.35s ==============================
```

---

## 6. How to Launch and Demonstrate to SIH Judges

### Single-Command Start:
```powershell
cd D:\SIH
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Presentation Walkthrough:
1. Open browser to **`http://127.0.0.1:8000/`**.
2. **First 15 Seconds:** Point to the Header status (`System Online: All 4 Models Loaded`) and the **Final Screening Decision Hero Card** (`PASS` with confidence level `HIGH`).
3. **Preset 1 (Healthy `SYN_C01216`):** Click the "Normal" preset button. Show the judge that IDDQ increases by only $+2.50\ \mu\text{A}$ ($+2.56\%$), Module A gives a low defect probability of $2.5\%$, and Module B projects $+0.97\%$ degradation. Result: **`PASS`**.
4. **Preset 2 (Latent Drifting `SYN_C01252`):** Click the "Drifting" preset button. Click the `24h Early Gate` button: notice the component receives `REVIEW` because the signal is marginal. Now click `96h Qualification`: the defect probability jumps to $81.8\%$ and projected degradation reaches $+9.09\%$. Result: **`REJECT`**. Explain to the judge how this prevents a latent field escape.
5. **Preset 3 (Gross Defect `SYN_C04946`):** Click the "Defect" preset button. Point to the $99.9\%$ defect probability at 24h and $+38.08\%$ drift forecast. Result: early **`REJECT`** at 24h, saving 144 hours of thermal chamber stress.
6. **Detailed Charts:** Show the judge Chart 1 (progression from 0h through 168h), Chart 2 (physical deltas), Chart 3 (measured vs predicted degradation), and Chart 4 (feature weights).
7. **Dataset Overview:** Switch to the "Dataset Overview" tab to display the authentic distribution of all 10,000 components ($70\%$ normal, $20\%$ drifting, $10\%$ anomalous) and explain the burn-in gate timeline.
