# Machine Learning Experiment Design & Pipeline Validation Report
**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Stage:** Pre-Training ML Pipeline Validation Checkpoint  
**Generated Date:** 2026-08-29  
**Source Architecture:** `src/data/`, `src/features/`, `src/models/`

---

## 1. Executive Summary & Validation Status

The ML experiment pipeline for **Module A (Early Anomaly/Drift Classification)** and **Module B (168h Degradation Prediction)** has been created, modularized, and validated against all temporal, target, and component leakage constraints.

```text
================================================================================
FINAL ML EXPERIMENT DESIGN SUMMARY
================================================================================

A24 (Early Screening @ 24h Gate):
  Features (11 total) = ['iddq_uA_0h', 'leakage_current_uA_0h', 'propagation_delay_ns_0h', 'voltage_V_0h', 'temperature_C_0h', 'iddq_uA_24h', 'leakage_current_uA_24h', 'propagation_delay_ns_24h', 'voltage_V_24h', 'temperature_C_24h', 'iddq_drift_24h_pct']
  Target              = module_a_label
  Future Info Used    = NO

A96 (Mid Screening @ 96h Gate):
  Features (19 total) = ['iddq_uA_0h', 'leakage_current_uA_0h', 'propagation_delay_ns_0h', 'voltage_V_0h', 'temperature_C_0h', 'iddq_uA_24h', 'leakage_current_uA_24h', 'propagation_delay_ns_24h', 'voltage_V_24h', 'temperature_C_24h', 'iddq_drift_24h_pct', 'iddq_uA_96h', 'leakage_current_uA_96h', 'propagation_delay_ns_96h', 'voltage_V_96h', 'temperature_C_96h', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct']
  Target              = module_a_label
  Future Info Used    = NO

B24 (Degradation Prediction @ 24h Gate):
  Features (11 total) = ['iddq_uA_0h', 'leakage_current_uA_0h', 'propagation_delay_ns_0h', 'voltage_V_0h', 'temperature_C_0h', 'iddq_uA_24h', 'leakage_current_uA_24h', 'propagation_delay_ns_24h', 'voltage_V_24h', 'temperature_C_24h', 'iddq_drift_24h_pct']
  Target              = iddq_drift_168h_true
  Future Info Used    = NO

B96 (Degradation Prediction @ 96h Gate):
  Features (19 total) = ['iddq_uA_0h', 'leakage_current_uA_0h', 'propagation_delay_ns_0h', 'voltage_V_0h', 'temperature_C_0h', 'iddq_uA_24h', 'leakage_current_uA_24h', 'propagation_delay_ns_24h', 'voltage_V_24h', 'temperature_C_24h', 'iddq_drift_24h_pct', 'iddq_uA_96h', 'leakage_current_uA_96h', 'propagation_delay_ns_96h', 'voltage_V_96h', 'temperature_C_96h', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct']
  Target              = iddq_drift_168h_true
  Future Info Used    = NO

--------------------------------------------------------------------------------
DATA SPLIT SUMMARY (Component-Level Stratified):
  Train Components      = 7,000 (70.0%) | 30.00% Defective (Class 1)
  Validation Components = 1,500 (15.0%) | 30.00% Defective (Class 1)
  Test Components       = 1,500 (15.0%) | 30.00% Defective (Class 1)
--------------------------------------------------------------------------------
MUTUAL EXCLUSIVITY CHECK: PASSED (Zero component ID overlap across partitions)
```

---

## 2. Project & Codebase Architecture

```text
SIH/
├── data/
│   ├── raw/raw_burnin_data.csv
│   ├── ground_truth/component_ground_truth.csv
│   └── ml_ready/ml_features.csv
│
├── eda/
│   ├── 01_eda.py
│   ├── 02_data_validation_audit.py
│   └── outputs/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py          # Data ingestion and schema validation
│   │   └── split_data.py         # Component-level stratified train/val/test splitting
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # Gate-specific feature extraction & preprocessing pipelines
│   │
│   └── models/
│       ├── __init__.py
│       ├── train_module_a.py     # Module A classification models & training harness
│       ├── train_module_b.py     # Module B regression models & training harness
│       └── evaluate.py           # Multi-metric evaluation and comparison logic
│
├── models/                       # Directory reserved for serialized model artifacts (.joblib)
├── reports/
│   ├── EDA_REPORT.md
│   ├── DATA_VALIDATION_REPORT.md
│   └── ML_EXPERIMENT_DESIGN.md
│
└── tests/
    └── test_ml_pipeline_design.py # End-to-end pipeline integrity test suite
```

---

## 3. Data Integrity & Leakage Prevention Protocol

### A. Target Leakage Prevention
1. **Module A Target (`module_a_label`):** Binary indicator ($0 = \text{Normal}, 1 = \text{Defective/Drifting}$). Extracted strictly into target vector $y_{\text{clf}}$. Never included in feature matrix $X$.
2. **Module B Target (`iddq_drift_168h_true`):** Continuous regression target ($168\text{h}$ percentage drift). Extracted strictly into target vector $y_{\text{reg}}$. Never included in feature matrix $X$.
3. **Ground Truth String Fields:** `component_type` (`normal`, `drifting`, `anomalous`) and baseline parameter columns from `component_ground_truth.csv` are strictly excluded from $X$.
4. **Identifier Isolation:** `component_id` is stripped from all feature matrices to eliminate sequence or index artifacts.

### B. Temporal Future Leakage Prevention
Burn-in testing decisions occur at discrete time gates. The feature sets are strictly bounded by temporal causality:

| Screening Gate Experiment | Allowed Time Points | Excluded (Forbidden) Future Time Points | Leakage Check Status |
| :--- | :--- | :--- | :--- |
| **A24 / B24** | **$0\text{h}, 24\text{h}$** | $96\text{h}$ measurements, $96\text{h}$ drift, $168\text{h}$ measurements | ✅ **Zero Future Leakage** |
| **A96 / B96** | **$0\text{h}, 24\text{h}, 96\text{h}$** | $168\text{h}$ measurements | ✅ **Zero Future Leakage** |

---

## 4. Component-Level Data Splitting

Because each component has multi-point temporal measurements, standard random shuffling at row level causes severe data contamination. 

We implemented a **Component-Level Stratified Partition**:
* **Train Set (70%):** $7,000$ components ($4,900$ Normal, $2,100$ Defective)
* **Validation Set (15%):** $1,500$ components ($1,050$ Normal, $450$ Defective)
* **Test Set (15%):** $1,500$ components ($1,050$ Normal, $450$ Defective)

```python
assert len(train_ids.intersection(val_ids)) == 0
assert len(train_ids.intersection(test_ids)) == 0
assert len(val_ids.intersection(test_ids)) == 0
```
*Guarantees zero overlap between training, tuning, and unbiased final evaluation.*

---

## 5. Missing Value & Preprocessing Pipeline

To handle the synthetic sensor dropout rate (~$1.5\%$ across electrical channels):
1. **Fit-Transform Isolation:** The imputer (`SimpleImputer(strategy='median')`) is fitted **ONLY on the Training set ($X_{\text{train}}$)** and subsequently used to transform $X_{\text{val}}$ and $X_{\text{test}}$.
2. **Feature Scaling:** `StandardScaler` is incorporated inside the pipeline for linear models (Logistic Regression, Ridge Regression) and omitted for tree ensembles (Random Forest, Gradient Boosting).
3. **No Component Dropping:** Median imputation retains $100\%$ of components ($10,000 / 10,000$), preserving data yield.

---

## 6. Model Suite & Evaluation Protocol

### A. Module A: Early Anomaly & Drift Classification
* **Algorithms:**
  1. `LogisticRegression` (Linear balanced baseline)
  2. `RandomForestClassifier` (Non-linear bagging baseline)
  3. `GradientBoostingClassifier` (Sequential tree boosting)
* **Evaluation Metrics:**
  * Accuracy, Precision, Recall (Class 1)
  * **False Negative Rate (FNR):** $\text{FNR} = \text{FN} / (\text{FN} + \text{TP})$ *(Primary reliability metric — defect escape rate)*
  * **False Positive Rate (FPR):** $\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$ *(Yield fallout rate)*
  * F1-Score, ROC-AUC, Confusion Matrix

### B. Module B: 168h Continuous Degradation Forecaster
* **Algorithms:**
  1. `LinearRegression / Ridge` (Linear baseline)
  2. `RandomForestRegressor` (Non-linear regression baseline)
  3. `GradientBoostingRegressor` (Gradient boosted regression)
* **Evaluation Metrics:**
  * Mean Absolute Error ($\text{MAE}$)
  * Root Mean Squared Error ($\text{RMSE}$)
  * Coefficient of Determination ($R^2$)

---

## 7. Next Step: Model Training & Comparison Plan

When instructed to proceed to the training phase, the execution will benchmark:
1. **A24 vs A96:** Quantify screening accuracy and FNR reduction achieved by running tests to $24\text{h}$ vs $96\text{h}$.
2. **B24 vs B96:** Quantify $R^2$ and RMSE improvement in forecasting final $168\text{h}$ degradation from early measurements.
3. **Model Selection:** Identify the optimal classifier and regressor based on test set generalization.

---
*Report validated and documented. Pipeline is halted and ready for model execution.*
