# Project Current-State Audit Report

**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Audit Type:** Read-Only Transition & Current-State Verification  
**Audit Timestamp:** 2026-09-02  
**Source of Truth:** Local Workspace Filesystem (`d:\SIH`)  

---

## 1. Actual Project Structure

The actual project structure on disk is organized as follows:

```text
SIH/
│
├── data/
│   ├── raw/
│   │   └── raw_burnin_data.csv                    (40,000 rows, 8 columns, 4.29 MB)
│   ├── ground_truth/
│   │   └── component_ground_truth.csv             (10,000 rows, 8 columns, 1.15 MB)
│   └── ml_ready/
│       └── ml_features.csv                        (10,000 rows, 27 columns, 4.73 MB)
│
├── eda/
│   ├── 01_eda.py                                  (Full EDA execution pipeline)
│   ├── 02_data_validation_audit.py                (Data integrity & 3-sigma validation)
│   └── outputs/
│       ├── data_quality_summary.csv
│       ├── drift_summary_by_component_type.csv
│       ├── ground_truth_class_distribution.csv
│       ├── latent_degradation_screening_escape.csv
│       ├── ml_features_validation_and_leakage_audit.csv
│       ├── raw_parameter_statistics.csv
│       ├── correlations/
│       │   ├── ml_features_correlation_matrix.png
│       │   └── raw_parameters_correlation_matrix.png
│       ├── degradation/
│       │   ├── component_trajectories_by_class.png
│       │   ├── drift_percentage_distributions.png
│       │   └── latent_degradation_static_limits_comparison.png
│       ├── distributions/
│       │   ├── iddq_uA_distribution_boxplot.png
│       │   ├── leakage_current_uA_distribution_boxplot.png
│       │   ├── propagation_delay_ns_distribution_boxplot.png
│       │   ├── temperature_C_distribution_boxplot.png
│       │   └── voltage_V_distribution_boxplot.png
│       ├── ground_truth/
│       │   └── ground_truth_baselines_and_true_drift.png
│       ├── ml/module_a/
│       │   ├── a24_confusion_matrix_test.png
│       │   ├── a96_confusion_matrix_test.png
│       │   ├── a96_feature_importances.png
│       │   ├── module_a_roc_curves_test.png
│       │   └── ablation/
│       │       ├── ablation_metrics_barplot.png
│       │       └── ablation_roc_comparison.png
│       └── validation/
│           ├── exact_escape_rates_audit.csv
│           ├── feature_leakage_and_availability_matrix.csv
│           ├── ground_truth_drift_distribution_by_class.csv
│           └── synthetic_data_difficulty_and_separability.csv
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py                           (Schema checks, target isolation)
│   │   └── split_data.py                          (Component-level stratified 70/15/15 split)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py                      (Gate feature definitions: A24, A96, B24, B96)
│   └── models/
│       ├── __init__.py
│       ├── evaluate.py                            (Classification and regression evaluation metrics)
│       ├── train_module_a.py                      (Module A training, selection, locked test evaluation)
│       ├── ablation_module_a.py                   (Controlled robustness and feature ablation study)
│       └── train_module_b.py                      (Module B model definitions & experiment scaffold)
│
├── models/
│   ├── module_a_24h_logisticregression.joblib     (Serialized best A24 model artifact)
│   └── module_a_96h_randomforest.joblib          (Serialized best A96 model artifact)
│
├── reports/
│   ├── EDA_REPORT.md                              (Completed EDA findings & physical analysis)
│   ├── DATA_VALIDATION_REPORT.md                  (Completed 3-sigma & leakage audit)
│   ├── ML_EXPERIMENT_DESIGN.md                    (Completed experimental design & split protocol)
│   ├── MODULE_A_TRAINING_REPORT.md                (Completed Module A training & test results)
│   ├── MODULE_A_ABLATION_REPORT.md                (Completed Module A ablation & robustness report)
│   ├── module_a_validation_results.csv            (Validation benchmark across all candidates)
│   ├── module_a_test_results.csv                  (Locked test set performance for A24 & A96)
│   ├── module_a_gate_comparison.csv               (A96 vs A24 delta comparison table)
│   ├── module_a_ablation_results.csv              (Ablation metrics across 4 experiments)
│   └── module_a_ablation_comparison_table.csv     (Formatted ablation comparison table)
│
├── tests/
│   └── test_ml_pipeline_design.py                (Validation test suite for ML pipeline constraints)
│
├── LICENSE                                        (MIT License)
├── README.md                                      (Project overview and directory layout)
└── requirements.txt                               (Dependency specifications)
```

---

## 2. Dataset Inventory

| Dataset Path | Rows | Columns | Key Columns | Role & Usage Restrictions | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data/raw/raw_burnin_data.csv` | **40,000** | **8** | `component_id`, `hour`, `iddq_uA`, `leakage_current_uA`, `propagation_delay_ns`, `voltage_V`, `temperature_C`, `is_measurement_outlier` | **Immutable Raw Data:** Longitudinal sensor readings across 10,000 components at 4 test gates (0h, 24h, 96h, 168h). Contains ~1.5% realistic sensor dropout. | ✅ Verified (0 duplicates, pristine integrity) |
| `data/ground_truth/component_ground_truth.csv` | **10,000** | **8** | `component_id`, `component_type` (7,000 normal, 2,000 drifting, 1,000 anomalous), `module_a_label` (7,000 '0', 3,000 '1'), `iddq_baseline`, `leakage_baseline`, `delay_baseline`, `voltage_baseline`, `iddq_drift_168h_true` | **Ground Truth Metadata:** Ground-truth labels and generative baselines. **STRICTLY EXCLUDED** from ML feature matrix $X$; used for benchmarking only. | ✅ Verified (0 missing values, complete) |
| `data/ml_ready/ml_features.csv` | **10,000** | **27** | `component_id`, 20 pivoted sensor columns (0h, 24h, 96h, 168h), `module_a_label`, `iddq_drift_168h_true`, `iddq_drift_24h_pct`, `iddq_drift_96h_pct`, `leakage_drift_96h_pct`, `delay_drift_96h_pct` | **Prepared Feature Matrix:** Multi-point features and engineered drift metrics for Module A & Module B modeling. | ✅ Verified (Schema assertions passed) |

---

## 3. EDA Status

* **Execution Script:** `eda/01_eda.py` (Fully implemented and executed).
* **Comprehensive Report:** `reports/EDA_REPORT.md` (Generated and complete).
* **Generated Visualizations & CSVs:**
  - `eda/outputs/data_quality_summary.csv`
  - `eda/outputs/raw_parameter_statistics.csv`
  - `eda/outputs/drift_summary_by_component_type.csv`
  - `eda/outputs/ground_truth_class_distribution.csv`
  - `eda/outputs/latent_degradation_screening_escape.csv`
  - `eda/outputs/ml_features_validation_and_leakage_audit.csv`
  - 10 publication-quality PNG figures in `eda/outputs/distributions/`, `correlations/`, `degradation/`, `ground_truth/`.
* **Key Validated Findings:**
  1. **Static 3-Sigma Limit Failure:** $98.25\%$ of drifting components escape static limits at $0\text{h}$; $97.95\%$ escape at $24\text{h}$; $81.60\%$ remain within nominal static bounds even after $168\text{h}$.
  2. **Trajectory Dynamics:** Normal components exhibit $+1.02\%$ mean $I_{DDQ}$ drift at $168\text{h}$, drifting components exhibit $+10.08\%$ mean drift, and anomalous components exhibit $+29.87\%$ mean drift.
* **Status:** **COMPLETE.**

---

## 4. Data Validation Status

* **Execution Script:** `eda/02_data_validation_audit.py` (Fully implemented and executed).
* **Reports:** `reports/DATA_VALIDATION_REPORT.md` and `reports/ML_EXPERIMENT_DESIGN.md`.
* **Safeguards & Audits Established:**
  1. **Target Leakage:** `module_a_label` and `iddq_drift_168h_true` are isolated into target vector $y$.
  2. **Temporal Future Leakage:** 24h gate models ($A24$, $B24$) strictly exclude $96\text{h}$ and $168\text{h}$ measurements; 96h gate models ($A96$, $B96$) strictly exclude $168\text{h}$ measurements.
  3. **Component ID Isolation:** `component_id` is excluded from all feature matrices.
  4. **Component-Level Stratified Partition:** 
     - **Train (70%):** 7,000 components (4,900 Normal, 2,100 Defective)
     - **Validation (15%):** 1,500 components (1,050 Normal, 450 Defective)
     - **Locked Test (15%):** 1,500 components (1,050 Normal, 450 Defective)
     - Zero component ID overlap asserted across all partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
* **Status:** **COMPLETE.**

---

## 5. Module A Status (Early Anomaly / Drift Classification)

* **Implemented:** ✅ Yes (`src/models/train_module_a.py`, `src/features/build_features.py`, `src/models/evaluate.py`).
* **Trained:** ✅ Yes (Candidate models `LogisticRegression`, `RandomForest`, `GradientBoosting` trained on Train set).
* **Validation Benchmarked & Model Selected:** ✅ Yes (Documented in `reports/module_a_validation_results.csv`).
  - Best A24 Model: `LogisticRegression (Balanced)` (Val Recall: $70.00\%$, FNR: $30.00\%$, F1: $0.6826$, ROC-AUC: $0.8473$).
  - Best A96 Model: `RandomForestClassifier` (Val Recall: $95.56\%$, FNR: $4.44\%$, F1: $0.9566$, ROC-AUC: $0.9953$).
* **Locked Test Set Evaluated:** ✅ Yes (Documented in `reports/module_a_test_results.csv` and `reports/MODULE_A_TRAINING_REPORT.md`).
  - **A24 Test Performance:** Accuracy = $82.40\%$, Precision = $69.46\%$, **Recall (Class 1) = $73.78\%$**, F1 = $0.7155$, **FNR = $26.22\%$**, FPR = $13.90\%$, **ROC-AUC = $0.8719$**.
  - **A96 Test Performance:** Accuracy = $97.67\%$, Precision = $98.14\%$, **Recall (Class 1) = $94.00\%$**, F1 = $0.9603$, **FNR = $6.00\%$**, FPR = $0.76\%$, **ROC-AUC = $0.9944$**.
* **Ablation & Robustness Tested:** ✅ Yes (`src/models/ablation_module_a.py`, `reports/MODULE_A_ABLATION_REPORT.md`, `reports/module_a_ablation_results.csv`, `reports/module_a_ablation_comparison_table.csv`).
  - Confirmed raw sensor features provide a strong base ($80.89\%$ Recall at 96h), while engineered drift features boost Recall to $94.00\%$ and reduce FNR from $19.11\%$ to $6.00\%$.
* **Serialized Model Artifacts:** ✅ Yes:
  - `models/module_a_24h_logisticregression.joblib`
  - `models/module_a_96h_randomforest.joblib`
* **Status:** **COMPLETE.**

---

## 6. Module B Status (168h Continuous Degradation Forecasting)

* **Feature Engineering Definitions:** ✅ Defined in `src/features/build_features.py` (B24: 11 features, B96: 19 features, Target: `iddq_drift_168h_true`).
* **Evaluation Functions:** ✅ Defined in `src/models/evaluate.py` (`evaluate_regression` computing MAE, RMSE, $R^2$).
* **Training Scaffolding:** ⚠️ Partial in `src/models/train_module_b.py` (`get_module_b_models`, `build_module_b_pipeline`, `prepare_module_b_experiment`).
* **Execution & Training Pipeline:** ❌ NOT implemented or executed (`run_module_b_training()` is missing; models have not been fit).
* **Trained Model Artifacts:** ❌ None serialized in `models/`.
* **Validation & Test Evaluation:** ❌ NOT evaluated on dataset partitions.
* **Reports & Results CSVs:** ❌ None generated (no `reports/MODULE_B_TRAINING_REPORT.md` or regression CSV tables).
* **Status:** **PENDING / INCOMPLETE.**

---

## 7. Tests Status

* **Test Files Present:** `tests/test_ml_pipeline_design.py`.
* **Execution Audit:**
  - Test script exercises schema validation, component-level partition disjointness, and experiment feature/target preparation.
  - Test currently references older function names (`prepare_module_a_experiment`, `get_module_a_models`) from `src.models.train_module_a`, where `train_module_a.py` has since been streamlined to `get_candidate_models` and `train_and_evaluate_gate`.
  - Host Python environment requires `scikit-learn` to run automated test commands via CLI.
* **Status:** **Requires minor test function alignment.**

---

## 8. Missing Components

1. **Module B End-to-End Training Pipeline:** Complete execution function in `src/models/train_module_b.py` for training candidate regressors (Ridge, Random Forest Regressor, Gradient Boosting Regressor) at 24h (B24) and 96h (B96) gates.
2. **Module B Serialized Artifacts:** Serialized regressors in `models/` (e.g. `module_b_24h_*.joblib`, `module_b_96h_*.joblib`).
3. **Module B Reports & Metrics:** `reports/MODULE_B_TRAINING_REPORT.md` and associated performance comparison CSV tables and diagnostic regression plots.
4. **Host Python Environment Setup:** Installing project dependencies (`requirements.txt`) in the local Python environment for execution.

---

## 9. Current Project Checkpoint

**CURRENT CHECKPOINT:**  
Module A (Early Anomaly & Drift Classification) is fully designed, implemented, trained, evaluated, ablation-tested, and serialized with all artifacts and reports complete. Module B (168h Continuous Degradation Forecasting) has its feature specifications and candidate definitions scaffolded, but has not yet been implemented as a runnable training pipeline, trained, evaluated, or serialized.

---

## 10. Recommended NEXT STEP

**NEXT STEP: Begin Module B — 168h Continuous Degradation Forecasting.**
