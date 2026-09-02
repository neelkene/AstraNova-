# Module B (168h Continuous Degradation Forecasting) Training & Evaluation Report

**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Module:** Module B — Continuous Degradation Forecaster  
**Target:** `iddq_drift_168h_true` (True continuous quiescent current degradation at 168h)  
**Execution Date:** 2026-09-02  
**Artifact Directory:** `models/` & `eda/outputs/ml/module_b/`  

---

## 1. Objective

Traditional burn-in testing requires running stress chambers for the full 168-hour qualification window to observe component degradation. The objective of **Module B (Continuous Degradation Forecaster)** is to quantitatively forecast a component's final end-of-test degradation ($\Delta I_{DDQ, 168\text{h}}$) at earlier burn-in test gates ($24\text{h}$ and $96\text{h}$).

By predicting the continuous parameter drift before completion:
1. Severely degrading components can be screened out or flagged immediately at $24\text{h}$, saving test chamber energy, thermal wear, and cycle time.
2. Marginal components can be tracked to determine whether they will exceed allowable parametric degradation thresholds by $168\text{h}$.
3. High-reliability screening decisions transition from reactive thresholding to proactive lifetime trajectory forecasting.

---

## 2. B24 Feature Definition (24h Screening Gate — 11 Features)

Experiment **B24** forecasts final 168h degradation using **ONLY** measurements and drift metrics available at the $24\text{h}$ early burn-in gate:

* **0h Pre-Burn-In Baselines (5 Features):**
  - `iddq_uA_0h`
  - `leakage_current_uA_0h`
  - `propagation_delay_ns_0h`
  - `voltage_V_0h`
  - `temperature_C_0h`
* **24h Early Burn-In Sensors (5 Features):**
  - `iddq_uA_24h`
  - `leakage_current_uA_24h`
  - `propagation_delay_ns_24h`
  - `voltage_V_24h`
  - `temperature_C_24h`
* **24h Calculated Parameter Drift (1 Feature):**
  - `iddq_drift_24h_pct` = $(I_{DDQ, 24\text{h}} - I_{DDQ, 0\text{h}}) / I_{DDQ, 0\text{h}}$

**Temporal Isolation:** Zero $96\text{h}$ or $168\text{h}$ measurements or features are permitted in B24.

---

## 3. B96 Feature Definition (96h Screening Gate — 19 Features)

Experiment **B96** forecasts final 168h degradation using measurements accumulated through the $96\text{h}$ mid burn-in gate:

* **All 11 B24 Features (0h baselines, 24h sensors, 24h drift)**
* **96h Mid Burn-In Sensors (5 Features):**
  - `iddq_uA_96h`
  - `leakage_current_uA_96h`
  - `propagation_delay_ns_96h`
  - `voltage_V_96h`
  - `temperature_C_96h`
* **96h Multi-Parameter Differential Drift (3 Features):**
  - `iddq_drift_96h_pct` = $(I_{DDQ, 96\text{h}} - I_{DDQ, 0\text{h}}) / I_{DDQ, 0\text{h}}$
  - `leakage_drift_96h_pct` = $(I_{leak, 96\text{h}} - I_{leak, 0\text{h}}) / I_{leak, 0\text{h}}$
  - `delay_drift_96h_pct` = $(t_{pd, 96\text{h}} - t_{pd, 0\text{h}}) / t_{pd, 0\text{h}}$

**Temporal Isolation:** Zero $168\text{h}$ sensor readings or end-of-test drift features are permitted in B96.

---

## 4. Target Definition

* **Target Variable:** `iddq_drift_168h_true`
* **Type:** Continuous ratio / percentage ($\in [0.00, +0.40]$) representing the true physical drift in quiescent current at the end of the $168\text{h}$ qualification test:
  $$\Delta I_{DDQ, 168\text{h, true}} = \frac{I_{DDQ, 168\text{h}} - I_{DDQ, \text{baseline}}}{I_{DDQ, \text{baseline}}}$$
* **Population Sub-Distributions:**
  - Normal components ($70\%$): Mean $\approx +0.0100$ ($+1.00\%$), Std $\approx 0.0058$
  - Drifting components ($20\%$): Mean $\approx +0.1001$ ($+10.01\%$), Std $\approx 0.0289$
  - Anomalous components ($10\%$): Mean $\approx +0.2992$ ($+29.92\%$), Std $\approx 0.0580$
* **Isolation Rule:** Extracted strictly into target vector $y_{\text{reg}}$. Never included in feature matrix $X$.

---

## 5. Data Split (Component-Level Stratified Partition)

To prevent longitudinal data contamination, data splitting is enforced at the **component level** using a fixed random seed (`random_state=42`):

| Partition | Component Count ($N$) | Percentage (%) | Mean Actual Degradation ($\mu_y$) | Std Dev ($\sigma_y$) |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | **7,000** | **70.0%** | $0.0543$ ($+5.43\%$) | $0.0863$ |
| **Validation Set** | **1,500** | **15.0%** | $0.0543$ ($+5.43\%$) | $0.0863$ |
| **Locked Test Set** | **1,500** | **15.0%** | $0.0550$ ($+5.50\%$) | $0.0877$ |
| **Total Population** | **10,000** | **100.0%** | **$0.0544$ ($+5.44\%$)** | **$0.0865$** |

* **Zero Leakage Assertions:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  All 4 temporal observations for any given component ID belong strictly to a single partition.

---

## 6. Preprocessing & Leak-Free Pipeline

1. **Missing Value Imputation:** Realistic sensor noise and dropouts (~$1.5\%$ per measurement channel) are imputed using `SimpleImputer(strategy='median')`.
2. **Fit-Transform Boundary:** The imputer and scalers are fitted **STRICTLY on the Training set ($X_{\text{train}}$)** and then applied to transform $X_{\text{val}}$ and $X_{\text{test}}$.
3. **Feature Scaling:** `StandardScaler` is applied for linear models (`LinearRegression`, `Ridge`) and omitted for tree ensembles (`RandomForestRegressor`, `GradientBoostingRegressor`).
4. **Pipeline Encapsulation:** Imputation, scaling, and regressor estimation are bundled inside a single `sklearn.pipeline.Pipeline` to eliminate leakage risks.

---

## 7. Models Tested

Four regression algorithms representing linear and non-linear families were benchmarked:

1. **LinearRegression:** Unregularized OLS linear regression with feature standardization.
2. **Ridge Regression:** L2-regularized linear model ($\alpha=1.0$) with feature standardization.
3. **RandomForestRegressor:** Bagged tree ensemble ($100$ estimators, `max_depth=10`, `random_state=42`).
4. **GradientBoostingRegressor:** Sequential boosted trees ($100$ estimators, `learning_rate=0.1`, `max_depth=5`, `random_state=42`).

---

## 8. B24 Results (Validation Benchmark: $N=1,500$)

Candidate models were trained on $X_{\text{train}}$ ($7,000$ parts) and evaluated on the independent Validation partition ($1,500$ parts):

| Candidate Model | Val RMSE | Val RMSE (%) | Val MAE | Val MAE (%) | Val $R^2$ Score | Model Selection Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LinearRegression` | $0.067920$ | $6.792\%$ | $0.044090$ | $4.409\%$ | $0.3804$ | Underfits early non-linear drift |
| `Ridge` | $0.067911$ | $6.791\%$ | $0.044090$ | $4.409\%$ | $0.3806$ | Underfits early non-linear drift |
| `RandomForestRegressor` | $0.043848$ | $4.385\%$ | $0.030024$ | $3.002\%$ | $0.7418$ | Strong non-linear capture |
| **`GradientBoostingRegressor`** | **$0.041570$** | **$4.157\%$** | **$0.028473$** | **$2.847\%$** | **$0.7679$** | 🏆 **SELECTED BEST (B24)** |

---

## 9. B96 Results (Validation Benchmark: $N=1,500$)

| Candidate Model | Val RMSE | Val RMSE (%) | Val MAE | Val MAE (%) | Val $R^2$ Score | Model Selection Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LinearRegression` | $0.029309$ | $2.931\%$ | $0.017707$ | $1.771\%$ | $0.8846$ | Strong linear trajectory baseline |
| `Ridge` | $0.029293$ | $2.929\%$ | $0.017702$ | $1.770\%$ | $0.8848$ | Strong linear trajectory baseline |
| **`RandomForestRegressor`** | **$0.013332$** | **$1.333\%$** | **$0.008822$** | **$0.882\%$** | **$0.9761$** | 🏆 **SELECTED BEST (B96)** |
| `GradientBoostingRegressor` | $0.013373$ | $1.337\%$ | $0.008950$ | $0.895\%$ | $0.9760$ | Near-identical top performance |

---

## 10. Locked Test Set Evaluation & B24 vs B96 Comparison

The selected best models were evaluated **exactly once** on the locked Test partition ($N=1,500$ unseen components):

```text
================================================================================
MODULE B LOCKED TEST SET EVALUATION (N=1,500 components)
================================================================================
Screening Gate    Selected Model              Test RMSE      Test MAE       Test R² Score
B24 (24h Gate)    GradientBoostingRegressor   4.033% (0.040) 2.721% (0.027) 0.7890
B96 (96h Gate)    RandomForestRegressor       1.415% (0.014) 0.877% (0.009) 0.9740
================================================================================
```

### Comprehensive Operational Gate Comparison:

| Evaluation Metric | B24 ($24\text{h}$ Gate) | B96 ($96\text{h}$ Gate) | Improvement ($\Delta_{\text{B96} - \text{B24}}$) | Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **$4.033\%$** | **$1.415\%$** | **$-2.618\%$ (64.9% relative reduction)** | Major reduction in large forecast outlier errors |
| **Mean Absolute Error (MAE)** | **$2.721\%$** | **$0.877\%$** | **$-1.843\%$ (67.8% relative reduction)** | Average prediction error drops below $1\%$ absolute drift |
| **Coefficient of Determination ($R^2$)** | **$0.7890$** | **$0.9740$** | **$+0.1851$ gain** | Explains $97.4\%$ of total end-of-test degradation variance |
| **Mean Actual Degradation** | $5.50\%$ | $5.50\%$ | Ground truth baseline | Consistent population baseline |
| **Mean Predicted Degradation** | $5.65\%$ | $5.41\%$ | High calibration fidelity | Unbiased estimation (no systematic over/under-drift) |

---

## 11. Best Model Selection & Rationale

1. **B24 Gate Selection — `GradientBoostingRegressor`:**
   - At $24\text{h}$, initial sensor noise and thermal settling create subtle non-linear interactions. Boosted shallow regression trees effectively filter measurement jitter, outperforming linear models ($R^2 = 0.7679$ vs $0.3804$) and standard Random Forest ($R^2 = 0.7418$).
2. **B96 Gate Selection — `RandomForestRegressor`:**
   - By $96\text{h}$, cumulative physical degradation dominates measurement noise. Random Forest achieves a remarkable $R^2$ of **$0.9740$** on the locked test set with a mean absolute error of just **$0.877\%$**.

---

## 12. Temporal Leakage & Integrity Verification

```text
================================================================================
TEMPORAL & TARGET LEAKAGE AUDIT MATRIX (MODULE B)
================================================================================
Requirement                         Verification Check                          Status
--------------------------------------------------------------------------------
1. No 168h features in B24/B96 X    Forbidden list [iddq_168h, etc.] checked   ✅ PASSED
2. No 96h features in B24 X         Explicit check for _96h in B24 features    ✅ PASSED
3. No module_a_label in X           Target isolation check                     ✅ PASSED
4. No component_type/ID in X        Metadata stripping check                   ✅ PASSED
5. Component-Level Split            Zero train/val/test ID overlap asserted    ✅ PASSED
6. Fitted preprocessor on train     Imputer/scaler fit only on X_train         ✅ PASSED
================================================================================
```

---

## 13. Practical SIH Engineering Interpretation

### Key Experiment Questions Answered:

1. **How accurately can 168h degradation be predicted using only 24h information?**
   - **Highly meaningfully ($R^2 = 0.7890$, $\text{MAE} = 2.72\%$).** Even with only 24 hours of stress data, the model explains nearly $79\%$ of variance in final degradation, enabling early identification of rapid drifters.
2. **How much does prediction improve when 96h information becomes available?**
   - **Dramatically.** Forecast error drops by **$67.8\%$** (MAE drops from $2.72\% \to 0.88\%$), and $R^2$ rises to **$0.9740$**.
3. **Which regression model performs best?**
   - `GradientBoostingRegressor` at $24\text{h}$ (best noise resilience); `RandomForestRegressor` at $96\text{h}$ (best multi-sensor non-linear scaling).
4. **Does B96 significantly outperform B24?**
   - **YES.** The additional 72 hours of burn-in allows physical aging trends to clearly differentiate from thermal noise, slashing RMSE from $4.03\%$ to $1.41\%$.
5. **Is early prediction at 24h practically useful, even if B96 is more accurate?**
   - **YES, EMPHATICALLY.** In semiconductor manufacturing, early triage at $24\text{h}$ allows ejecting severely degrading units early, freeing up expensive burn-in oven capacity and saving electrical stress costs. Marginal units can continue testing to $96\text{h}$ for precision qualification.
6. **Does the model respect temporal causality with ZERO future leakage?**
   - **YES.** Verified by explicit column masking and test assertions.

---

## 14. Limitations & Boundary Conditions

1. **Sensor Dropouts:** Extreme multi-channel sensor dropouts at $0\text{h}$ can increase baseline drift uncertainty; median imputation provides robustness but individual sensor recovery remains vital.
2. **Extreme Outlier Caps:** Forecasts for catastrophic early failures (>40% drift) exhibit minor regression toward the mean, which is safely mitigated when paired with Module A's discrete classification.
3. **Multi-Stress Environmental Interaction:** The model assumes standard JEDEC burn-in temperature ($125^\circ\text{C}$) and voltage ($1.2\text{V}$); varying stress chamber conditions would require ambient normalization.

---

## 15. Final Conclusion

* **Module B is fully designed, implemented, trained, evaluated, and serialized.**
* **B24 achieves $R^2 = 0.7890$, $\text{MAE} = 2.721\%$, $\text{RMSE} = 4.033\%$** using 11 early features.
* **B96 achieves $R^2 = 0.9740$, $\text{MAE} = 0.877\%$, $\text{RMSE} = 1.415\%$** using 19 multi-point features.
* Combined with **Module A** ($97.67\%$ classification accuracy, $94.00\%$ defect capture), the SIH 2026 burn-in screening system delivers a complete, leak-free, two-stage AI architecture for semiconductor quality assurance.

---

## Generated Artifacts Summary

* **Training Code:** [`src/models/train_module_b.py`](file:///d:/SIH/src/models/train_module_b.py)
* **Model Artifacts:**
  - [`models/module_b_24h_gradientboostingregressor.joblib`](file:///d:/SIH/models/module_b_24h_gradientboostingregressor.joblib)
  - [`models/module_b_96h_randomforestregressor.joblib`](file:///d:/SIH/models/module_b_96h_randomforestregressor.joblib)
* **Result Tables:**
  - [`reports/module_b_validation_results.csv`](file:///d:/SIH/reports/module_b_validation_results.csv)
  - [`reports/module_b_test_results.csv`](file:///d:/SIH/reports/module_b_test_results.csv)
  - [`reports/module_b_gate_comparison.csv`](file:///d:/SIH/reports/module_b_gate_comparison.csv)
* **Diagnostic Visualizations:**
  - [`eda/outputs/ml/module_b/b24_actual_vs_predicted_test.png`](file:///d:/SIH/eda/outputs/ml/module_b/b24_actual_vs_predicted_test.png)
  - [`eda/outputs/ml/module_b/b96_actual_vs_predicted_test.png`](file:///d:/SIH/eda/outputs/ml/module_b/b96_actual_vs_predicted_test.png)
  - [`eda/outputs/ml/module_b/b24_residuals_test.png`](file:///d:/SIH/eda/outputs/ml/module_b/b24_residuals_test.png)
  - [`eda/outputs/ml/module_b/b96_residuals_test.png`](file:///d:/SIH/eda/outputs/ml/module_b/b96_residuals_test.png)
  - [`eda/outputs/ml/module_b/module_b_regression_metrics_comparison.png`](file:///d:/SIH/eda/outputs/ml/module_b/module_b_regression_metrics_comparison.png)
  - [`eda/outputs/ml/module_b/b24_feature_importances.png`](file:///d:/SIH/eda/outputs/ml/module_b/b24_feature_importances.png)
  - [`eda/outputs/ml/module_b/b96_feature_importances.png`](file:///d:/SIH/eda/outputs/ml/module_b/b96_feature_importances.png)
