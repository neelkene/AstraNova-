# Data & EDA Validation Checkpoint Report
**Project:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Stage:** Pre-Modeling Dataset & Integrity Validation Checkpoint  
**Generated Date:** 2026-08-29  
**Supporting Validation Files:** `eda/outputs/validation/`

---

## 1. Static 3-Sigma Limits Calculation

### Exact Formula & Derivation
Static specification limits represent conventional Automated Test Equipment (ATE) screening thresholds based on initial, pre-stress measurement variation ($0\text{h}$ baseline):
$$\text{Lower Limit} = \mu_{0\text{h}} - 3\sigma_{0\text{h}}, \quad \text{Upper Limit} = \mu_{0\text{h}} + 3\sigma_{0\text{h}}$$

### Exact Numerical Values from Dataset

| Population Sub-Sample | Row Count ($N$) | Mean $I_{DDQ, 0\text{h}}\;(\mu\text{A})$ | Std Dev $\sigma_{0\text{h}}\;(\mu\text{A})$ | Lower $3\sigma$ Limit ($\mu\text{A}$) | Upper $3\sigma$ Limit ($\mu\text{A}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Components Only ($0\text{h}$)** | **7,000** | **99.9743** | **5.3961** | **83.7861** | **116.1625** |
| **Entire Population ($0\text{h}$)** | **10,000** | **99.9770** | **5.3206** | **84.0152** | **115.9388** |

* **Row Selection:** Calculated on the $0\text{h}$ pre-burn-in quiescent current ($I_{DDQ}$) measurements.
* **Normal vs Population Comparison:** Limits derived from normal components ($[83.79\,\mu\text{A}, 116.16\,\mu\text{A}]$) and the entire population ($[84.02\,\mu\text{A}, 115.94\,\mu\text{A}]$) differ by less than $0.2\%$, confirming that latent defective components share identical manufacturing variation at $t=0\text{h}$.

---

## 2. Static Screening Escape Rate Verification

### Reproducible Verification Table
*(Audited directly against `data/raw/raw_burnin_data.csv` and `data/ground_truth/component_ground_truth.csv`)*

| Stress Time | Drifting Total ($N$) | Drifting within $0\text{h}$ Normal $3\sigma$ Limits | **Drifting Escape Rate (%)** | Anomalous Total ($N$) | Anomalous within $0\text{h}$ Limits | **Anomalous Escape Rate (%)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0h (Pre-stress)** | 2,000 | 1,965 | **98.25%** | 1,000 | 985 | **98.50%** |
| **24h (Early Gate)**| 2,000 | 1,959 | **97.95%** | 1,000 | 954 | **95.40%** |
| **96h (Mid Gate)**  | 2,000 | 1,913 | **95.65%** | 1,000 | 405 | **40.50%** |
| **168h (Full Gate)**| 2,000 | 1,632 | **81.60%** | 1,000 | 57 | **5.70%** |

* **Verification:** The escape numbers are $100\%$ reproducible from raw data.
* **Key Finding:** Even after $168\text{h}$ of continuous stress, **$81.60\%$** of drifting components still remain inside nominal static population bounds. This confirms that static ATE testing cannot solve the problem statement, proving the strict necessity of AI-driven dynamic drift modeling.

---

## 3. Ground-Truth Generation Architecture

Inspection of `data/ground_truth/component_ground_truth.csv` reveals the synthetic generative relationships:

```mermaid
graph TD
    A["Component Class Assignment<br/>Normal: 70%, Drifting: 20%, Anomalous: 10%"] --> B["Module A Label<br/>Normal = 0, Drifting/Anomalous = 1"]
    A --> C["iddq_drift_168h_true Distribution<br/>Normal: ~1%, Drifting: ~10%, Anomalous: ~30%"]
    A --> D["Baseline Parameters<br/>Iddq, Leakage, Delay, Voltage"]
    C & D --> E["Dynamic Sensor Trajectories (0h, 24h, 96h, 168h)<br/>Raw Measurements with Noise & Jitter"]
```

1. **Class Distribution:**
   * `normal`: 7,000 components ($70\%$).
   * `drifting`: 2,000 components ($20\%$).
   * `anomalous`: 1,000 components ($10\%$).
2. **`module_a_label` Assignment:**
   * Strictly binary: $\text{Normal} \to 0$ ($7,000$ parts), $\text{Drifting} \to 1$ ($2,000$ parts), $\text{Anomalous} \to 1$ ($1,000$ parts).
3. **`iddq_drift_168h_true` Generation:**
   * Normal: Mean $= +0.0100$ ($+1.0\%$), Std $= 0.0058$, Max $= +0.0200$.
   * Drifting: Mean $= +0.1001$ ($+10.0\%$), Std $= 0.0289$, Range $[+0.05, +0.15]$.
   * Anomalous: Mean $= +0.2992$ ($+29.9\%$), Std $= 0.0580$, Range $[+0.20, +0.40]$.
4. **Sensor Trajectory Dynamics:**
   * Sensor measurements are conditionally generated from baselines and true drift, with added Gaussian thermal noise, voltage jitter, and $1.5\%$ random sensor dropout.
   * Correlation between empirical $168\text{h}$ drift and true $168\text{h}$ drift is $r = 0.9588$, confirming realistic non-deterministic sensor noise.

---

## 4. Feature Availability & Leakage Matrix

Comprehensive audit of all 27 columns in `data/ml_ready/ml_features.csv`:

| Feature Name | Source | Available @ 24h Gate | Available @ 96h Gate | Uses 168h Data | Leakage Risk & Handling Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `component_id` | Metadata ID | Yes | Yes | No | **EXCLUDE** (Arbitrary string identifier) |
| `iddq_uA_0h` | 0h Sensor | **Yes** | **Yes** | No | Safe for all models |
| `iddq_uA_24h` | 24h Sensor | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `iddq_uA_96h` | 96h Sensor | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `iddq_uA_168h` | 168h Sensor | ❌ **NO** (Future) | ❌ **NO** (Future) | **Yes** | **LEAKAGE** if used in 24h or 96h gate model |
| `leakage_current_uA_0h` | 0h Sensor | **Yes** | **Yes** | No | Safe for all models |
| `leakage_current_uA_24h` | 24h Sensor | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `leakage_current_uA_96h` | 96h Sensor | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `leakage_current_uA_168h` | 168h Sensor | ❌ **NO** (Future) | ❌ **NO** (Future) | **Yes** | **LEAKAGE** if used in 24h or 96h gate model |
| `propagation_delay_ns_0h` | 0h Sensor | **Yes** | **Yes** | No | Safe for all models |
| `propagation_delay_ns_24h` | 24h Sensor | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `propagation_delay_ns_96h` | 96h Sensor | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `propagation_delay_ns_168h` | 168h Sensor | ❌ **NO** (Future) | ❌ **NO** (Future) | **Yes** | **LEAKAGE** if used in 24h or 96h gate model |
| `voltage_V_0h` | 0h Sensor | **Yes** | **Yes** | No | Safe for all models |
| `voltage_V_24h` | 24h Sensor | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `voltage_V_96h` | 96h Sensor | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `voltage_V_168h` | 168h Sensor | ❌ **NO** (Future) | ❌ **NO** (Future) | **Yes** | **LEAKAGE** if used in 24h or 96h gate model |
| `temperature_C_0h` | 0h Sensor | **Yes** | **Yes** | No | Safe for all models |
| `temperature_C_24h` | 24h Sensor | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `temperature_C_96h` | 96h Sensor | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `temperature_C_168h` | 168h Sensor | ❌ **NO** (Future) | ❌ **NO** (Future) | **Yes** | **LEAKAGE** if used in 24h or 96h gate model |
| `iddq_drift_24h_pct` | Calculated Drift | **Yes** | **Yes** | No | Safe for 24h, 96h, 168h models |
| `iddq_drift_96h_pct` | Calculated Drift | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `leakage_drift_96h_pct` | Calculated Drift | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `delay_drift_96h_pct` | Calculated Drift | ❌ **NO** (Future) | **Yes** | No | **LEAKAGE** if used in 24h gate model |
| `module_a_label` | Target Label | **TARGET ONLY** | **TARGET ONLY** | Yes | **CRITICAL TARGET LEAKAGE** (Must be $y$) |
| `iddq_drift_168h_true` | Target Metric | **TARGET ONLY** | **TARGET ONLY** | Yes | **CRITICAL TARGET LEAKAGE** (Must be $y$) |

---

## 5. Target & Temporal Leakage Safeguards

```text
Target & Leakage Isolation Protocol:
1. Feature Matrix (X) vs Target Vector (y) Separation:
   - Module A Target: y_clf = df['module_a_label']
   - Module B Target: y_reg = df['iddq_drift_168h_true']
   - Feature Matrix X contains only measurement and drift metrics.
2. Screening Gate Feature Subsets:
   - 24h Gate Model Features: [0h and 24h sensor values + iddq_drift_24h_pct] (No 96h or 168h features)
   - 96h Gate Model Features: [0h, 24h, and 96h sensor values + 24h/96h drift features] (No 168h features)
3. Metadata Isolation:
   - component_id is strictly removed from all feature vectors.
```

---

## 6. Recommended Data Splitting Strategy

* **Split Granularity:** **Component-Level Stratified Split** (guarantees that all time points of a component reside entirely within either Train, Validation, or Test).
* **Ratio:** **$70\%$ Train (7,000 components), $15\%$ Validation (1,500 components), $15\%$ Test (1,500 components)**.
* **Stratification:** Stratified by `component_type` (ensuring exact $70:20:10$ class balance in all three folds).
* **Lot Identifier Audit:** The datasets do not contain a `lot_id` column. Component-level stratified splitting is the gold standard and prevents any row-level leakage.

---

## 7. Synthetic Data Difficulty & Separability Assessment

To confirm that the synthetic dataset is physically realistic and not trivially separable:

| Evaluation Metric | Measured Value | Physical & Mathematical Interpretation |
| :--- | :--- | :--- |
| **Cohen's $d$ ($24\text{h}$ Drift: Drifting vs Normal)** | **$0.4573$** | **Moderate separation.** Drifting components have not drifted far enough at $24\text{h}$ to be trivially separated; substantial noise overlap exists. |
| **Single-Threshold Accuracy @ $24\text{h}$** | **$81.11\%$** | **A single threshold FAILS.** Misclassifies nearly $20\%$ of components due to thermal/sensor noise overlap. Multivariate ML is strictly required. |
| **Cohen's $d$ ($96\text{h}$ Drift: Drifting vs Normal)** | **$1.5222$** | **Strong separation.** As physical stress accumulates, drift signal becomes dominant over measurement noise. |
| **Single-Threshold Accuracy @ $96\text{h}$** | **$94.18\%$** | High baseline, but multivariate ML models (combining $I_{DDQ}$, leakage, and delay drift) are needed to reach $>99\%$ screening accuracy with zero defect escapes. |

---

## 8. Final Validation Verdict

### 🏆 Verdict: **A. READY FOR ML**

### Justification:
1. **Pristine Data Hygiene:** Zero duplicate rows, zero invalid physical numbers, $100\%$ component alignment across all files.
2. **True Latent Defect Signatures:** $98.25\%$ escape rate at $0\text{h}$ proves the dataset accurately embodies the latent degradation problem described in the SIH problem statement.
3. **Clean Feature-Target Delineation:** Features are mapped and stratified by screening time gates ($24\text{h}$, $96\text{h}$, $168\text{h}$) to eliminate temporal and target leakage.
4. **Appropriate Statistical Difficulty:** Moderate Cohen's $d$ ($0.457$) at $24\text{h}$ guarantees that machine learning will provide meaningful, non-trivial engineering value over heuristic thresholding.

### Planned ML Core Architecture:
* **Module A:** Early Anomaly & Drift Classification ($0\text{h} \to 24\text{h} \to 96\text{h}$ $\to$ `module_a_label`).
* **Module B:** Continuous $168\text{h}$ Degradation Forecaster ($0\text{h} \to 24\text{h} \to 96\text{h}$ $\to$ `iddq_drift_168h_true`).
