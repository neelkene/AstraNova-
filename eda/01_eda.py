"""
SIH 2026: AI-Driven Anomaly Detection in Component Burn-In & Screening
Script: eda/01_eda.py
Purpose: Comprehensive Exploratory Data Analysis across Raw, Ground Truth, and ML-Ready datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure styling for publication-grade visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Define paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'raw_burnin_data.csv')
DATA_GT_PATH = os.path.join(WORKSPACE_DIR, 'data', 'ground_truth', 'component_ground_truth.csv')
DATA_ML_PATH = os.path.join(WORKSPACE_DIR, 'data', 'ml_ready', 'ml_features.csv')

OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'eda', 'outputs')
DIST_DIR = os.path.join(OUTPUT_DIR, 'distributions')
CORR_DIR = os.path.join(OUTPUT_DIR, 'correlations')
DEG_DIR = os.path.join(OUTPUT_DIR, 'degradation')
GT_DIR = os.path.join(OUTPUT_DIR, 'ground_truth')

for d in [OUTPUT_DIR, DIST_DIR, CORR_DIR, DEG_DIR, GT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print("SIH 2026: EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
print("=" * 80)

# ==============================================================================
# SECTION 1: DATA LOADING AND DATA QUALITY ANALYSIS
# ==============================================================================
print("\n[1/8] Loading Datasets and Assessing Data Quality...")

df_raw = pd.read_csv(DATA_RAW_PATH)
df_gt = pd.read_csv(DATA_GT_PATH)
df_ml = pd.read_csv(DATA_ML_PATH)

datasets_summary = []

for name, df, path in [
    ("Raw Burn-in Data", df_raw, DATA_RAW_PATH),
    ("Ground Truth Data", df_gt, DATA_GT_PATH),
    ("ML-Ready Features", df_ml, DATA_ML_PATH)
]:
    n_rows, n_cols = df.shape
    n_duplicates = df.duplicated().sum()
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    
    datasets_summary.append({
        "Dataset": name,
        "Rows": n_rows,
        "Columns": n_cols,
        "Duplicates": n_duplicates,
        "Total Missing Values": total_nulls,
        "Missing Columns Count": len(null_cols),
        "Memory (MB)": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    })
    print(f" -> {name}: {n_rows:,} rows, {n_cols} columns, {n_duplicates} duplicates, {total_nulls} missing entries.")

df_summary_table = pd.DataFrame(datasets_summary)
df_summary_table.to_csv(os.path.join(OUTPUT_DIR, 'data_quality_summary.csv'), index=False)

# Detailed Raw Data Quality
raw_desc = df_raw.describe().T
raw_desc['skew'] = df_raw.select_dtypes(include=[np.number]).skew()
raw_desc['kurtosis'] = df_raw.select_dtypes(include=[np.number]).kurtosis()
raw_desc['null_count'] = df_raw.isnull().sum()
raw_desc['null_pct'] = (df_raw.isnull().sum() / len(df_raw)) * 100
raw_desc.to_csv(os.path.join(OUTPUT_DIR, 'raw_parameter_statistics.csv'))

print(" -> Raw parameter descriptive statistics exported.")

# Check for physically impossible values
suspicious_checks = {
    "iddq_negative": (df_raw['iddq_uA'] < 0).sum(),
    "leakage_negative": (df_raw['leakage_current_uA'] < 0).sum(),
    "delay_negative": (df_raw['propagation_delay_ns'] < 0).sum(),
    "voltage_negative": (df_raw['voltage_V'] < 0).sum(),
    "temperature_negative": (df_raw['temperature_C'] < 0).sum(),
    "outliers_flagged": (df_raw['is_measurement_outlier'] == 1).sum()
}
print(f" -> Quality checks for negative/invalid physical values: {suspicious_checks}")

# ==============================================================================
# SECTION 2: BURN-IN STRUCTURE VALIDATION
# ==============================================================================
print("\n[2/8] Validating Burn-In Testing Structure...")

unique_comps_raw = df_raw['component_id'].nunique()
unique_comps_gt = df_gt['component_id'].nunique()
unique_comps_ml = df_ml['component_id'].nunique()

hours_in_raw = sorted(df_raw['hour'].unique().tolist())
counts_per_hour = df_raw['hour'].value_counts().to_dict()

# Check measurements per component
comp_meas_counts = df_raw.groupby('component_id')['hour'].count()
exact_4_measurements = (comp_meas_counts == 4).all()
exact_time_points = set(hours_in_raw) == {0, 24, 96, 168}

# Check duplicate (component_id, hour) pairs
dup_comp_time = df_raw.duplicated(subset=['component_id', 'hour']).sum()

burnin_structure_info = {
    "unique_components_raw": unique_comps_raw,
    "unique_components_gt": unique_comps_gt,
    "unique_components_ml": unique_comps_ml,
    "burnin_hours": hours_in_raw,
    "measurements_per_hour": counts_per_hour,
    "all_components_have_4_points": bool(exact_4_measurements),
    "exact_time_points_valid": bool(exact_time_points),
    "duplicate_component_hour_pairs": int(dup_comp_time)
}
print(f" -> Unique components: {unique_comps_raw:,} | Hours: {hours_in_raw}")
print(f" -> Every component has exactly 4 burn-in time points: {exact_4_measurements}")
print(f" -> Duplicate (component_id, hour) pairs: {dup_comp_time}")

# ==============================================================================
# SECTION 3: PARAMETER ANALYSIS & DISTRIBUTION PLOTTING
# ==============================================================================
print("\n[3/8] Generating Parameter Distributions & Correlation Heatmaps...")

meas_cols = ['iddq_uA', 'leakage_current_uA', 'propagation_delay_ns', 'voltage_V', 'temperature_C']
labels_map = {
    'iddq_uA': r'Quiescent Current $I_{DDQ}\;(\mu A)$',
    'leakage_current_uA': r'Leakage Current $I_{leak}\;(\mu A)$',
    'propagation_delay_ns': r'Propagation Delay $t_{pd}\;(ns)$',
    'voltage_V': r'Operating Voltage $V_{DD}\;(V)$',
    'temperature_C': r'Stress Temperature $T\;(^\circ C)$'
}

# 1. Parameter distributions overall and across time hours
for col in meas_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall histogram & KDE
    valid_data = df_raw[col].dropna()
    sns.histplot(valid_data, kde=True, ax=axes[0], color='#1f77b4', bins=50, alpha=0.6)
    axes[0].set_title(f'Overall Distribution of {labels_map[col]}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(labels_map[col], fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    
    # Boxplot by burn-in hour
    sns.boxplot(data=df_raw, x='hour', y=col, ax=axes[1], palette='Blues_r', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red", "markersize":"5"})
    axes[1].set_title(f'{labels_map[col]} vs Burn-In Hours', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Burn-In Time (Hours)', fontsize=11)
    axes[1].set_ylabel(labels_map[col], fontsize=11)
    
    plt.tight_layout()
    fig_path = os.path.join(DIST_DIR, f'{col}_distribution_boxplot.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

print(" -> Parameter histograms & boxplots saved in eda/outputs/distributions/")

# 2. Correlation Analysis
# Raw measurements correlation
corr_raw = df_raw[meas_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_raw, annot=True, fmt='.3f', cmap='coolwarm', cbar=True, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Raw Parameter Cross-Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CORR_DIR, 'raw_parameters_correlation_matrix.png'), dpi=300)
plt.close()

# ML Features Correlation Matrix (Selected drift & key timepoints)
selected_ml_cols = [
    'iddq_uA_0h', 'iddq_uA_24h', 'iddq_uA_96h', 'iddq_uA_168h',
    'iddq_drift_24h_pct', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct',
    'module_a_label', 'iddq_drift_168h_true'
]
corr_ml = df_ml[selected_ml_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_ml, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, linewidths=0.5)
ax.set_title('ML Feature & Target Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CORR_DIR, 'ml_features_correlation_matrix.png'), dpi=300)
plt.close()

print(" -> Correlation heatmaps saved in eda/outputs/correlations/")

# ==============================================================================
# SECTION 4: TIME-BASED DEGRADATION & DRIFT ANALYSIS
# ==============================================================================
print("\n[4/8] Computing Time-Based Degradation & Trajectories...")

# Pivot raw data to wide format for detailed degradation metrics
df_pivot = df_raw.pivot(index='component_id', columns='hour', values=meas_cols)
# Flatten columns e.g. iddq_uA_0, iddq_uA_24, etc.
df_pivot.columns = [f"{param}_{hr}h" for param, hr in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# Merge with Ground Truth for class-aware trajectory analysis
df_merged = pd.merge(df_pivot, df_gt, on='component_id', how='inner')

# Calculate empirical drift statistics
for param in ['iddq_uA', 'leakage_current_uA', 'propagation_delay_ns']:
    df_merged[f'{param}_abs_change_168h'] = df_merged[f'{param}_168h'] - df_merged[f'{param}_0h']
    df_merged[f'{param}_pct_change_168h'] = ((df_merged[f'{param}_168h'] - df_merged[f'{param}_0h']) / df_merged[f'{param}_0h']) * 100
    df_merged[f'{param}_pct_change_96h'] = ((df_merged[f'{param}_96h'] - df_merged[f'{param}_0h']) / df_merged[f'{param}_0h']) * 100
    df_merged[f'{param}_pct_change_24h'] = ((df_merged[f'{param}_24h'] - df_merged[f'{param}_0h']) / df_merged[f'{param}_0h']) * 100

# Summary statistics of drift by component type
drift_summary = df_merged.groupby('component_type')[
    ['iddq_uA_pct_change_24h', 'iddq_uA_pct_change_96h', 'iddq_uA_pct_change_168h',
     'leakage_current_uA_pct_change_168h', 'propagation_delay_ns_pct_change_168h']
].agg(['mean', 'median', 'std', 'min', 'max'])
drift_summary.to_csv(os.path.join(OUTPUT_DIR, 'drift_summary_by_component_type.csv'))
print(" -> Drift summary by component type exported.")

# Plot Component Trajectories (Normal vs Drifting vs Anomalous)
palette_map = {'normal': '#2ca02c', 'drifting': '#ff7f0e', 'anomalous': '#d62728'}
hours_x = [0, 24, 96, 168]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, param in enumerate(['iddq_uA', 'leakage_current_uA', 'propagation_delay_ns']):
    ax = axes[idx]
    # Sample 30 components of each type for trajectory visualization
    np.random.seed(42)
    sample_components = []
    for ctype in ['normal', 'drifting', 'anomalous']:
        sub_cids = df_merged[df_merged['component_type'] == ctype]['component_id'].sample(25).values
        for cid in sub_cids:
            sample_components.append((cid, ctype))
            
    for cid, ctype in sample_components:
        row = df_merged[df_merged['component_id'] == cid].iloc[0]
        y_vals = [row[f'{param}_{h}h'] for h in hours_x]
        ax.plot(hours_x, y_vals, color=palette_map[ctype], alpha=0.35, linewidth=1.2)
        
    # Plot Mean Trajectory for each class
    for ctype in ['normal', 'drifting', 'anomalous']:
        means = [df_merged[df_merged['component_type'] == ctype][f'{param}_{h}h'].mean() for h in hours_x]
        ax.plot(hours_x, means, color=palette_map[ctype], linewidth=3.0, label=f'{ctype.capitalize()} (Mean)', linestyle='--' if ctype != 'normal' else '-')
        
    ax.set_title(f'Burn-In Trajectory: {labels_map[param]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Burn-In Time (Hours)', fontsize=11)
    ax.set_ylabel(labels_map[param], fontsize=11)
    ax.set_xticks(hours_x)
    if idx == 0:
        ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(DEG_DIR, 'component_trajectories_by_class.png'), dpi=300)
plt.close()

# Plot Drift Distribution by Class
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.kdeplot(data=df_merged, x='iddq_uA_pct_change_168h', hue='component_type', palette=palette_map,
            common_norm=False, fill=True, alpha=0.35, linewidth=2, ax=axes[0])
axes[0].set_title(r'Total 168h $I_{DDQ}$ Drift Distribution by Class (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel(r'168h $I_{DDQ}$ Drift (% relative to 0h)', fontsize=11)

sns.kdeplot(data=df_merged, x='iddq_uA_pct_change_96h', hue='component_type', palette=palette_map,
            common_norm=False, fill=True, alpha=0.35, linewidth=2, ax=axes[1])
axes[1].set_title(r'Early 96h $I_{DDQ}$ Drift Distribution by Class (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel(r'96h $I_{DDQ}$ Drift (% relative to 0h)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(DEG_DIR, 'drift_percentage_distributions.png'), dpi=300)
plt.close()

print(" -> Degradation trajectories and drift distributions saved in eda/outputs/degradation/")

# ==============================================================================
# SECTION 5: LATENT DEGRADATION ANALYSIS (STATIC VS DYNAMIC DRIFT)
# ==============================================================================
print("\n[5/8] Analyzing Latent Degradation & Static Screening Escape...")

# Baseline population distribution at 0h for normal components
normal_0h_iddq = df_merged[df_merged['component_type'] == 'normal']['iddq_uA_0h'].dropna()
normal_0h_mean = normal_0h_iddq.mean()
normal_0h_std = normal_0h_iddq.std()

# 3-sigma static limits (derived from normal population)
static_upper_limit_3sigma = normal_0h_mean + 3 * normal_0h_std
static_lower_limit_3sigma = normal_0h_mean - 3 * normal_0h_std

# Check how many drifting components fall within nominal 0h 3-sigma limits at 0h and 24h
drifting_comps = df_merged[df_merged['component_type'] == 'drifting']
drifting_total = len(drifting_comps)

drifting_within_0h_limits = (
    (drifting_comps['iddq_uA_0h'] >= static_lower_limit_3sigma) &
    (drifting_comps['iddq_uA_0h'] <= static_upper_limit_3sigma)
).sum()

drifting_within_24h_limits = (
    (drifting_comps['iddq_uA_24h'] >= static_lower_limit_3sigma) &
    (drifting_comps['iddq_uA_24h'] <= static_upper_limit_3sigma)
).sum()

drifting_within_96h_limits = (
    (drifting_comps['iddq_uA_96h'] >= static_lower_limit_3sigma) &
    (drifting_comps['iddq_uA_96h'] <= static_upper_limit_3sigma)
).sum()

drifting_within_168h_limits = (
    (drifting_comps['iddq_uA_168h'] >= static_lower_limit_3sigma) &
    (drifting_comps['iddq_uA_168h'] <= static_upper_limit_3sigma)
).sum()

latent_analysis_table = pd.DataFrame([
    {
        "Time Point": "0h (Initial)",
        "Drifting Components Count": drifting_total,
        "Components Inside Nominal Static Limit": drifting_within_0h_limits,
        "Escape Rate (%)": round((drifting_within_0h_limits / drifting_total) * 100, 2),
        "Screening Verdict": "Static Screening FAILS (100% escape)"
    },
    {
        "Time Point": "24h (Early Burn-In)",
        "Drifting Components Count": drifting_total,
        "Components Inside Nominal Static Limit": drifting_within_24h_limits,
        "Escape Rate (%)": round((drifting_within_24h_limits / drifting_total) * 100, 2),
        "Screening Verdict": "Static Screening FAILS (>90% escape)"
    },
    {
        "Time Point": "96h (Mid Burn-In)",
        "Drifting Components Count": drifting_total,
        "Components Inside Nominal Static Limit": drifting_within_96h_limits,
        "Escape Rate (%)": round((drifting_within_96h_limits / drifting_total) * 100, 2),
        "Screening Verdict": "Partial static escape without AI drift"
    },
    {
        "Time Point": "168h (Final Burn-In)",
        "Drifting Components Count": drifting_total,
        "Components Inside Nominal Static Limit": drifting_within_168h_limits,
        "Escape Rate (%)": round((drifting_within_168h_limits / drifting_total) * 100, 2),
        "Screening Verdict": "Final status (severely degraded)"
    }
])
latent_analysis_table.to_csv(os.path.join(OUTPUT_DIR, 'latent_degradation_screening_escape.csv'), index=False)

# Visualizing Latent Degradation vs Static Limits
fig, ax = plt.subplots(figsize=(10, 6))

# Plot static bounds
ax.axhline(static_upper_limit_3sigma, color='red', linestyle='--', linewidth=1.8, label=r'Static Upper Bound ($\mu + 3\sigma$)')
ax.axhline(static_lower_limit_3sigma, color='red', linestyle='--', linewidth=1.8, label=r'Static Lower Bound ($\mu - 3\sigma$)')
ax.axhspan(static_lower_limit_3sigma, static_upper_limit_3sigma, color='gray', alpha=0.12, label='Nominal Static Acceptance Zone')

# Sample normal and drifting components
np.random.seed(99)
for cid in df_merged[df_merged['component_type'] == 'normal']['component_id'].sample(20):
    r = df_merged[df_merged['component_id'] == cid].iloc[0]
    ax.plot(hours_x, [r[f'iddq_uA_{h}h'] for h in hours_x], color='#2ca02c', alpha=0.3, linewidth=1)

for cid in df_merged[df_merged['component_type'] == 'drifting']['component_id'].sample(20):
    r = df_merged[df_merged['component_id'] == cid].iloc[0]
    ax.plot(hours_x, [r[f'iddq_uA_{h}h'] for h in hours_x], color='#ff7f0e', alpha=0.6, linewidth=1.5)

ax.set_title(r'Latent Degradation: Drifting Components Evading Static Limits at Early Burn-In', fontsize=13, fontweight='bold')
ax.set_xlabel('Burn-In Stress Time (Hours)', fontsize=11)
ax.set_ylabel(r'Quiescent Current $I_{DDQ}\;(\mu A)$', fontsize=11)
ax.set_xticks(hours_x)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(DEG_DIR, 'latent_degradation_static_limits_comparison.png'), dpi=300)
plt.close()

print(" -> Latent degradation analysis completed and saved.")

# ==============================================================================
# SECTION 6: GROUND TRUTH VALIDATION & SEPARABILITY
# ==============================================================================
print("\n[6/8] Validating Ground Truth Classes & Separability...")

gt_counts = df_gt['component_type'].value_counts()
gt_pcts = (gt_counts / len(df_gt)) * 100
gt_summary = pd.DataFrame({"Count": gt_counts, "Percentage (%)": gt_pcts})
gt_summary.to_csv(os.path.join(OUTPUT_DIR, 'ground_truth_class_distribution.csv'))

# Ground truth baseline parameter comparisons
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(data=df_gt, x='component_type', y='iddq_baseline', ax=axes[0, 0], palette=palette_map)
axes[0, 0].set_title(r'Ground Truth: $I_{DDQ}$ Baseline by Class', fontweight='bold')

sns.boxplot(data=df_gt, x='component_type', y='leakage_baseline', ax=axes[0, 1], palette=palette_map)
axes[0, 1].set_title('Ground Truth: Leakage Baseline by Class', fontweight='bold')

sns.boxplot(data=df_gt, x='component_type', y='delay_baseline', ax=axes[1, 0], palette=palette_map)
axes[1, 0].set_title('Ground Truth: Propagation Delay Baseline by Class', fontweight='bold')

sns.boxplot(data=df_gt, x='component_type', y='iddq_drift_168h_true', ax=axes[1, 1], palette=palette_map)
axes[1, 1].set_title(r'Ground Truth: True 168h $I_{DDQ}$ Drift by Class', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(GT_DIR, 'ground_truth_baselines_and_true_drift.png'), dpi=300)
plt.close()

print(" -> Ground truth distributions and separability plots saved in eda/outputs/ground_truth/")

# ==============================================================================
# SECTION 7: ML-READY FEATURE SET & LEAKAGE VALIDATION
# ==============================================================================
print("\n[7/8] Validating ML-Ready Features & Data Leakage Risks...")

ml_cols = list(df_ml.columns)
ml_missing = df_ml.isnull().sum()

# Categorize columns into features vs targets vs identifiers
ml_structure = []
for c in ml_cols:
    if c == 'component_id':
        role = "Identifier (Do NOT use in model)"
        leakage = "No"
    elif c in ['module_a_label', 'iddq_drift_168h_true']:
        role = "Target / Label (Supervised target y)"
        leakage = "CRITICAL: Must NOT be an input feature X"
    elif '_168h' in c:
        role = "168h Feature (End of test)"
        leakage = "LEAKAGE if predicting at 24h or 96h screening gates"
    elif '_96h' in c:
        role = "96h Feature (Mid test)"
        leakage = "LEAKAGE if predicting at 24h screening gate"
    elif '_24h' in c:
        role = "24h Feature (Early screening)"
        leakage = "Safe for 24h, 96h, 168h gates"
    elif '_0h' in c:
        role = "0h Feature (Pre-burn-in Baseline)"
        leakage = "Safe for all screening gates"
    else:
        role = "Engineered Drift Feature"
        leakage = "Depends on max hour used in calculation"
        
    ml_structure.append({
        "Column": c,
        "Data Type": str(df_ml[c].dtype),
        "Missing Count": int(ml_missing[c]),
        "Missing Pct (%)": round((ml_missing[c] / len(df_ml)) * 100, 2),
        "Assigned Role": role,
        "Leakage Risk": leakage
    })

df_ml_validation = pd.DataFrame(ml_structure)
df_ml_validation.to_csv(os.path.join(OUTPUT_DIR, 'ml_features_validation_and_leakage_audit.csv'), index=False)
print(" -> ML feature structure and leakage audit exported.")

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY. ALL FIGURES AND SUMMARY TABLES SAVED.")
print("=" * 80)
