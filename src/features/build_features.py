"""
Module: src.features.build_features
Purpose: Construct gate-specific feature subsets (24h vs 96h) and build leak-free preprocessing pipelines.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data.load_data import TARGET_MODULE_A, TARGET_MODULE_B, IDENTIFIER_COL

# ------------------------------------------------------------------------------
# GATE-SPECIFIC FEATURE DEFINITIONS
# ------------------------------------------------------------------------------

# 1. Available at 24h screening gate (0h + 24h only)
FEATURES_24H_GATE: List[str] = [
    # 0h Pre-burn-in Baselines
    'iddq_uA_0h',
    'leakage_current_uA_0h',
    'propagation_delay_ns_0h',
    'voltage_V_0h',
    'temperature_C_0h',
    # 24h Early Burn-In Measurements
    'iddq_uA_24h',
    'leakage_current_uA_24h',
    'propagation_delay_ns_24h',
    'voltage_V_24h',
    'temperature_C_24h',
    # 24h Calculated Drift
    'iddq_drift_24h_pct'
]

# 2. Available at 96h screening gate (0h + 24h + 96h)
FEATURES_96H_GATE: List[str] = FEATURES_24H_GATE + [
    # 96h Mid Burn-In Measurements
    'iddq_uA_96h',
    'leakage_current_uA_96h',
    'propagation_delay_ns_96h',
    'voltage_V_96h',
    'temperature_C_96h',
    # 96h Calculated Multi-Parameter Drift
    'iddq_drift_96h_pct',
    'leakage_drift_96h_pct',
    'delay_drift_96h_pct'
]

# Forbidden 168h end-of-test features (must NEVER be used in early screening X)
FEATURES_168H_FORBIDDEN: List[str] = [
    'iddq_uA_168h',
    'leakage_current_uA_168h',
    'propagation_delay_ns_168h',
    'voltage_V_168h',
    'temperature_C_168h'
]


def get_gate_feature_names(gate: str) -> List[str]:
    """
    Returns the feature list for a given screening gate ('24h' or '96h').
    """
    gate_lower = gate.lower().strip()
    if gate_lower in ['24h', '24', 'a24', 'b24']:
        return list(FEATURES_24H_GATE)
    elif gate_lower in ['96h', '96', 'a96', 'b96']:
        return list(FEATURES_96H_GATE)
    else:
        raise ValueError(f"Unknown screening gate: '{gate}'. Expected '24h' or '96h'.")


def extract_features_and_target(
    df: pd.DataFrame,
    experiment: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts feature matrix X and target vector y for a specific experiment:
    - 'A24': Module A @ 24h Gate (target: module_a_label)
    - 'A96': Module A @ 96h Gate (target: module_a_label)
    - 'B24': Module B @ 24h Gate (target: iddq_drift_168h_true)
    - 'B96': Module B @ 96h Gate (target: iddq_drift_168h_true)
    """
    exp_upper = experiment.upper().strip()
    
    if exp_upper == 'A24':
        feature_cols = get_gate_feature_names('24h')
        target_col = TARGET_MODULE_A
    elif exp_upper == 'A96':
        feature_cols = get_gate_feature_names('96h')
        target_col = TARGET_MODULE_A
    elif exp_upper == 'B24':
        feature_cols = get_gate_feature_names('24h')
        target_col = TARGET_MODULE_B
    elif exp_upper == 'B96':
        feature_cols = get_gate_feature_names('96h')
        target_col = TARGET_MODULE_B
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. Expected A24, A96, B24, or B96.")
        
    # Temporal & Target Leakage Assertions
    assert target_col not in feature_cols, f"Target {target_col} leaked into feature columns!"
    assert IDENTIFIER_COL not in feature_cols, "component_id found in feature columns!"
    for f in FEATURES_168H_FORBIDDEN:
        assert f not in feature_cols, f"168h end-of-test feature {f} leaked into {experiment}!"
    if '24' in exp_upper:
        for f in ['iddq_uA_96h', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct']:
            assert f not in feature_cols, f"96h future feature {f} leaked into 24h gate experiment {experiment}!"

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessing_pipeline(
    scale_features: bool = False,
    impute_strategy: str = 'median'
) -> Pipeline:
    """
    Constructs a scikit-learn preprocessing pipeline with leak-free median imputation
    and optional standard scaling.
    """
    steps = [
        ('imputer', SimpleImputer(strategy=impute_strategy))
    ]
    if scale_features:
        steps.append(('scaler', StandardScaler()))
        
    return Pipeline(steps=steps)


if __name__ == "__main__":
    from src.data.load_data import load_ml_ready_data
    df = load_ml_ready_data()
    print("Gate Feature Sets Audit:")
    for exp in ['A24', 'A96', 'B24', 'B96']:
        X, y = extract_features_and_target(df, exp)
        print(f"  [{exp}] X shape: {X.shape}, y shape: {y.shape}, Target: {y.name}")
