"""
Module: src.data.load_data
Purpose: Load and validate the ML-ready dataset with strict schema and integrity checks.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

DEFAULT_ML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'ml_ready', 'ml_features.csv'
)

EXPECTED_COLUMNS = [
    'component_id',
    'iddq_uA_0h', 'iddq_uA_24h', 'iddq_uA_96h', 'iddq_uA_168h',
    'leakage_current_uA_0h', 'leakage_current_uA_24h', 'leakage_current_uA_96h', 'leakage_current_uA_168h',
    'propagation_delay_ns_0h', 'propagation_delay_ns_24h', 'propagation_delay_ns_96h', 'propagation_delay_ns_168h',
    'voltage_V_0h', 'voltage_V_24h', 'voltage_V_96h', 'voltage_V_168h',
    'temperature_C_0h', 'temperature_C_24h', 'temperature_C_96h', 'temperature_C_168h',
    'module_a_label', 'iddq_drift_168h_true',
    'iddq_drift_24h_pct', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct'
]

TARGET_MODULE_A = 'module_a_label'
TARGET_MODULE_B = 'iddq_drift_168h_true'
IDENTIFIER_COL = 'component_id'


def load_ml_ready_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the prepared ML-ready dataset and validates integrity.
    """
    if filepath is None:
        filepath = DEFAULT_ML_PATH
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ML-ready dataset not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    validate_ml_schema(df)
    return df


def validate_ml_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs assertions and validation checks on dataset schema.
    """
    # 1. Column presence check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema mismatch: missing expected columns: {missing_cols}")
        
    # 2. Check targets and ID presence
    assert IDENTIFIER_COL in df.columns, "Identifier column 'component_id' is missing."
    assert TARGET_MODULE_A in df.columns, f"Target '{TARGET_MODULE_A}' is missing."
    assert TARGET_MODULE_B in df.columns, f"Target '{TARGET_MODULE_B}' is missing."
    
    # 3. Check duplicate components
    n_duplicates = df[IDENTIFIER_COL].duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(f"Found {n_duplicates} duplicate component IDs.")
        
    # 4. Check targets have zero missing values
    assert df[TARGET_MODULE_A].isnull().sum() == 0, f"Target '{TARGET_MODULE_A}' contains nulls."
    assert df[TARGET_MODULE_B].isnull().sum() == 0, f"Target '{TARGET_MODULE_B}' contains nulls."
    
    # 5. Check target types and valid values
    assert set(df[TARGET_MODULE_A].unique()).issubset({0, 1}), f"Invalid values in {TARGET_MODULE_A}"
    
    validation_meta = {
        "num_rows": len(df),
        "num_cols": df.shape[1],
        "unique_components": df[IDENTIFIER_COL].nunique(),
        "module_a_class_balance": df[TARGET_MODULE_A].value_counts().to_dict(),
        "target_b_mean": float(df[TARGET_MODULE_B].mean()),
        "target_b_std": float(df[TARGET_MODULE_B].std()),
        "total_missing_entries": int(df.isnull().sum().sum())
    }
    return validation_meta


if __name__ == "__main__":
    df = load_ml_ready_data()
    meta = validate_ml_schema(df)
    print("ML Ready Dataset successfully loaded and validated:")
    for k, v in meta.items():
        print(f"  - {k}: {v}")
