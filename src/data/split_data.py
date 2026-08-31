"""
Module: src.data.split_data
Purpose: Component-level stratified data splitting for Train (70%), Validation (15%), and Test (15%).
"""

import os
from typing import Tuple, Dict, Any, Set
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import load_ml_ready_data, TARGET_MODULE_A, IDENTIFIER_COL

DEFAULT_RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_components(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a component-level stratified split (70% Train, 15% Val, 15% Test).
    Ensures 100% mutual exclusivity of component IDs across all partitions.
    """
    # 1. First split: 70% Train, 30% Temporary (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=random_state,
        stratify=df[TARGET_MODULE_A]
    )
    
    # 2. Second split: Split remaining 30% equally into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.50,
        random_state=random_state,
        stratify=temp_df[TARGET_MODULE_A]
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # 3. Assert zero leakage across component partitions
    train_ids: Set[str] = set(train_df[IDENTIFIER_COL])
    val_ids: Set[str] = set(val_df[IDENTIFIER_COL])
    test_ids: Set[str] = set(test_df[IDENTIFIER_COL])
    
    assert len(train_ids.intersection(val_ids)) == 0, "FATAL: Component ID leakage between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "FATAL: Component ID leakage between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "FATAL: Component ID leakage between Val and Test!"
    
    total_comps = len(train_ids) + len(val_ids) + len(test_ids)
    assert total_comps == len(df), f"Expected {len(df)} total components, but got {total_comps}"
    
    return train_df, val_df, test_df


def get_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Returns split metadata and class distribution checks.
    """
    def _split_meta(split_name: str, sub_df: pd.DataFrame) -> Dict[str, Any]:
        counts = sub_df[TARGET_MODULE_A].value_counts().to_dict()
        pct_defective = (counts.get(1, 0) / len(sub_df)) * 100
        return {
            "split": split_name,
            "components": len(sub_df),
            "percentage": f"{(len(sub_df) / (len(train_df) + len(val_df) + len(test_df))) * 100:.1f}%",
            "class_0_normal": counts.get(0, 0),
            "class_1_defective": counts.get(1, 0),
            "defective_rate_pct": f"{pct_defective:.2f}%"
        }
        
    return {
        "train": _split_meta("Train (70%)", train_df),
        "validation": _split_meta("Validation (15%)", val_df),
        "test": _split_meta("Test (15%)", test_df),
        "leakage_check_passed": True
    }


if __name__ == "__main__":
    df = load_ml_ready_data()
    train_df, val_df, test_df = split_components(df)
    summary = get_split_summary(train_df, val_df, test_df)
    print("Component Splitting Verification:")
    for s_name in ['train', 'validation', 'test']:
        print(f"  {summary[s_name]['split']}: {summary[s_name]['components']} components, Defective Rate = {summary[s_name]['defective_rate_pct']}")
    print(f"  Component Mutual Exclusivity: {summary['leakage_check_passed']}")
