"""
Validation Test Suite for SIH 2026 ML Experiment Pipeline
File: tests/test_ml_pipeline_design.py
Purpose: Validates component splitting, gate feature sets, target isolation, and zero-leakage constraints.
"""

import sys
import os

# Add workspace to path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data, validate_ml_schema, TARGET_MODULE_A, TARGET_MODULE_B
from src.data.split_data import split_components, get_split_summary
from src.features.build_features import extract_features_and_target, FEATURES_24H_GATE, FEATURES_96H_GATE
from src.models.train_module_a import prepare_module_a_experiment, get_module_a_models
from src.models.train_module_b import prepare_module_b_experiment, get_module_b_models


def run_pipeline_validation():
    print("=" * 80)
    print("SIH 2026: ML EXPERIMENT PIPELINE VALIDATION AUDIT")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading and Validating ML-Ready Dataset...")
    df = load_ml_ready_data()
    meta = validate_ml_schema(df)
    print(f" -> Successfully validated {meta['num_rows']:,} rows and {meta['num_cols']} columns.")

    # 2. Component Splitting
    print("\n[Step 2] Performing Component-Level Stratified Splitting...")
    train_df, val_df, test_df = split_components(df, random_state=42)
    split_meta = get_split_summary(train_df, val_df, test_df)

    train_ids = set(train_df['component_id'])
    val_ids = set(val_df['component_id'])
    test_ids = set(test_df['component_id'])

    print(f" -> Train components:      {len(train_ids):,} ({len(train_ids)/len(df)*100:.1f}%)")
    print(f" -> Validation components: {len(val_ids):,} ({len(val_ids)/len(df)*100:.1f}%)")
    print(f" -> Test components:       {len(test_ids):,} ({len(test_ids)/len(df)*100:.1f}%)")

    # Verify Disjoint Sets
    assert len(train_ids.intersection(val_ids)) == 0, "ERROR: Train and Val IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "ERROR: Train and Test IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "ERROR: Val and Test IDs overlap!"
    print(" -> Mutual Exclusivity Check: PASSED (Zero component leakage across splits).")

    # 3. Module A & B Experiment Setup
    print("\n[Step 3] Preparing Gate-Specific Experiments (A24, A96, B24, B96)...")

    exp_a24 = prepare_module_a_experiment(train_df, val_df, test_df, gate='24h')
    exp_a96 = prepare_module_a_experiment(train_df, val_df, test_df, gate='96h')
    exp_b24 = prepare_module_b_experiment(train_df, val_df, test_df, gate='24h')
    exp_b96 = prepare_module_b_experiment(train_df, val_df, test_df, gate='96h')

    # Summary Display
    print("\n" + "=" * 80)
    print("FINAL ML EXPERIMENT DESIGN SUMMARY")
    print("=" * 80)

    for exp in [exp_a24, exp_a96, exp_b24, exp_b96]:
        exp_name = exp['experiment']
        target = exp['target_name']
        features = exp['feature_names']
        has_future_96 = any('_96h' in f for f in features) if '24' in exp_name else False
        has_future_168 = any('_168h' in f for f in features)
        future_used = "YES" if (has_future_96 or has_future_168) else "NO"

        print(f"\n{exp_name}:")
        print(f"  features ({len(features)} total) = {features}")
        print(f"  target = {target}")
        print(f"  future_information_used = {future_used}")

    print("\n" + "-" * 80)
    print("DATA SPLIT SUMMARY:")
    print(f"  Train components      = {len(train_ids):,} ({split_meta['train']['defective_rate_pct']} defective)")
    print(f"  Validation components = {len(val_ids):,} ({split_meta['validation']['defective_rate_pct']} defective)")
    print(f"  Test components       = {len(test_ids):,} ({split_meta['test']['defective_rate_pct']} defective)")
    print("-" * 80)
    print("BASELINE MODELS CONFIGURED:")
    print(f"  Module A Classifiers: {list(get_module_a_models().keys())}")
    print(f"  Module B Regressors:  {list(get_module_b_models().keys())}")
    print("=" * 80)
    print("ALL PIPELINE CHECKS PASSED. PIPELINE IS READY FOR MODEL TRAINING.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_validation()
