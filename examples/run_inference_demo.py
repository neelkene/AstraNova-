"""
Example Demonstration: End-to-End Burn-In Screening & Sequential Decision Pipeline
File: examples/run_inference_demo.py
Project: AI-Driven Anomaly Detection in Component Burn-In & Screening (SIH 2026)

Purpose:
Demonstrates the final inference and decision layer on real semiconductor burn-in samples:
1. 24h Early Screening Gate (Module A A24 + Module B B24)
2. 96h Mid Screening Gate (Module A A96 + Module B B96)
3. 2-Stage Sequential Early-Exit Workflow (Early PASS, Early REJECT, Continue to 96h)
4. Batch screening statistics showing chamber time savings.
"""

import os
import sys
import pandas as pd

# Ensure workspace root is in python path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data
from src.data.split_data import split_components
from src.inference.predict import load_models, predict_24h, predict_96h
from src.decision.screening_decision import (
    DecisionConfig,
    DEFAULT_CONFIG,
    make_screening_decision,
    run_screening_pipeline,
    run_sequential_screening,
)


def run_demo():
    print("=" * 85)
    print("SIH 2026: AI-DRIVEN COMPONENT BURN-IN SCREENING INFERENCE & DECISION DEMO")
    print("=" * 85)

    # 1. Load Data and Models
    print("\n[Step 1] Loading Trained Production Models and Test Dataset...")
    models = load_models()
    print(" -> Models loaded successfully:")
    for m_name in models.keys():
        print(f"    * {m_name}: {models[m_name].named_steps['classifier' if 'a' in m_name else 'regressor'].__class__.__name__}")

    df_all = load_ml_ready_data()
    _, _, test_df = split_components(df_all, random_state=42)
    print(f" -> Locked Test Set loaded: {len(test_df)} components")

    # Select representative samples from distinct classes
    # Ground truth: 0 = Normal, 1 = Defective (drifting / anomalous)
    sample_normal = test_df[test_df['module_a_label'] == 0].iloc[0]
    sample_drifting = test_df[(test_df['module_a_label'] == 1) & (test_df['iddq_drift_168h_true'] < 0.18)].iloc[0]
    sample_anomalous = test_df[(test_df['module_a_label'] == 1) & (test_df['iddq_drift_168h_true'] >= 0.18)].iloc[0]

    demo_components = [
        ("Normal Component (Healthy)", sample_normal),
        ("Latent Drifting Component (Marginal Defect)", sample_drifting),
        ("Gross Anomalous Component (Severe Defect)", sample_anomalous),
    ]

    # 2. Individual Component Walkthrough
    print("\n" + "=" * 85)
    print("[Step 2] Detailed Screening Walkthrough on Representative Components")
    print("=" * 85)

    for desc, comp in demo_components:
        comp_id = comp['component_id']
        actual_168h_drift = comp['iddq_drift_168h_true'] * 100.0
        ground_truth_label = "Defective (1)" if comp['module_a_label'] == 1 else "Normal (0)"

        print(f"\n" + "-" * 85)
        print(f"COMPONENT ID: {comp_id} | Type: {desc}")
        print(f"Ground Truth: Class = {ground_truth_label} | Actual 168h Iddq Drift = {actual_168h_drift:.2f}%")
        print("-" * 85)

        # Stage 1: 24h Screening
        res_24 = run_screening_pipeline(comp, gate="24h", config=DEFAULT_CONFIG, models=models)
        print(f"\n[24h Screening Gate]")
        print(f"  * Defect Probability:       {res_24['defect_probability']*100:.1f}%")
        print(f"  * Forecasted 168h Drift:    {res_24['predicted_168h_iddq_drift_pct']:.2f}%")
        print(f"  * Operational Decision:     {res_24['decision']} ({res_24['confidence_level']} confidence)")
        print(f"  * Reason:                   {res_24['reason']}")
        print(f"  * Action Recommendation:    {res_24['recommendation']}")

        # Stage 2: 96h Screening
        res_96 = run_screening_pipeline(comp, gate="96h", config=DEFAULT_CONFIG, models=models)
        print(f"\n[96h Screening Gate]")
        print(f"  * Defect Probability:       {res_96['defect_probability']*100:.1f}%")
        print(f"  * Forecasted 168h Drift:    {res_96['predicted_168h_iddq_drift_pct']:.2f}%")
        print(f"  * Operational Decision:     {res_96['decision']} ({res_96['confidence_level']} confidence)")
        print(f"  * Reason:                   {res_96['reason']}")
        print(f"  * Action Recommendation:    {res_96['recommendation']}")

        # Sequential Workflow
        seq_res = run_sequential_screening(comp, config=DEFAULT_CONFIG, models=models)
        print(f"\n[2-Stage Sequential Workflow Result]")
        print(f"  * Final Decision:           {seq_res['final_decision']} @ {seq_res['final_screening_gate']} gate")
        print(f"  * Early Exit Applied:       {seq_res['early_exit_applied']}")
        print(f"  * Executive Summary:        {seq_res['summary']}")

    # 3. Batch Simulation over Locked Test Partition
    print("\n" + "=" * 85)
    print("[Step 3] Batch Sequential Screening Simulation on Locked Test Set (N=1,500)")
    print("=" * 85)

    batch_results = run_sequential_screening(test_df, config=DEFAULT_CONFIG, models=models)
    
    total_parts = len(batch_results)
    early_exit_count = sum(1 for r in batch_results if r['early_exit_applied'])
    continued_96h_count = total_parts - early_exit_count

    early_pass_count = sum(1 for r in batch_results if r['early_exit_applied'] and r['final_decision'] == 'PASS')
    early_reject_count = sum(1 for r in batch_results if r['early_exit_applied'] and r['final_decision'] == 'REJECT')

    final_pass = sum(1 for r in batch_results if r['final_decision'] == 'PASS')
    final_reject = sum(1 for r in batch_results if r['final_decision'] == 'REJECT')
    final_review = sum(1 for r in batch_results if r['final_decision'] == 'REVIEW')

    print(f"\nBATCH SCREENING STATISTICS:")
    print(f"  Total Test Components Evaluated:   {total_parts:,}")
    print(f"  -> Early Exit at 24h:              {early_exit_count:,} ({early_exit_count/total_parts*100:.1f}%)")
    print(f"     * Early PASS (Safe Release):     {early_pass_count:,} ({early_pass_count/total_parts*100:.1f}%)")
    print(f"     * Early REJECT (Severe Defects): {early_reject_count:,} ({early_reject_count/total_parts*100:.1f}%)")
    print(f"  -> Continued to 96h Burn-In:        {continued_96h_count:,} ({continued_96h_count/total_parts*100:.1f}%)")
    print(f"\nFINAL DECISION BREAKDOWN:")
    print(f"  * PASS:   {final_pass:,} ({final_pass/total_parts*100:.1f}%)")
    print(f"  * REJECT: {final_reject:,} ({final_reject/total_parts*100:.1f}%)")
    print(f"  * REVIEW: {final_review:,} ({final_review/total_parts*100:.1f}%)")

    # Estimated Burn-In Chamber Time Savings
    # Standard testing: 1,500 parts * 168h = 252,000 component-hours
    # AI-screened: (early_exit * 24h) + (continued * 96h)
    std_hours = total_parts * 168.0
    ai_hours = (early_exit_count * 24.0) + (continued_96h_count * 96.0)
    savings_pct = ((std_hours - ai_hours) / std_hours) * 100.0

    print(f"\nCHAMBER ENERGY & TEST TIME OPTIMIZATION:")
    print(f"  * Standard 168h Stress Hours:      {std_hours:,.0f} component-hours")
    print(f"  * AI-Optimized Stress Hours:       {ai_hours:,.0f} component-hours")
    print(f"  * Net Test Chamber Time Reduction: {savings_pct:.1f}% reduction")
    print("=" * 85)


if __name__ == "__main__":
    run_demo()
