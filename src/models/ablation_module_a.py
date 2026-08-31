"""
Script: src.models.ablation_module_a
Purpose: Controlled Robustness & Ablation Analysis for Module A.
         Quantifies performance dependency on engineered drift features vs raw multi-point measurements.
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data, TARGET_MODULE_A
from src.data.split_data import split_components
from src.features.build_features import build_preprocessing_pipeline, FEATURES_24H_GATE, FEATURES_96H_GATE
from src.models.evaluate import evaluate_classification

REPORTS_DIR = os.path.join(WORKSPACE_DIR, 'reports')
ABLATION_PLOTS_DIR = os.path.join(WORKSPACE_DIR, 'eda', 'outputs', 'ml', 'module_a', 'ablation')
os.makedirs(ABLATION_PLOTS_DIR, exist_ok=True)

# Define Ablation Feature Subsets
DRIFT_COLS_24H = ['iddq_drift_24h_pct']
DRIFT_COLS_96H = ['iddq_drift_24h_pct', 'iddq_drift_96h_pct', 'leakage_drift_96h_pct', 'delay_drift_96h_pct']

FEATURES_A24_FULL = list(FEATURES_24H_GATE)
FEATURES_A24_NO_DRIFT = [c for c in FEATURES_24H_GATE if c not in DRIFT_COLS_24H]

FEATURES_A96_FULL = list(FEATURES_96H_GATE)
FEATURES_A96_NO_DRIFT = [c for c in FEATURES_96H_GATE if c not in DRIFT_COLS_96H]

EXPERIMENTS = {
    "A24-FULL": {
        "features": FEATURES_A24_FULL,
        "description": "0h + 24h Raw Sensors + 24h Iddq Drift"
    },
    "A24-NO-DRIFT": {
        "features": FEATURES_A24_NO_DRIFT,
        "description": "0h + 24h Raw Sensors ONLY (No Drift Features)"
    },
    "A96-FULL": {
        "features": FEATURES_A96_FULL,
        "description": "0h + 24h + 96h Raw Sensors + 24h/96h Multi-Drift"
    },
    "A96-NO-DRIFT": {
        "features": FEATURES_A96_NO_DRIFT,
        "description": "0h + 24h + 96h Raw Sensors ONLY (No Drift Features)"
    }
}


def run_ablation_experiments():
    print("=" * 80)
    print("SIH 2026: MODULE A ROBUSTNESS & ABLATION STUDY")
    print("=" * 80)
    
    # Load and split
    df = load_ml_ready_data()
    train_df, val_df, test_df = split_components(df, random_state=42)
    
    results = []
    roc_data = {}
    
    for exp_name, exp_cfg in EXPERIMENTS.items():
        feat_list = exp_cfg["features"]
        
        X_train = train_df[feat_list].copy()
        y_train = train_df[TARGET_MODULE_A].copy()
        
        X_test = test_df[feat_list].copy()
        y_test = test_df[TARGET_MODULE_A].copy()
        
        # Controlled Random Forest model
        preprocessor = build_preprocessing_pipeline(scale_features=False)
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', rf_model)
        ])
        
        # Fit on train, predict on test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_classification(y_test, y_pred, y_prob)
        metrics["Experiment"] = exp_name
        metrics["Num_Features"] = len(feat_list)
        metrics["Description"] = exp_cfg["description"]
        results.append(metrics)
        
        roc_data[exp_name] = {
            "y_test": y_test,
            "y_prob": y_prob,
            "auc": metrics["roc_auc"]
        }
        
        print(f"\n[{exp_name}] ({len(feat_list)} Features):")
        print(f" -> Accuracy: {metrics['accuracy']*100:.2f}% | Recall(1): {metrics['recall_class_1']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | F1: {metrics['f1_score']:.4f} | FNR: {metrics['false_negative_rate_fnr']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f}")

    df_results = pd.DataFrame(results)
    
    # Save results CSV
    df_results.to_csv(os.path.join(REPORTS_DIR, 'module_a_ablation_results.csv'), index=False)
    
    # Create Comparison Table
    comp_df = pd.DataFrame([
        {
            "Experiment": r["Experiment"],
            "Features": r["Num_Features"],
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall (Class 1)": f"{r['recall_class_1']*100:.2f}%",
            "F1-Score": f"{r['f1_score']:.4f}",
            "FNR (Escapes)": f"{r['false_negative_rate_fnr']*100:.2f}%",
            "FPR (False Alarms)": f"{r['false_positive_rate_fpr']*100:.2f}%",
            "ROC-AUC": f"{r['roc_auc']:.4f}"
        }
        for r in results
    ])
    comp_df.to_csv(os.path.join(REPORTS_DIR, 'module_a_ablation_comparison_table.csv'), index=False)
    
    # Plot ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'A24-FULL': '#1f77b4', 'A24-NO-DRIFT': '#aec7e8', 'A96-FULL': '#2ca02c', 'A96-NO-DRIFT': '#98df8a'}
    for name, d in roc_data.items():
        fpr, tpr, _ = roc_curve(d["y_test"], d["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {d['auc']:.4f})", color=colors[name], lw=2.2, linestyle='-' if 'FULL' in name else '--')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax.set_title('ROC Curves: Module A Ablation (With vs Without Engineered Drift)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('False Positive Rate (Fallout)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_PLOTS_DIR, 'ablation_roc_comparison.png'), dpi=300)
    plt.close()
    
    # Plot Bar Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recall comparison
    sns.barplot(data=df_results, x='Experiment', y='recall_class_1', ax=axes[0], palette=['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a'])
    axes[0].set_title('Class 1 Recall (Defect Capture) Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Recall (Class 1)', fontsize=11)
    axes[0].set_ylim(0, 1.05)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')
                         
    # F1 comparison
    sns.barplot(data=df_results, x='Experiment', y='f1_score', ax=axes[1], palette=['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a'])
    axes[1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=11)
    axes[1].set_ylim(0, 1.05)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')
                         
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_PLOTS_DIR, 'ablation_metrics_barplot.png'), dpi=300)
    plt.close()
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation_experiments()
