"""
Script / Module: src.models.train_module_a
Purpose: End-to-end training, validation, selection, testing, and artifact serialization
         for Module A (Early Anomaly & Drift Classification) at 24h (A24) and 96h (A96) gates.
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure workspace root is in path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data
from src.data.split_data import split_components, get_split_summary
from src.features.build_features import build_preprocessing_pipeline, extract_features_and_target
from src.models.evaluate import evaluate_classification

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, confusion_matrix

# Directories
MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
REPORTS_DIR = os.path.join(WORKSPACE_DIR, 'reports')
PLOTS_DIR = os.path.join(WORKSPACE_DIR, 'eda', 'outputs', 'ml', 'module_a')

for d in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_candidate_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns candidate models with their specific scaling requirements.
    """
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
            "scale_features": True
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=random_state, class_weight='balanced'),
            "scale_features": False
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state),
            "scale_features": False
        }
    }


def train_and_evaluate_gate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    gate: str,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Pipeline]]:
    """
    Trains all candidate models on Train set, evaluates on Validation set.
    """
    exp_name = f"A{gate.replace('h', '')}"
    X_train, y_train = extract_features_and_target(train_df, exp_name)
    X_val, y_val = extract_features_and_target(val_df, exp_name)
    
    candidates = get_candidate_models(random_state=random_state)
    val_results = {}
    trained_pipelines = {}
    
    print(f"\n--- Training Candidate Models for Experiment {exp_name} ({gate} Gate: {X_train.shape[1]} Features) ---")
    
    for name, cfg in candidates.items():
        preprocessor = build_preprocessing_pipeline(scale_features=cfg["scale_features"])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', cfg["model"])
        ])
        
        # Fit ONLY on Training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predict on Validation set
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        metrics = evaluate_classification(y_val, y_val_pred, y_val_prob)
        metrics["model"] = name
        metrics["gate"] = gate
        metrics["experiment"] = exp_name
        metrics["num_features"] = X_train.shape[1]
        val_results[name] = metrics
        
        print(f" [{name}] Val Recall(Class 1): {metrics['recall_class_1']*100:.2f}% | FNR: {metrics['false_negative_rate_fnr']*100:.2f}% | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']*100:.2f}%")
        
    return val_results, trained_pipelines


def select_best_model(val_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Selects the best model using primary reliability metric (Recall Class 1 / FNR),
    with F1 and ROC-AUC as tie-breakers.
    """
    df_val = pd.DataFrame.from_dict(val_results, orient='index')
    # Rank primarily on Recall (highest), then F1 (highest), then ROC-AUC (highest)
    df_val['score'] = df_val['recall_class_1'] * 0.50 + df_val['f1_score'] * 0.30 + df_val['roc_auc'] * 0.20
    best_model_name = df_val.sort_values(by='score', ascending=False).index[0]
    return best_model_name


def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Normal (0)', 'Defective (1)'],
                yticklabels=['Normal (0)', 'Defective (1)'])
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_roc_curves(test_evals: Dict[str, Dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(7, 6))
    for exp_name, res in test_evals.items():
        fpr_pts, tpr_pts, _ = roc_curve(res['y_test'], res['y_prob'])
        auc_val = res['metrics']['roc_auc']
        ax.plot(fpr_pts, tpr_pts, lw=2, label=f"{exp_name} - {res['best_model']} (AUC = {auc_val:.4f})")
        
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Fallout)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_title('ROC Curves: Module A Locked Test Evaluation', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'module_a_roc_curves_test.png'), dpi=300)
    plt.close()


def plot_feature_importance(pipeline: Pipeline, feature_names: List[str], title: str, filename: str):
    clf = pipeline.named_steps['classifier']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        idx_sort = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[idx_sort], y=[feature_names[i] for i in idx_sort], ax=ax, palette='Blues_r')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Relative Gini / Impurity Importance', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()


def run_module_a_training():
    print("=" * 80)
    print("SIH 2026: MODULE A (EARLY ANOMALY & DRIFT CLASSIFICATION) TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    df = load_ml_ready_data()
    train_df, val_df, test_df = split_components(df, random_state=42)
    split_meta = get_split_summary(train_df, val_df, test_df)
    
    print("\n[Step 1] Dataset Partition Summary:")
    print(f" -> Train:      {len(train_df)} components (Class 0: {split_meta['train']['class_0_normal']}, Class 1: {split_meta['train']['class_1_defective']})")
    print(f" -> Validation: {len(val_df)} components (Class 0: {split_meta['validation']['class_0_normal']}, Class 1: {split_meta['validation']['class_1_defective']})")
    print(f" -> Test:       {len(test_df)} components (Class 0: {split_meta['test']['class_0_normal']}, Class 1: {split_meta['test']['class_1_defective']})")
    
    # 2. Train & Validate A24 and A96
    val_results_a24, pipelines_a24 = train_and_evaluate_gate(train_df, val_df, gate='24h', random_state=42)
    val_results_a96, pipelines_a96 = train_and_evaluate_gate(train_df, val_df, gate='96h', random_state=42)
    
    # Save validation table
    all_val_rows = list(val_results_a24.values()) + list(val_results_a96.values())
    df_val_all = pd.DataFrame(all_val_rows)
    df_val_all.to_csv(os.path.join(REPORTS_DIR, 'module_a_validation_results.csv'), index=False)
    
    # 3. Model Selection
    best_a24_name = select_best_model(val_results_a24)
    best_a96_name = select_best_model(val_results_a96)
    
    print("\n[Step 2] Model Selection (Based STRICTLY on Validation Set):")
    print(f" -> Selected Best A24 Model: {best_a24_name}")
    print(f"    Validation Recall(1): {val_results_a24[best_a24_name]['recall_class_1']*100:.2f}% | F1: {val_results_a24[best_a24_name]['f1_score']:.4f} | FNR: {val_results_a24[best_a24_name]['false_negative_rate_fnr']*100:.2f}%")
    print(f" -> Selected Best A96 Model: {best_a96_name}")
    print(f"    Validation Recall(1): {val_results_a96[best_a96_name]['recall_class_1']*100:.2f}% | F1: {val_results_a96[best_a96_name]['f1_score']:.4f} | FNR: {val_results_a96[best_a96_name]['false_negative_rate_fnr']*100:.2f}%")
    
    # 4. Final Evaluation on Locked Test Set
    print("\n[Step 3] Evaluating Selected Models ONCE on Locked Test Set...")
    
    selected_a24_pipeline = pipelines_a24[best_a24_name]
    selected_a96_pipeline = pipelines_a96[best_a96_name]
    
    X_test_a24, y_test_a24 = extract_features_and_target(test_df, 'A24')
    X_test_a96, y_test_a96 = extract_features_and_target(test_df, 'A96')
    
    y_test_pred_a24 = selected_a24_pipeline.predict(X_test_a24)
    y_test_prob_a24 = selected_a24_pipeline.predict_proba(X_test_a24)[:, 1]
    test_metrics_a24 = evaluate_classification(y_test_a24, y_test_pred_a24, y_test_prob_a24)
    test_metrics_a24["experiment"] = "A24"
    test_metrics_a24["model"] = best_a24_name
    test_metrics_a24["screening_gate"] = "24h"
    
    y_test_pred_a96 = selected_a96_pipeline.predict(X_test_a96)
    y_test_prob_a96 = selected_a96_pipeline.predict_proba(X_test_a96)[:, 1]
    test_metrics_a96 = evaluate_classification(y_test_a96, y_test_pred_a96, y_test_prob_a96)
    test_metrics_a96["experiment"] = "A96"
    test_metrics_a96["model"] = best_a96_name
    test_metrics_a96["screening_gate"] = "96h"
    
    df_test_results = pd.DataFrame([test_metrics_a24, test_metrics_a96])
    df_test_results.to_csv(os.path.join(REPORTS_DIR, 'module_a_test_results.csv'), index=False)
    
    # Gate Comparison Table
    comparison_table = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "A24 (24h Gate)": f"{test_metrics_a24['accuracy']*100:.2f}%",
            "A96 (96h Gate)": f"{test_metrics_a96['accuracy']*100:.2f}%",
            "Improvement (A96 vs A24)": f"{(test_metrics_a96['accuracy'] - test_metrics_a24['accuracy'])*100:+.2f}%"
        },
        {
            "Metric": "Precision (Class 1)",
            "A24 (24h Gate)": f"{test_metrics_a24['precision']*100:.2f}%",
            "A96 (96h Gate)": f"{test_metrics_a96['precision']*100:.2f}%",
            "Improvement (A96 vs A24)": f"{(test_metrics_a96['precision'] - test_metrics_a24['precision'])*100:+.2f}%"
        },
        {
            "Metric": "Recall (Class 1) [Primary]",
            "A24 (24h Gate)": f"{test_metrics_a24['recall_class_1']*100:.2f}%",
            "A96 (96h Gate)": f"{test_metrics_a96['recall_class_1']*100:.2f}%",
            "Improvement (A96 vs A24)": f"{(test_metrics_a96['recall_class_1'] - test_metrics_a24['recall_class_1'])*100:+.2f}%"
        },
        {
            "Metric": "F1-Score",
            "A24 (24h Gate)": f"{test_metrics_a24['f1_score']:.4f}",
            "A96 (96h Gate)": f"{test_metrics_a96['f1_score']:.4f}",
            "Improvement (A96 vs A24)": f"{test_metrics_a96['f1_score'] - test_metrics_a24['f1_score']:+.4f}"
        },
        {
            "Metric": "False Negative Rate (Defect Escape)",
            "A24 (24h Gate)": f"{test_metrics_a24['false_negative_rate_fnr']*100:.2f}%",
            "A96 (96h Gate)": f"{test_metrics_a96['false_negative_rate_fnr']*100:.2f}%",
            "Improvement (A96 vs A24)": f"{(test_metrics_a96['false_negative_rate_fnr'] - test_metrics_a24['false_negative_rate_fnr'])*100:+.2f}%"
        },
        {
            "Metric": "False Positive Rate (False Alarms)",
            "A24 (24h Gate)": f"{test_metrics_a24['false_positive_rate_fpr']*100:.2f}%",
            "A96 (96h Gate)": f"{test_metrics_a96['false_positive_rate_fpr']*100:.2f}%",
            "Improvement (A96 vs A24)": f"{(test_metrics_a96['false_positive_rate_fpr'] - test_metrics_a24['false_positive_rate_fpr'])*100:+.2f}%"
        },
        {
            "Metric": "ROC-AUC",
            "A24 (24h Gate)": f"{test_metrics_a24['roc_auc']:.4f}",
            "A96 (96h Gate)": f"{test_metrics_a96['roc_auc']:.4f}",
            "Improvement (A96 vs A24)": f"{test_metrics_a96['roc_auc'] - test_metrics_a24['roc_auc']:+.4f}"
        }
    ])
    comparison_table.to_csv(os.path.join(REPORTS_DIR, 'module_a_gate_comparison.csv'), index=False)
    
    # 5. Visualizations
    cm_a24 = confusion_matrix(y_test_a24, y_test_pred_a24)
    cm_a96 = confusion_matrix(y_test_a96, y_test_pred_a96)
    
    plot_confusion_matrix(cm_a24, f"A24 Confusion Matrix (Test Set: {best_a24_name})", "a24_confusion_matrix_test.png")
    plot_confusion_matrix(cm_a96, f"A96 Confusion Matrix (Test Set: {best_a96_name})", "a96_confusion_matrix_test.png")
    
    plot_roc_curves({
        "A24 (24h)": {"metrics": test_metrics_a24, "y_test": y_test_a24, "y_prob": y_test_prob_a24, "best_model": best_a24_name},
        "A96 (96h)": {"metrics": test_metrics_a96, "y_test": y_test_a96, "y_prob": y_test_prob_a96, "best_model": best_a96_name}
    })
    
    plot_feature_importance(selected_a24_pipeline, list(X_test_a24.columns), "A24 Feature Importances (24h Gate)", "a24_feature_importances.png")
    plot_feature_importance(selected_a96_pipeline, list(X_test_a96.columns), "A96 Feature Importances (96h Gate)", "a96_feature_importances.png")
    
    # 6. Save Model Artifacts
    path_a24 = os.path.join(MODELS_DIR, f"module_a_24h_{best_a24_name.lower()}.joblib")
    path_a96 = os.path.join(MODELS_DIR, f"module_a_96h_{best_a96_name.lower()}.joblib")
    
    joblib.dump(selected_a24_pipeline, path_a24)
    joblib.dump(selected_a96_pipeline, path_a96)
    
    print(f"\n[Step 4] Serialized Final Models:")
    print(f" -> A24 Artifact: {path_a24}")
    print(f" -> A96 Artifact: {path_a96}")
    
    return {
        "val_results_a24": val_results_a24,
        "val_results_a96": val_results_a96,
        "best_a24_name": best_a24_name,
        "best_a96_name": best_a96_name,
        "test_metrics_a24": test_metrics_a24,
        "test_metrics_a96": test_metrics_a96,
        "comparison_table": comparison_table,
        "path_a24": path_a24,
        "path_a96": path_a96
    }


if __name__ == "__main__":
    results = run_module_a_training()
