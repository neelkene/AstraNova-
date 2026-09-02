"""
Module: src.models.train_module_b
Purpose: End-to-end training, validation benchmarking, model selection, locked test evaluation,
         diagnostic visualization, and artifact serialization for Module B (168h Continuous Degradation Forecasting).
Experiments:
- B24 (24h Gate: 11 Features -> iddq_drift_168h_true)
- B96 (96h Gate: 19 Features -> iddq_drift_168h_true)
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
from src.models.evaluate import evaluate_regression

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Directories
MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
REPORTS_DIR = os.path.join(WORKSPACE_DIR, 'reports')
PLOTS_DIR = os.path.join(WORKSPACE_DIR, 'eda', 'outputs', 'ml', 'module_b')

for d in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_module_b_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns the regression candidate model suite:
    1. LinearRegression (with StandardScaler)
    2. Ridge Regression (with StandardScaler)
    3. RandomForestRegressor (Tree ensemble, unscaled)
    4. GradientBoostingRegressor (Boosted trees, unscaled)
    """
    return {
        "LinearRegression": {
            "model": LinearRegression(),
            "scale_features": True
        },
        "Ridge": {
            "model": Ridge(alpha=1.0, random_state=random_state),
            "scale_features": True
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
            "scale_features": False
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state),
            "scale_features": False
        }
    }


def build_module_b_pipeline(
    model_name: str,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a complete end-to-end Pipeline (imputer -> optional scaler -> regressor).
    """
    models_dict = get_module_b_models(random_state=random_state)
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: {list(models_dict.keys())}")
        
    cfg = models_dict[model_name]
    preprocessor = build_preprocessing_pipeline(scale_features=cfg["scale_features"])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', cfg["model"])
    ])
    return pipeline


def prepare_module_b_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gate: str = '24h'
) -> Dict[str, Any]:
    """
    Prepares X and y splits for Module B at specified gate ('24h' for B24, '96h' for B96).
    """
    exp_name = f"B{gate.replace('h', '')}"
    X_train, y_train = extract_features_and_target(train_df, exp_name)
    X_val, y_val = extract_features_and_target(val_df, exp_name)
    X_test, y_test = extract_features_and_target(test_df, exp_name)
    
    return {
        "experiment": exp_name,
        "gate": gate,
        "feature_names": list(X_train.columns),
        "target_name": y_train.name,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


def train_and_evaluate_gate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    gate: str,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Pipeline]]:
    """
    Trains all candidate regression models on Train set, evaluates on Validation set.
    """
    exp_name = f"B{gate.replace('h', '')}"
    X_train, y_train = extract_features_and_target(train_df, exp_name)
    X_val, y_val = extract_features_and_target(val_df, exp_name)
    
    candidates = get_module_b_models(random_state=random_state)
    val_results = {}
    trained_pipelines = {}
    
    print(f"\n--- Training Candidate Models for Experiment {exp_name} ({gate} Gate: {X_train.shape[1]} Features) ---")
    
    for name, cfg in candidates.items():
        preprocessor = build_preprocessing_pipeline(scale_features=cfg["scale_features"])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', cfg["model"])
        ])
        
        # Fit ONLY on Training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predict on Validation set
        y_val_pred = pipeline.predict(X_val)
        metrics = evaluate_regression(y_val, y_val_pred)
        metrics["model"] = name
        metrics["gate"] = gate
        metrics["experiment"] = exp_name
        metrics["num_features"] = X_train.shape[1]
        val_results[name] = metrics
        
        print(f" [{name}] Val RMSE: {metrics['rmse']:.6f} | MAE: {metrics['mae']:.6f} | R²: {metrics['r2_score']:.4f}")
        
    return val_results, trained_pipelines


def select_best_regression_model(val_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Selects the best regression model based on minimum validation RMSE (and highest R²).
    """
    df_val = pd.DataFrame.from_dict(val_results, orient='index')
    best_model_name = df_val.sort_values(by='rmse', ascending=True).index[0]
    return best_model_name


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
    r2_val: float,
    rmse_val: float
):
    """
    Generates scatter plot of Actual vs Predicted 168h degradation with ideal 1:1 line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true * 100, y_pred * 100, alpha=0.35, color='#1f77b4', edgecolors='none', s=25)
    
    # Ideal 1:1 line
    min_val = min(np.min(y_true), np.min(y_pred)) * 100
    max_val = max(np.max(y_true), np.max(y_pred)) * 100
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Perfect Forecast (1:1)')
    
    ax.set_xlabel('Actual 168h Iddq Drift (%)', fontsize=11)
    ax.set_ylabel('Predicted 168h Iddq Drift (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', frameon=True)
    
    # Annotate stats
    ax.text(0.05, 0.85, f"R² = {r2_val:.4f}\nRMSE = {rmse_val*100:.2f}%\nMAE = {np.mean(np.abs(y_true - y_pred))*100:.2f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str
):
    """
    Generates Residuals vs Predicted plot and Residual distribution histogram.
    """
    residuals = (y_true - y_pred) * 100
    pred_pct = y_pred * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(pred_pct, residuals, alpha=0.35, color='#2ca02c', edgecolors='none', s=25)
    axes[0].axhline(0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted 168h Drift (%)', fontsize=11)
    axes[0].set_ylabel('Residual (Actual - Predicted) (%)', fontsize=11)
    axes[0].set_title(f'{title} — Residuals vs Predicted', fontsize=11, fontweight='bold')
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#2ca02c', bins=40)
    axes[1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual Error (%)', fontsize=11)
    axes[1].set_ylabel('Component Count', fontsize=11)
    axes[1].set_title(f'{title} — Residual Error Distribution', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_metrics_comparison(
    val_results_b24: Dict[str, Dict[str, Any]],
    val_results_b96: Dict[str, Dict[str, Any]]
):
    """
    Bar plots comparing RMSE, MAE, and R² across candidate models and screening gates.
    """
    rows = []
    for m_name, res in val_results_b24.items():
        rows.append({"Model": m_name, "Gate": "B24 (24h)", "RMSE": res["rmse"] * 100, "MAE": res["mae"] * 100, "R2": res["r2_score"]})
    for m_name, res in val_results_b96.items():
        rows.append({"Model": m_name, "Gate": "B96 (96h)", "RMSE": res["rmse"] * 100, "MAE": res["mae"] * 100, "R2": res["r2_score"]})
        
    df_plot = pd.DataFrame(rows)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # RMSE
    sns.barplot(data=df_plot, x='Model', y='RMSE', hue='Gate', ax=axes[0], palette=['#1f77b4', '#2ca02c'])
    axes[0].set_title('Root Mean Squared Error (RMSE %)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('RMSE (%) — Lower is Better', fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)
    
    # MAE
    sns.barplot(data=df_plot, x='Model', y='MAE', hue='Gate', ax=axes[1], palette=['#1f77b4', '#2ca02c'])
    axes[1].set_title('Mean Absolute Error (MAE %)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('MAE (%) — Lower is Better', fontsize=10)
    axes[1].tick_params(axis='x', rotation=30)
    
    # R²
    sns.barplot(data=df_plot, x='Model', y='R2', hue='Gate', ax=axes[2], palette=['#1f77b4', '#2ca02c'])
    axes[2].set_title('Coefficient of Determination (R²)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('R² Score — Higher is Better', fontsize=10)
    axes[2].set_ylim(0, 1.05)
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'module_b_regression_metrics_comparison.png'), dpi=300)
    plt.close()


def plot_feature_importance_regression(
    pipeline: Pipeline,
    feature_names: List[str],
    title: str,
    filename: str
):
    """
    Plots feature importances (or regression coefficients) for the selected best model.
    """
    reg = pipeline.named_steps['regressor']
    if hasattr(reg, 'feature_importances_'):
        importances = reg.feature_importances_
        idx_sort = np.argsort(importances)[::-1]
        sorted_feats = [feature_names[i] for i in idx_sort]
        sorted_vals = importances[idx_sort]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sorted_vals, y=sorted_feats, ax=ax, hue=sorted_feats, palette='viridis_r', legend=False)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Relative Feature Importance', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()
    elif hasattr(reg, 'coef_'):
        coefs = np.abs(reg.coef_)
        idx_sort = np.argsort(coefs)[::-1]
        sorted_feats = [feature_names[i] for i in idx_sort]
        sorted_vals = coefs[idx_sort]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sorted_vals, y=sorted_feats, ax=ax, hue=sorted_feats, palette='viridis_r', legend=False)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Absolute Standardized Coefficient Magnitude', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()


def run_module_b_training() -> Dict[str, Any]:
    print("=" * 80)
    print("SIH 2026: MODULE B (168h CONTINUOUS DEGRADATION FORECASTING) PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    df = load_ml_ready_data()
    train_df, val_df, test_df = split_components(df, random_state=42)
    split_meta = get_split_summary(train_df, val_df, test_df)
    
    print("\n[Step 1] Dataset Partition Summary:")
    print(f" -> Train:      {len(train_df)} components")
    print(f" -> Validation: {len(val_df)} components")
    print(f" -> Locked Test:{len(test_df)} components")
    
    # 2. Train & Validate B24 (24h gate) and B96 (96h gate)
    val_results_b24, pipelines_b24 = train_and_evaluate_gate(train_df, val_df, gate='24h', random_state=42)
    val_results_b96, pipelines_b96 = train_and_evaluate_gate(train_df, val_df, gate='96h', random_state=42)
    
    # Save validation table
    all_val_rows = list(val_results_b24.values()) + list(val_results_b96.values())
    df_val_all = pd.DataFrame(all_val_rows)
    df_val_all.to_csv(os.path.join(REPORTS_DIR, 'module_b_validation_results.csv'), index=False)
    
    # 3. Model Selection
    best_b24_name = select_best_regression_model(val_results_b24)
    best_b96_name = select_best_regression_model(val_results_b96)
    
    print("\n[Step 2] Model Selection (Based STRICTLY on Validation Set):")
    print(f" -> Selected Best B24 Model: {best_b24_name}")
    print(f"    Val RMSE: {val_results_b24[best_b24_name]['rmse']:.6f} | MAE: {val_results_b24[best_b24_name]['mae']:.6f} | R²: {val_results_b24[best_b24_name]['r2_score']:.4f}")
    print(f" -> Selected Best B96 Model: {best_b96_name}")
    print(f"    Val RMSE: {val_results_b96[best_b96_name]['rmse']:.6f} | MAE: {val_results_b96[best_b96_name]['mae']:.6f} | R²: {val_results_b96[best_b96_name]['r2_score']:.4f}")
    
    # 4. Final Evaluation on Locked Test Set
    print("\n[Step 3] Evaluating Selected Models ONCE on Locked Test Set ($N=1,500$)...")
    
    selected_b24_pipeline = pipelines_b24[best_b24_name]
    selected_b96_pipeline = pipelines_b96[best_b96_name]
    
    X_test_b24, y_test_b24 = extract_features_and_target(test_df, 'B24')
    X_test_b96, y_test_b96 = extract_features_and_target(test_df, 'B96')
    
    y_test_pred_b24 = selected_b24_pipeline.predict(X_test_b24)
    test_metrics_b24 = evaluate_regression(y_test_b24, y_test_pred_b24)
    test_metrics_b24["experiment"] = "B24"
    test_metrics_b24["model"] = best_b24_name
    test_metrics_b24["screening_gate"] = "24h"
    test_metrics_b24["num_features"] = X_test_b24.shape[1]
    
    y_test_pred_b96 = selected_b96_pipeline.predict(X_test_b96)
    test_metrics_b96 = evaluate_regression(y_test_b96, y_test_pred_b96)
    test_metrics_b96["experiment"] = "B96"
    test_metrics_b96["model"] = best_b96_name
    test_metrics_b96["screening_gate"] = "96h"
    test_metrics_b96["num_features"] = X_test_b96.shape[1]
    
    df_test_results = pd.DataFrame([test_metrics_b24, test_metrics_b96])
    df_test_results.to_csv(os.path.join(REPORTS_DIR, 'module_b_test_results.csv'), index=False)
    
    print("\n[Test Set Performance]")
    print(f" -> B24 ({best_b24_name}): RMSE = {test_metrics_b24['rmse']*100:.3f}% | MAE = {test_metrics_b24['mae']*100:.3f}% | R² = {test_metrics_b24['r2_score']:.4f}")
    print(f" -> B96 ({best_b96_name}): RMSE = {test_metrics_b96['rmse']*100:.3f}% | MAE = {test_metrics_b96['mae']*100:.3f}% | R² = {test_metrics_b96['r2_score']:.4f}")
    
    # Gate Comparison Table
    rmse_delta = (test_metrics_b96['rmse'] - test_metrics_b24['rmse']) * 100
    mae_delta = (test_metrics_b96['mae'] - test_metrics_b24['mae']) * 100
    r2_delta = test_metrics_b96['r2_score'] - test_metrics_b24['r2_score']
    
    comparison_table = pd.DataFrame([
        {
            "Metric": "Root Mean Squared Error (RMSE)",
            "B24 (24h Gate)": f"{test_metrics_b24['rmse']*100:.3f}%",
            "B96 (96h Gate)": f"{test_metrics_b96['rmse']*100:.3f}%",
            "Improvement (B96 vs B24)": f"{rmse_delta:.3f}% ({((test_metrics_b24['rmse'] - test_metrics_b96['rmse'])/test_metrics_b24['rmse'])*100:.1f}% relative reduction)"
        },
        {
            "Metric": "Mean Absolute Error (MAE)",
            "B24 (24h Gate)": f"{test_metrics_b24['mae']*100:.3f}%",
            "B96 (96h Gate)": f"{test_metrics_b96['mae']*100:.3f}%",
            "Improvement (B96 vs B24)": f"{mae_delta:.3f}% ({((test_metrics_b24['mae'] - test_metrics_b96['mae'])/test_metrics_b24['mae'])*100:.1f}% relative reduction)"
        },
        {
            "Metric": "Coefficient of Determination (R²)",
            "B24 (24h Gate)": f"{test_metrics_b24['r2_score']:.4f}",
            "B96 (96h Gate)": f"{test_metrics_b96['r2_score']:.4f}",
            "Improvement (B96 vs B24)": f"{r2_delta:+.4f}"
        },
        {
            "Metric": "Mean Actual Degradation",
            "B24 (24h Gate)": f"{test_metrics_b24['mean_actual_drift']*100:.2f}%",
            "B96 (96h Gate)": f"{test_metrics_b96['mean_actual_drift']*100:.2f}%",
            "Improvement (B96 vs B24)": "Baseline Truth"
        },
        {
            "Metric": "Mean Forecasted Degradation",
            "B24 (24h Gate)": f"{test_metrics_b24['mean_predicted_drift']*100:.2f}%",
            "B96 (96h Gate)": f"{test_metrics_b96['mean_predicted_drift']*100:.2f}%",
            "Improvement (B96 vs B24)": "High calibration fidelity"
        }
    ])
    comparison_table.to_csv(os.path.join(REPORTS_DIR, 'module_b_gate_comparison.csv'), index=False)
    
    # 5. Diagnostic Visualizations
    plot_actual_vs_predicted(
        y_test_b24.values, y_test_pred_b24,
        f"B24: Actual vs Predicted 168h Drift ({best_b24_name})",
        "b24_actual_vs_predicted_test.png",
        test_metrics_b24['r2_score'], test_metrics_b24['rmse']
    )
    
    plot_actual_vs_predicted(
        y_test_b96.values, y_test_pred_b96,
        f"B96: Actual vs Predicted 168h Drift ({best_b96_name})",
        "b96_actual_vs_predicted_test.png",
        test_metrics_b96['r2_score'], test_metrics_b96['rmse']
    )
    
    plot_residuals(
        y_test_b24.values, y_test_pred_b24,
        f"B24 ({best_b24_name})",
        "b24_residuals_test.png"
    )
    
    plot_residuals(
        y_test_b96.values, y_test_pred_b96,
        f"B96 ({best_b96_name})",
        "b96_residuals_test.png"
    )
    
    plot_metrics_comparison(val_results_b24, val_results_b96)
    
    plot_feature_importance_regression(
        selected_b24_pipeline, list(X_test_b24.columns),
        f"B24 Feature Importances ({best_b24_name})",
        "b24_feature_importances.png"
    )
    
    plot_feature_importance_regression(
        selected_b96_pipeline, list(X_test_b96.columns),
        f"B96 Feature Importances ({best_b96_name})",
        "b96_feature_importances.png"
    )
    
    # 6. Save Model Artifacts
    path_b24 = os.path.join(MODELS_DIR, f"module_b_24h_{best_b24_name.lower()}.joblib")
    path_b96 = os.path.join(MODELS_DIR, f"module_b_96h_{best_b96_name.lower()}.joblib")
    
    joblib.dump(selected_b24_pipeline, path_b24)
    joblib.dump(selected_b96_pipeline, path_b96)
    
    print(f"\n[Step 4] Serialized Final Regressors:")
    print(f" -> B24 Artifact: {path_b24}")
    print(f" -> B96 Artifact: {path_b96}")
    
    return {
        "val_results_b24": val_results_b24,
        "val_results_b96": val_results_b96,
        "best_b24_name": best_b24_name,
        "best_b96_name": best_b96_name,
        "test_metrics_b24": test_metrics_b24,
        "test_metrics_b96": test_metrics_b96,
        "comparison_table": comparison_table,
        "path_b24": path_b24,
        "path_b96": path_b96
    }


if __name__ == "__main__":
    results = run_module_b_training()
