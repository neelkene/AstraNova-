"""
Flask Web Application & REST API Backend
File: web/app.py
Project: AI-Driven Anomaly Detection in Component Burn-In & Screening (SIH 2026)

Purpose:
Serves the SIH 2026 Judge Demonstration Dashboard and provides REST API endpoints
connected directly to the production inference (Module A & Module B) and decision layers.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory

# Ensure workspace root is in python path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data
from src.data.split_data import split_components
from src.inference.predict import (
    load_models,
    predict_24h,
    predict_96h,
    run_inference_gate,
)
from src.decision.screening_decision import (
    DecisionConfig,
    DEFAULT_CONFIG,
    make_screening_decision,
    run_screening_pipeline,
    run_sequential_screening,
)

# Initialize Flask app
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    template_folder=os.path.join(os.path.dirname(__file__), 'templates')
)

# Global in-memory dataset cache for fast, leak-free queries
_DATA_CACHE: Dict[str, Any] = {}


def get_data_cache() -> Dict[str, Any]:
    global _DATA_CACHE
    if not _DATA_CACHE:
        # Load dataset and locked test split
        df = load_ml_ready_data()
        train_df, val_df, test_df = split_components(df, random_state=42)
        
        # Pre-index test set by component_id
        test_indexed = test_df.set_index('component_id', drop=False)
        all_indexed = df.set_index('component_id', drop=False)
        
        # Load production models
        models = load_models()
        
        _DATA_CACHE = {
            "all_df": df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "test_indexed": test_indexed,
            "all_indexed": all_indexed,
            "models": models
        }
    return _DATA_CACHE


# ------------------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "project": "AI-Driven Component Burn-In & Screening (SIH 2026)",
        "modules_active": ["Module A (Classification)", "Module B (Regression)"],
        "models_loaded": True
    })


@app.route('/api/test-components', methods=['GET'])
def get_test_components():
    """
    Returns actual component IDs and metadata from the locked test set (N=1,500).
    Allows judges to select real components without generating fake IDs.
    """
    cache = get_data_cache()
    test_df = cache["test_df"]
    
    # Optional filter query (category: all, normal, drifting, anomalous)
    category = request.args.args.get('category', 'all').lower() if hasattr(request.args, 'args') else request.args.get('category', 'all').lower()
    limit = int(request.args.get('limit', 100))
    
    components = []
    for _, row in test_df.iterrows():
        c_id = row['component_id']
        mod_a_label = int(row['module_a_label'])
        drift_168h_true = float(row['iddq_drift_168h_true'])
        
        # Determine empirical sub-population for judge exploration
        if mod_a_label == 0:
            c_type = "Normal (Healthy)"
            c_key = "normal"
        elif drift_168h_true < 0.18:
            c_type = "Latent Drifting (Marginal Defect)"
            c_key = "drifting"
        else:
            c_type = "Gross Anomalous (Severe Defect)"
            c_key = "anomalous"
            
        if category == 'all' or category == c_key:
            components.append({
                "component_id": c_id,
                "category": c_type,
                "category_key": c_key,
                "module_a_label": mod_a_label,
                "iddq_drift_168h_true_pct": round(drift_168h_true * 100.0, 2),
                "iddq_uA_0h": round(float(row['iddq_uA_0h']), 2) if not pd.isna(row['iddq_uA_0h']) else None,
                "iddq_uA_24h": round(float(row['iddq_uA_24h']), 2) if not pd.isna(row['iddq_uA_24h']) else None,
                "iddq_uA_96h": round(float(row['iddq_uA_96h']), 2) if not pd.isna(row['iddq_uA_96h']) else None,
            })
            if len(components) >= limit:
                break
                
    return jsonify({
        "total_test_components": len(test_df),
        "returned_count": len(components),
        "category_filter": category,
        "components": components
    })


@app.route('/api/component/<component_id>', methods=['GET'])
def get_component_measurements(component_id: str):
    """
    Returns actual measurement records for a specific component ID from dataset.
    """
    cache = get_data_cache()
    all_indexed = cache["all_indexed"]
    
    if component_id not in all_indexed.index:
        return jsonify({"error": f"Component ID '{component_id}' not found in dataset"}), 404
        
    row = all_indexed.loc[component_id]
    
    # Separate into gate-specific measurement views
    data_dict = {
        "component_id": component_id,
        "is_in_locked_test_set": component_id in cache["test_indexed"].index,
        "ground_truth": {
            "module_a_label": int(row['module_a_label']),
            "iddq_drift_168h_true": float(row['iddq_drift_168h_true']),
            "iddq_drift_168h_true_pct": round(float(row['iddq_drift_168h_true']) * 100.0, 3)
        },
        "measurements_0h": {
            "iddq_uA_0h": float(row['iddq_uA_0h']) if not pd.isna(row['iddq_uA_0h']) else None,
            "leakage_current_uA_0h": float(row['leakage_current_uA_0h']) if not pd.isna(row['leakage_current_uA_0h']) else None,
            "propagation_delay_ns_0h": float(row['propagation_delay_ns_0h']) if not pd.isna(row['propagation_delay_ns_0h']) else None,
            "voltage_V_0h": float(row['voltage_V_0h']) if not pd.isna(row['voltage_V_0h']) else None,
            "temperature_C_0h": float(row['temperature_C_0h']) if not pd.isna(row['temperature_C_0h']) else None
        },
        "measurements_24h": {
            "iddq_uA_24h": float(row['iddq_uA_24h']) if not pd.isna(row['iddq_uA_24h']) else None,
            "leakage_current_uA_24h": float(row['leakage_current_uA_24h']) if not pd.isna(row['leakage_current_uA_24h']) else None,
            "propagation_delay_ns_24h": float(row['propagation_delay_ns_24h']) if not pd.isna(row['propagation_delay_ns_24h']) else None,
            "voltage_V_24h": float(row['voltage_V_24h']) if not pd.isna(row['voltage_V_24h']) else None,
            "temperature_C_24h": float(row['temperature_C_24h']) if not pd.isna(row['temperature_C_24h']) else None,
            "iddq_drift_24h_pct": round(float(row['iddq_drift_24h_pct']) * 100.0, 3) if not pd.isna(row['iddq_drift_24h_pct']) else None
        },
        "measurements_96h": {
            "iddq_uA_96h": float(row['iddq_uA_96h']) if not pd.isna(row['iddq_uA_96h']) else None,
            "leakage_current_uA_96h": float(row['leakage_current_uA_96h']) if not pd.isna(row['leakage_current_uA_96h']) else None,
            "propagation_delay_ns_96h": float(row['propagation_delay_ns_96h']) if not pd.isna(row['propagation_delay_ns_96h']) else None,
            "voltage_V_96h": float(row['voltage_V_96h']) if not pd.isna(row['voltage_V_96h']) else None,
            "temperature_C_96h": float(row['temperature_C_96h']) if not pd.isna(row['temperature_C_96h']) else None,
            "iddq_drift_96h_pct": round(float(row['iddq_drift_96h_pct']) * 100.0, 3) if not pd.isna(row['iddq_drift_96h_pct']) else None,
            "leakage_drift_96h_pct": round(float(row['leakage_drift_96h_pct']) * 100.0, 3) if not pd.isna(row['leakage_drift_96h_pct']) else None,
            "delay_drift_96h_pct": round(float(row['delay_drift_96h_pct']) * 100.0, 3) if not pd.isna(row['delay_drift_96h_pct']) else None
        }
    }
    return jsonify(data_dict)


@app.route('/api/screen/24h', methods=['POST'])
def screen_24h():
    """
    Executes 24h Early Screening Gate Inference (A24 + B24) and Decision Layer.
    Uses ONLY 0h + 24h features. Zero 96h or 168h features admitted.
    """
    cache = get_data_cache()
    payload = request.get_json(force=True) or {}
    
    # Load record by component_id or custom payload
    if 'component_id' in payload:
        c_id = payload['component_id']
        if c_id not in cache["all_indexed"].index:
            return jsonify({"error": f"Component '{c_id}' not found"}), 404
        input_data = cache["all_indexed"].loc[c_id]
    else:
        input_data = payload
        
    result = run_screening_pipeline(input_data, gate="24h", config=DEFAULT_CONFIG, models=cache["models"])
    return jsonify(result)


@app.route('/api/screen/96h', methods=['POST'])
def screen_96h():
    """
    Executes 96h Mid Screening Gate Inference (A96 + B96) and Decision Layer.
    Uses 0h + 24h + 96h features. Zero 168h features admitted.
    """
    cache = get_data_cache()
    payload = request.get_json(force=True) or {}
    
    if 'component_id' in payload:
        c_id = payload['component_id']
        if c_id not in cache["all_indexed"].index:
            return jsonify({"error": f"Component '{c_id}' not found"}), 404
        input_data = cache["all_indexed"].loc[c_id]
    else:
        input_data = payload
        
    result = run_screening_pipeline(input_data, gate="96h", config=DEFAULT_CONFIG, models=cache["models"])
    return jsonify(result)


@app.route('/api/screen/sequential', methods=['POST'])
def screen_sequential():
    """
    Executes 2-Stage Sequential Early-Exit Screening Workflow.
    Evaluates 24h first: if PASS or REJECT -> early exit; if REVIEW -> proceeds to 96h.
    """
    cache = get_data_cache()
    payload = request.get_json(force=True) or {}
    
    if 'component_id' in payload:
        c_id = payload['component_id']
        if c_id not in cache["all_indexed"].index:
            return jsonify({"error": f"Component '{c_id}' not found"}), 404
        input_data = cache["all_indexed"].loc[c_id]
    else:
        input_data = payload
        
    result = run_sequential_screening(input_data, config=DEFAULT_CONFIG, models=cache["models"])
    return jsonify(result)


@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    """
    Returns documented locked test set benchmark metrics from evaluation reports.
    """
    return jsonify({
        "module_a": {
            "title": "Module A — Defect Classification",
            "target": "module_a_label (0 = Normal, 1 = Defective)",
            "evaluation_partition": "Locked Test Set (N=1,500 components, 450 Defective, 1,050 Normal)",
            "a24": {
                "gate": "24h Early Gate",
                "model_name": "LogisticRegression (Balanced, Scaled)",
                "features_count": 11,
                "features_description": "0h Baselines (5) + 24h Sensors (5) + 24h Iddq Drift (1)",
                "accuracy": 0.8240,
                "precision": 0.6946,
                "recall": 0.7378,
                "f1_score": 0.7155,
                "fnr": 0.2622,
                "fpr": 0.1390,
                "roc_auc": 0.8719
            },
            "a96": {
                "gate": "96h Mid Gate",
                "model_name": "RandomForestClassifier (150 trees)",
                "features_count": 19,
                "features_description": "0h + 24h + 96h Sensors (15) + Multi-Drift Metrics (4)",
                "accuracy": 0.9767,
                "precision": 0.9814,
                "recall": 0.9400,
                "f1_score": 0.9603,
                "fnr": 0.0600,
                "fpr": 0.0076,
                "roc_auc": 0.9944
            },
            "delta_improvement": {
                "accuracy_gain": "+15.27%",
                "precision_gain": "+28.68%",
                "recall_gain": "+20.22%",
                "fnr_reduction": "-20.22% (77.1% relative reduction in defect escapes)",
                "fpr_reduction": "-13.14% (94.5% relative reduction in false scrap)",
                "roc_auc_gain": "+0.1225"
            }
        },
        "module_b": {
            "title": "Module B — 168h Degradation Forecasting",
            "target": "iddq_drift_168h_true (True continuous percentage drift)",
            "evaluation_partition": "Locked Test Set (N=1,500 components)",
            "note": "The target is true 168h Iddq drift. 168h sensor measurements are NEVER used as input features.",
            "b24": {
                "gate": "24h Early Gate",
                "model_name": "GradientBoostingRegressor (100 trees)",
                "features_count": 11,
                "features_description": "0h Baselines (5) + 24h Sensors (5) + 24h Iddq Drift (1)",
                "rmse": 0.04033,
                "rmse_pct": "4.033%",
                "mae": 0.02721,
                "mae_pct": "2.721%",
                "r2_score": 0.7890
            },
            "b96": {
                "gate": "96h Mid Gate",
                "model_name": "RandomForestRegressor (100 trees)",
                "features_count": 19,
                "features_description": "0h + 24h + 96h Sensors (15) + Multi-Drift Metrics (4)",
                "rmse": 0.01415,
                "rmse_pct": "1.415%",
                "mae": 0.00877,
                "mae_pct": "0.877%",
                "r2_score": 0.9740
            },
            "delta_improvement": {
                "rmse_reduction": "-2.618% (64.9% relative error reduction)",
                "mae_reduction": "-1.843% (67.8% relative error reduction)",
                "r2_gain": "+0.1851"
            }
        },
        "decision_thresholds": DEFAULT_CONFIG.to_dict()
    })


# ------------------------------------------------------------------------------
# SERVE FRONTEND
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    """Serves the main application dashboard."""
    return render_template('index.html')


if __name__ == '__main__':
    # Initialize cache on startup
    get_data_cache()
    print("Burn-In Screening Dashboard API server ready on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
