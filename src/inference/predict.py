"""
Module: src.inference.predict
Purpose: Production inference layer combining Module A (Classification) and Module B (Regression)
         for early burn-in screening at 24h and 96h operational test gates.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import joblib

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.features.build_features import (
    FEATURES_24H_GATE,
    FEATURES_96H_GATE,
    FEATURES_168H_FORBIDDEN,
)

MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')

# Standard Serialized Model File Paths
MODEL_PATHS = {
    "a24": os.path.join(MODELS_DIR, "module_a_24h_logisticregression.joblib"),
    "a96": os.path.join(MODELS_DIR, "module_a_96h_randomforest.joblib"),
    "b24": os.path.join(MODELS_DIR, "module_b_24h_gradientboostingregressor.joblib"),
    "b96": os.path.join(MODELS_DIR, "module_b_96h_randomforestregressor.joblib"),
}

# Module-level model cache to avoid repeated disk reads
_MODEL_CACHE: Dict[str, Any] = {}


def load_models(models_dir: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads and caches the 4 serialized production models:
    - a24: Module A 24h Classifier (Logistic Regression)
    - a96: Module A 96h Classifier (Random Forest)
    - b24: Module B 24h Regressor (Gradient Boosting Regressor)
    - b96: Module B 96h Regressor (Random Forest Regressor)
    """
    global _MODEL_CACHE
    if _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE

    target_dir = models_dir or MODELS_DIR
    loaded = {}
    for key, default_path in MODEL_PATHS.items():
        fname = os.path.basename(default_path)
        full_path = os.path.join(target_dir, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Required model artifact not found: {full_path}")
        loaded[key] = joblib.load(full_path)

    _MODEL_CACHE = loaded
    return _MODEL_CACHE


def prepare_inference_features(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    gate: str = "24h"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts input record(s) into a strictly validated feature DataFrame for inference.
    Automatically computes any missing percentage drift features from raw sensor readings.
    Enforces strict temporal isolation and checks against forbidden future features.
    """
    gate_clean = gate.lower().strip()
    if gate_clean not in ["24h", "24", "a24", "b24", "96h", "96", "a96", "b96"]:
        raise ValueError(f"Unknown screening gate '{gate}'. Expected '24h' or '96h'.")

    is_24h = "24" in gate_clean
    required_features = list(FEATURES_24H_GATE if is_24h else FEATURES_96H_GATE)

    # 1. Standardize to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data.to_dict()])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected DataFrame, Series, or dict.")

    # 2. Automatically compute drift metrics if raw channels are provided but drift column is missing
    if "iddq_drift_24h_pct" not in df.columns and "iddq_uA_24h" in df.columns and "iddq_uA_0h" in df.columns:
        df["iddq_drift_24h_pct"] = (df["iddq_uA_24h"] - df["iddq_uA_0h"]) / df["iddq_uA_0h"]

    if not is_24h:
        if "iddq_drift_96h_pct" not in df.columns and "iddq_uA_96h" in df.columns and "iddq_uA_0h" in df.columns:
            df["iddq_drift_96h_pct"] = (df["iddq_uA_96h"] - df["iddq_uA_0h"]) / df["iddq_uA_0h"]
        if "leakage_drift_96h_pct" not in df.columns and "leakage_current_uA_96h" in df.columns and "leakage_current_uA_0h" in df.columns:
            df["leakage_drift_96h_pct"] = (df["leakage_current_uA_96h"] - df["leakage_current_uA_0h"]) / df["leakage_current_uA_0h"]
        if "delay_drift_96h_pct" not in df.columns and "propagation_delay_ns_96h" in df.columns and "propagation_delay_ns_0h" in df.columns:
            df["delay_drift_96h_pct"] = (df["propagation_delay_ns_96h"] - df["propagation_delay_ns_0h"]) / df["propagation_delay_ns_0h"]

    # 3. Check for missing required features
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required {gate} feature(s): {missing}")

    # 4. Strict Temporal Integrity Checks
    for f in FEATURES_168H_FORBIDDEN:
        if f in df.columns and f in required_features:
            raise ValueError(f"FATAL: 168h end-of-test feature '{f}' cannot enter early inference feature matrix!")

    if is_24h:
        for f in ["iddq_uA_96h", "leakage_current_uA_96h", "propagation_delay_ns_96h", "voltage_V_96h", "temperature_C_96h", "iddq_drift_96h_pct", "leakage_drift_96h_pct", "delay_drift_96h_pct"]:
            if f in required_features:
                raise ValueError(f"FATAL: 96h feature '{f}' cannot enter 24h gate inference matrix!")

    X = df[required_features].copy()
    return X, required_features


def predict_24h(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    models: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs 24h screening inference (Module A A24 Classifier + Module B B24 Regressor).
    
    Inputs:
      0h baselines + 24h measurements + 24h drift (11 features)
      
    Outputs per component:
      - defect_probability: float (Class 1 defect probability)
      - predicted_class: int (0 = Normal, 1 = Defective)
      - predicted_168h_iddq_drift: float (raw fractional 168h degradation)
      - predicted_168h_iddq_drift_pct: float (percentage 168h degradation)
      - screening_gate: "24h"
      - model_a_name: "LogisticRegression"
      - model_b_name: "GradientBoostingRegressor"
    """
    model_suite = models or load_models()
    clf_a24 = model_suite["a24"]
    reg_b24 = model_suite["b24"]

    X, feature_names = prepare_inference_features(data, gate="24h")

    # Module A: Classification
    probs = clf_a24.predict_proba(X)[:, 1] if hasattr(clf_a24, "predict_proba") else clf_a24.predict(X).astype(float)
    preds_cls = (probs >= 0.50).astype(int)

    # Module B: 168h Drift Regression
    preds_drift_raw = reg_b24.predict(X)

    results = []
    for i in range(len(X)):
        prob_val = float(probs[i])
        cls_val = int(preds_cls[i])
        drift_raw = float(preds_drift_raw[i])
        drift_pct = float(drift_raw * 100.0)

        res = {
            "defect_probability": round(prob_val, 4),
            "predicted_class": cls_val,
            "predicted_168h_iddq_drift": round(drift_raw, 6),
            "predicted_168h_iddq_drift_pct": round(drift_pct, 3),
            "screening_gate": "24h",
            "model_a_name": "LogisticRegression",
            "model_b_name": "GradientBoostingRegressor",
            "num_features_used": len(feature_names),
        }
        results.append(res)

    return results[0] if (isinstance(data, (dict, pd.Series)) or len(results) == 1) else results


def predict_96h(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    models: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs 96h screening inference (Module A A96 Classifier + Module B B96 Regressor).
    
    Inputs:
      0h baselines + 24h measurements + 96h measurements + multi-drift metrics (19 features)
      
    Outputs per component:
      - defect_probability: float (Class 1 defect probability)
      - predicted_class: int (0 = Normal, 1 = Defective)
      - predicted_168h_iddq_drift: float (raw fractional 168h degradation)
      - predicted_168h_iddq_drift_pct: float (percentage 168h degradation)
      - screening_gate: "96h"
      - model_a_name: "RandomForestClassifier"
      - model_b_name: "RandomForestRegressor"
    """
    model_suite = models or load_models()
    clf_a96 = model_suite["a96"]
    reg_b96 = model_suite["b96"]

    X, feature_names = prepare_inference_features(data, gate="96h")

    # Module A: Classification
    probs = clf_a96.predict_proba(X)[:, 1] if hasattr(clf_a96, "predict_proba") else clf_a96.predict(X).astype(float)
    preds_cls = (probs >= 0.50).astype(int)

    # Module B: 168h Drift Regression
    preds_drift_raw = reg_b96.predict(X)

    results = []
    for i in range(len(X)):
        prob_val = float(probs[i])
        cls_val = int(preds_cls[i])
        drift_raw = float(preds_drift_raw[i])
        drift_pct = float(drift_raw * 100.0)

        res = {
            "defect_probability": round(prob_val, 4),
            "predicted_class": cls_val,
            "predicted_168h_iddq_drift": round(drift_raw, 6),
            "predicted_168h_iddq_drift_pct": round(drift_pct, 3),
            "screening_gate": "96h",
            "model_a_name": "RandomForestClassifier",
            "model_b_name": "RandomForestRegressor",
            "num_features_used": len(feature_names),
        }
        results.append(res)

    return results[0] if (isinstance(data, (dict, pd.Series)) or len(results) == 1) else results


def run_inference_gate(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    gate: str = "24h",
    models: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Unified entry point for running screening inference at a specified test gate ('24h' or '96h').
    """
    gate_clean = gate.lower().strip()
    if "24" in gate_clean:
        return predict_24h(data, models=models)
    elif "96" in gate_clean:
        return predict_96h(data, models=models)
    else:
        raise ValueError(f"Invalid gate '{gate}'. Must be '24h' or '96h'.")


if __name__ == "__main__":
    from src.data.load_data import load_ml_ready_data
    df_sample = load_ml_ready_data().head(3)
    
    print("=== Testing 24h Gate Inference ===")
    res_24 = predict_24h(df_sample.iloc[0])
    for k, v in res_24.items():
        print(f"  {k}: {v}")
        
    print("\n=== Testing 96h Gate Inference ===")
    res_96 = predict_96h(df_sample.iloc[0])
    for k, v in res_96.items():
        print(f"  {k}: {v}")
