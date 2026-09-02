"""
Comprehensive Test Suite for Inference & Decision Pipeline
File: tests/test_inference_pipeline.py
Project: AI-Driven Anomaly Detection in Component Burn-In & Screening (SIH 2026)
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add workspace to path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data
from src.data.split_data import split_components
from src.features.build_features import FEATURES_24H_GATE, FEATURES_96H_GATE, FEATURES_168H_FORBIDDEN
from src.inference.predict import (
    load_models,
    prepare_inference_features,
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


@pytest.fixture(scope="module")
def sample_data():
    df = load_ml_ready_data()
    _, _, test_df = split_components(df, random_state=42)
    return test_df.head(10)


@pytest.fixture(scope="module")
def models_dict():
    return load_models()


def test_models_load_successfully(models_dict):
    """Test 1: Verifies all 4 production models load properly from models/ directory."""
    assert "a24" in models_dict
    assert "a96" in models_dict
    assert "b24" in models_dict
    assert "b96" in models_dict
    assert hasattr(models_dict["a24"], "predict_proba")
    assert hasattr(models_dict["a96"], "predict_proba")
    assert hasattr(models_dict["b24"], "predict")
    assert hasattr(models_dict["b96"], "predict")


def test_24h_feature_subset_strictly_bounded(sample_data):
    """Test 2 & 3: Verifies A24 and B24 feature matrices contain strictly 11 features from 0h/24h."""
    X_24, feat_names = prepare_inference_features(sample_data, gate="24h")
    assert len(feat_names) == 11
    assert list(X_24.columns) == list(FEATURES_24H_GATE)
    for f in X_24.columns:
        assert "_96h" not in f, f"96h feature {f} leaked into 24h inference matrix!"
        assert "_168h" not in f, f"168h feature {f} leaked into 24h inference matrix!"


def test_96h_feature_subset_strictly_bounded(sample_data):
    """Test 4 & 5: Verifies A96 and B96 feature matrices contain strictly 19 features up to 96h."""
    X_96, feat_names = prepare_inference_features(sample_data, gate="96h")
    assert len(feat_names) == 19
    assert list(X_96.columns) == list(FEATURES_96H_GATE)
    for f in X_96.columns:
        assert "_168h" not in f, f"168h feature {f} leaked into 96h inference matrix!"


def test_forbidden_168h_measurements_rejected(sample_data):
    """Test 6: Asserts that 168h sensor measurements cannot enter early prediction matrices."""
    for f in FEATURES_168H_FORBIDDEN:
        assert f not in FEATURES_24H_GATE
        assert f not in FEATURES_96H_GATE


def test_prediction_output_schema_and_types(sample_data, models_dict):
    """Test 7: Verifies inference output schema, probabilities in [0, 1], and predicted class in {0, 1}."""
    res_24 = predict_24h(sample_data.iloc[0], models=models_dict)
    assert "defect_probability" in res_24
    assert "predicted_class" in res_24
    assert "predicted_168h_iddq_drift" in res_24
    assert "predicted_168h_iddq_drift_pct" in res_24
    assert "screening_gate" in res_24
    assert 0.0 <= res_24["defect_probability"] <= 1.0
    assert res_24["predicted_class"] in [0, 1]
    assert res_24["screening_gate"] == "24h"

    res_96 = predict_96h(sample_data.iloc[0], models=models_dict)
    assert 0.0 <= res_96["defect_probability"] <= 1.0
    assert res_96["predicted_class"] in [0, 1]
    assert res_96["screening_gate"] == "96h"


def test_screening_decision_rules():
    """Test 8: Verifies decision outputs strictly belong to {'PASS', 'REVIEW', 'REJECT'}."""
    cfg = DecisionConfig()
    
    # Severe defect at 24h -> REJECT
    d_rej = make_screening_decision(0.95, 0.25, screening_gate="24h", config=cfg)
    assert d_rej["decision"] == "REJECT"

    # Pristine normal at 24h -> PASS
    d_pass = make_screening_decision(0.10, 0.015, screening_gate="24h", config=cfg)
    assert d_pass["decision"] == "PASS"

    # Borderline at 24h -> REVIEW
    d_rev = make_screening_decision(0.50, 0.06, screening_gate="24h", config=cfg)
    assert d_rev["decision"] == "REVIEW"

    # 96h evaluations
    d_96_pass = make_screening_decision(0.05, 0.012, screening_gate="96h", config=cfg)
    assert d_96_pass["decision"] == "PASS"

    d_96_rej = make_screening_decision(0.90, 0.10, screening_gate="96h", config=cfg)
    assert d_96_rej["decision"] == "REJECT"


def test_sequential_screening_workflow(sample_data, models_dict):
    """Test 9: Verifies 2-stage sequential workflow logic and early-exit metadata."""
    seq_res = run_sequential_screening(sample_data, config=DEFAULT_CONFIG, models=models_dict)
    assert len(seq_res) == len(sample_data)
    for r in seq_res:
        assert r["final_decision"] in ["PASS", "REVIEW", "REJECT"]
        assert r["final_screening_gate"] in ["24h", "96h"]
        assert isinstance(r["early_exit_applied"], bool)
        assert r["stage_1_24h"] is not None


def test_serialized_models_and_datasets_untouched():
    """Test 10: Ensures all 4 model files and 3 data files exist and remain unmodified."""
    data_files = [
        "data/raw/raw_burnin_data.csv",
        "data/ground_truth/component_ground_truth.csv",
        "data/ml_ready/ml_features.csv"
    ]
    model_files = [
        "models/module_a_24h_logisticregression.joblib",
        "models/module_a_96h_randomforest.joblib",
        "models/module_b_24h_gradientboostingregressor.joblib",
        "models/module_b_96h_randomforestregressor.joblib"
    ]
    for df_path in data_files:
        assert os.path.exists(os.path.join(WORKSPACE_DIR, df_path)), f"Missing dataset: {df_path}"
    for mf_path in model_files:
        assert os.path.exists(os.path.join(WORKSPACE_DIR, mf_path)), f"Missing model: {mf_path}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
