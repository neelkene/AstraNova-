"""
Module: src.models.train_module_b
Purpose: Baseline model definitions and training harness for Module B (168h Degradation Prediction).
Experiments: B24 (24h gate) and B96 (96h gate).
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.features.build_features import build_preprocessing_pipeline, extract_features_and_target
from src.models.evaluate import evaluate_regression


def get_module_b_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the standard baseline model suite for Module B regression:
    1. Linear Regression (with scaling)
    2. Random Forest Regressor
    3. Gradient Boosting Regressor
    """
    return {
        "LinearRegression": {
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
