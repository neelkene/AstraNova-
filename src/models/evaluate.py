"""
Module: src.models.evaluate
Purpose: Comprehensive evaluation metrics for Module A (Classification) and Module B (Regression).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics for Module A.
    Includes False Negative Rate (missed defects) and False Positive Rate (false alarms).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate_fnr": float(fnr),
        "false_positive_rate_fpr": float(fpr),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
        
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes regression metrics for Module B (168h drift prediction).
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mean_actual_drift": float(np.mean(y_true)),
        "mean_predicted_drift": float(np.mean(y_pred))
    }


def format_metrics_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats multi-model or multi-gate experiment results into a clean DataFrame.
    """
    return pd.DataFrame.from_dict(results, orient='index')
