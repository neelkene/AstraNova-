"""
Module: backend.services.model_service
Purpose: Singleton model registry — loads all 4 trained .joblib pipelines exactly once
         at FastAPI startup and provides accessor methods.

Loading strategy:
  - Delegates to src.inference.predict.load_models() which already handles
    FileNotFoundError with a clear message.
  - Models are cached in the module-level _registry dict.
  - Startup failure terminates the application with a descriptive error.

Feature importance extraction:
  - Supported for tree-based models (RandomForest*, GradientBoosting*) via .feature_importances_
  - Logistic Regression exposes .coef_ but is treated as unsupported for importance
    (multinomial coefficients are not comparable feature importances in the tree sense).
  - When unsupported, returns an empty list — never fabricates scores.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Ensure workspace root is importable
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from src.inference.predict import load_models, MODEL_PATHS  # noqa: E402
from backend.schemas import FeatureImportanceItem  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------
_registry: Dict[str, Any] = {}


def initialize_models() -> None:
    """
    Load all four production models from disk.
    Called once during FastAPI lifespan startup.
    Raises FileNotFoundError with a descriptive message if any model is absent.
    """
    global _registry

    # Verify model files exist before delegating to src layer
    missing = [
        path for path in MODEL_PATHS.values()
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Required model artifact(s) not found — cannot start the API server.\n"
            + "\n".join(f"  MISSING: {p}" for p in missing)
        )

    _registry = load_models(force_reload=True)
    logger.info(
        "ModelRegistry: loaded %d models: %s",
        len(_registry),
        list(_registry.keys()),
    )


def get_models() -> Dict[str, Any]:
    """Return the cached model registry dict. Raises RuntimeError if not initialised."""
    if not _registry:
        raise RuntimeError(
            "Model registry is empty. Ensure initialize_models() was called during startup."
        )
    return _registry


def is_loaded() -> bool:
    return bool(_registry)


def module_a_loaded() -> bool:
    return "a24" in _registry and "a96" in _registry


def module_b_loaded() -> bool:
    return "b24" in _registry and "b96" in _registry


def model_load_status() -> Dict[str, bool]:
    return {key: key in _registry for key in ("a24", "a96", "b24", "b96")}


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def get_feature_importances(
    pipeline: Any,
    feature_names: List[str],
    step_name: str = "classifier",
    top_n: int = 10,
) -> List[FeatureImportanceItem]:
    """
    Extracts and ranks feature importances from a scikit-learn Pipeline.

    Supported:
      - Tree-based models with `.feature_importances_` (RandomForest*, GradientBoosting*)

    Not supported (returns empty list):
      - LogisticRegression (`.coef_` magnitude is not a reliable importance signal
        for the frontend without additional context)
      - Any step that doesn't expose a recognised importance attribute

    IMPORTANT: Importance scores describe which input features the model weighted
    most heavily in its predictions. They do NOT imply physical causation.
    """
    try:
        step = pipeline.named_steps.get(step_name)
        if step is None:
            # Try common alternative step names
            for alt in ("regressor", "estimator"):
                step = pipeline.named_steps.get(alt)
                if step is not None:
                    break

        if step is None or not hasattr(step, "feature_importances_"):
            return []

        importances: np.ndarray = step.feature_importances_
        if len(importances) != len(feature_names):
            logger.warning(
                "Feature importance length %d != feature_names length %d; skipping.",
                len(importances),
                len(feature_names),
            )
            return []

        sorted_idx = np.argsort(importances)[::-1]
        top_idx = sorted_idx[:top_n]

        return [
            FeatureImportanceItem(
                feature=feature_names[i],
                importance=round(float(importances[i]), 6),
            )
            for i in top_idx
            if importances[i] > 0.0
        ]

    except Exception as exc:  # pragma: no cover
        logger.warning("Could not extract feature importances: %s", exc)
        return []
