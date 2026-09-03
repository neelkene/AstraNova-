"""
Module: backend.services.component_service
Purpose: Dataset-backed component lookup service.

Critical design constraints:
  1. Ground-truth columns (module_a_label, iddq_drift_168h_true, component_type)
     are NEVER returned through any public method of this service.
  2. The component_id column is available for display but is NOT an inference feature.
  3. Drift columns stored in the dataset (iddq_drift_24h_pct etc.) are available
     as raw row data — these are pre-computed during data preparation and match
     the values the models were trained on.
  4. The service is initialised once at startup; component data is read-only.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

from src.data.load_data import load_ml_ready_data  # noqa
from src.data.split_data import split_components  # noqa

# Ground-truth columns that must NEVER appear in API responses during normal inference
_GROUND_TRUTH_COLS = frozenset({
    "module_a_label",
    "iddq_drift_168h_true",
    "component_type",       # not in schema but guard anyway
})

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_all_df: Optional[pd.DataFrame] = None
_test_ids: Optional[frozenset] = None
_all_indexed: Optional[pd.DataFrame] = None


def initialize_dataset() -> None:
    """
    Load the ML-ready dataset and create look-up indices.
    Called once during FastAPI lifespan startup.
    """
    global _all_df, _test_ids, _all_indexed

    df = load_ml_ready_data()
    _, _, test_df = split_components(df, random_state=42)

    _all_df = df
    _test_ids = frozenset(test_df["component_id"].tolist())
    _all_indexed = df.set_index("component_id", drop=False)

    logger.info(
        "ComponentService: loaded %d components (%d in locked test set)",
        len(df),
        len(_test_ids),
    )


def _check_initialised() -> None:
    if _all_indexed is None:
        raise RuntimeError(
            "ComponentService is not initialised. Ensure initialize_dataset() was called at startup."
        )


# ---------------------------------------------------------------------------
# Public: get_component_row
# ---------------------------------------------------------------------------

def get_component_row(component_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns a flat dict of all sensor readings for the given component ID.

    Returns None if the component is not found.

    Ground-truth columns are STRIPPED before returning.
    The returned dict is safe to pass to prediction_service and to expose via API.
    """
    _check_initialised()

    if component_id not in _all_indexed.index:
        return None

    row: pd.Series = _all_indexed.loc[component_id]
    result: Dict[str, Any] = {}

    for col, val in row.items():
        if col in _GROUND_TRUTH_COLS:
            continue
        if isinstance(val, float) and math.isnan(val):
            result[col] = None
        else:
            result[col] = val

    return result


# ---------------------------------------------------------------------------
# Public: get_component_lookup_payload
# ---------------------------------------------------------------------------

def get_component_lookup_payload(component_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns a structured lookup payload for the GET /api/components/{id} endpoint.
    Splits measurements into time-point blocks for frontend consumption.
    Ground-truth labels are NEVER included.
    """
    _check_initialised()

    if component_id not in _all_indexed.index:
        return None

    row: pd.Series = _all_indexed.loc[component_id]
    is_test = component_id in _test_ids

    def safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    def pct_round(v: Any) -> Optional[float]:
        """Convert raw fractional drift to percentage for display."""
        sf = safe_float(v)
        return None if sf is None else round(sf * 100.0, 3)

    return {
        "component_id": component_id,
        "is_in_locked_test_set": is_test,
        "measurements_0h": {
            "iddq_uA_0h":               safe_float(row.get("iddq_uA_0h")),
            "leakage_current_uA_0h":    safe_float(row.get("leakage_current_uA_0h")),
            "propagation_delay_ns_0h":  safe_float(row.get("propagation_delay_ns_0h")),
            "voltage_V_0h":             safe_float(row.get("voltage_V_0h")),
            "temperature_C_0h":         safe_float(row.get("temperature_C_0h")),
        },
        "measurements_24h": {
            "iddq_uA_24h":              safe_float(row.get("iddq_uA_24h")),
            "leakage_current_uA_24h":   safe_float(row.get("leakage_current_uA_24h")),
            "propagation_delay_ns_24h": safe_float(row.get("propagation_delay_ns_24h")),
            "voltage_V_24h":            safe_float(row.get("voltage_V_24h")),
            "temperature_C_24h":        safe_float(row.get("temperature_C_24h")),
            "iddq_drift_24h_pct":       pct_round(row.get("iddq_drift_24h_pct")),
        },
        "measurements_96h": {
            "iddq_uA_96h":              safe_float(row.get("iddq_uA_96h")),
            "leakage_current_uA_96h":   safe_float(row.get("leakage_current_uA_96h")),
            "propagation_delay_ns_96h": safe_float(row.get("propagation_delay_ns_96h")),
            "voltage_V_96h":            safe_float(row.get("voltage_V_96h")),
            "temperature_C_96h":        safe_float(row.get("temperature_C_96h")),
            "iddq_drift_96h_pct":       pct_round(row.get("iddq_drift_96h_pct")),
            "leakage_drift_96h_pct":    pct_round(row.get("leakage_drift_96h_pct")),
            "delay_drift_96h_pct":      pct_round(row.get("delay_drift_96h_pct")),
        },
        "measurements_168h": {
            "iddq_uA_168h":              safe_float(row.get("iddq_uA_168h")),
            "leakage_current_uA_168h":   safe_float(row.get("leakage_current_uA_168h")),
            "propagation_delay_ns_168h": safe_float(row.get("propagation_delay_ns_168h")),
            "voltage_V_168h":            safe_float(row.get("voltage_V_168h")),
            "temperature_C_168h":        safe_float(row.get("temperature_C_168h")),
        } if any(row.get(c) is not None for c in ["iddq_uA_168h", "leakage_current_uA_168h"]) else None,
    }


# ---------------------------------------------------------------------------
# Public: get_dataset_overview
# ---------------------------------------------------------------------------

def get_dataset_overview() -> Dict[str, Any]:
    """
    Computes aggregated dataset statistics from the ground-truth dataset
    for the overview dashboard section.
    Individual component labels are never exposed here.
    """
    _check_initialised()
    gt_path = os.path.join(_WORKSPACE_DIR, 'data', 'ground_truth', 'component_ground_truth.csv')
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        counts = gt_df['component_type'].value_counts()
        total = int(len(gt_df))
        normal = int(counts.get('normal', 0))
        drifting = int(counts.get('drifting', 0))
        anomalous = int(counts.get('anomalous', 0))
    else:
        # Fallback from loaded data
        assert _all_df is not None
        total = int(len(_all_df))
        normal = int((_all_df['module_a_label'] == 0).sum())
        drifting = 0
        anomalous = int((_all_df['module_a_label'] == 1).sum())

    return {
        "total_components": total,
        "normal_count": normal,
        "normal_pct": round((normal / total) * 100.0, 1) if total > 0 else 0.0,
        "drifting_count": drifting,
        "drifting_pct": round((drifting / total) * 100.0, 1) if total > 0 else 0.0,
        "anomalous_count": anomalous,
        "anomalous_pct": round((anomalous / total) * 100.0, 1) if total > 0 else 0.0,
        "burnin_gates": ["0h (Pre-Burn-In Baseline)", "24h (Early Screening Gate)", "96h (Mid Qualification Gate)", "168h (Final Qualification Benchmark)"],
    }



# ---------------------------------------------------------------------------
# Public: list_components
# ---------------------------------------------------------------------------

def list_components(
    split: str = "all",
    limit: int = 100,
    offset: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Returns a paginated list of component IDs with basic IDDQ readings.

    Parameters
    ----------
    split  : 'all' | 'test' — filter to the locked test partition
    limit  : max rows to return
    offset : row offset for pagination

    Returns
    -------
    (total_available, list_of_items)

    Ground-truth labels are NEVER returned.
    """
    _check_initialised()

    assert _all_df is not None

    if split == "test":
        df_filtered = _all_df[_all_df["component_id"].isin(_test_ids)]
    else:
        df_filtered = _all_df

    total = len(df_filtered)
    page = df_filtered.iloc[offset: offset + limit]

    def _sf(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 2)
        except (TypeError, ValueError):
            return None

    items = [
        {
            "component_id": row["component_id"],
            "iddq_uA_0h":   _sf(row.get("iddq_uA_0h")),
            "iddq_uA_24h":  _sf(row.get("iddq_uA_24h")),
            "iddq_uA_96h":  _sf(row.get("iddq_uA_96h")),
        }
        for _, row in page.iterrows()
    ]

    return total, items


# ---------------------------------------------------------------------------
# Public: component_exists
# ---------------------------------------------------------------------------

def component_exists(component_id: str) -> bool:
    _check_initialised()
    return component_id in _all_indexed.index
