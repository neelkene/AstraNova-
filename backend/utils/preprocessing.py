"""
Module: backend.utils.preprocessing
Purpose: Request-to-inference row assembly and human-friendly measurement summaries.

Key design principles:
  1. Drift features are computed IDENTICALLY to the training pipeline logic defined in
     src/inference/predict.py:prepare_inference_features().  We do NOT re-fit any
     imputer here — the imputer is already baked into the saved sklearn Pipeline objects.
  2. Ground-truth columns (module_a_label, iddq_drift_168h_true, component_type) are
     stripped before any data leaves this layer.
  3. No statistics are computed from inference-time data for imputation.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional

import pandas as pd

from backend.schemas import (
    Measurements0h,
    Measurements24h,
    Measurements96h,
    ObservedChange,
)

# ---------------------------------------------------------------------------
# Ground-truth / identifier columns that must never leave the backend
# ---------------------------------------------------------------------------
_FORBIDDEN_OUTPUT_COLS = frozenset({
    "module_a_label",
    "iddq_drift_168h_true",
    "component_type",
})

# Columns that are pre-computed drift features stored in the dataset
# (they will be re-computed on-the-fly from raw sensor values for raw requests)
_DATASET_DRIFT_COLS = frozenset({
    "iddq_drift_24h_pct",
    "iddq_drift_96h_pct",
    "leakage_drift_96h_pct",
    "delay_drift_96h_pct",
})


# ---------------------------------------------------------------------------
# Public: build_inference_row
# ---------------------------------------------------------------------------

def build_inference_row(
    m0h: Measurements0h,
    m24h: Measurements24h,
    m96h: Optional[Measurements96h] = None,
) -> Dict[str, Any]:
    """
    Assembles a flat dictionary from structured Pydantic sensor objects.
    Computes percentage drift features using the same formulae as
    ``src/inference/predict.py:prepare_inference_features()``.

    The result is a dict that can be passed directly to
    ``src.inference.predict.prepare_inference_features(data, gate=...)``.

    NOTE: The model pipelines contain a fitted SimpleImputer that handles
    any remaining NaN values.  We do NOT fit a new imputer here.
    """
    row: Dict[str, Any] = {}

    # 0h baselines
    row["iddq_uA_0h"] = m0h.iddq_uA_0h
    row["leakage_current_uA_0h"] = m0h.leakage_current_uA_0h
    row["propagation_delay_ns_0h"] = m0h.propagation_delay_ns_0h
    row["voltage_V_0h"] = m0h.voltage_V_0h
    row["temperature_C_0h"] = m0h.temperature_C_0h

    # 24h measurements
    row["iddq_uA_24h"] = m24h.iddq_uA_24h
    row["leakage_current_uA_24h"] = m24h.leakage_current_uA_24h
    row["propagation_delay_ns_24h"] = m24h.propagation_delay_ns_24h
    row["voltage_V_24h"] = m24h.voltage_V_24h
    row["temperature_C_24h"] = m24h.temperature_C_24h

    # 24h IDDQ drift — mirrors prepare_inference_features logic exactly
    row["iddq_drift_24h_pct"] = _safe_pct_drift(
        row.get("iddq_uA_24h"), row.get("iddq_uA_0h")
    )

    # 96h measurements (optional)
    if m96h is not None:
        row["iddq_uA_96h"] = m96h.iddq_uA_96h
        row["leakage_current_uA_96h"] = m96h.leakage_current_uA_96h
        row["propagation_delay_ns_96h"] = m96h.propagation_delay_ns_96h
        row["voltage_V_96h"] = m96h.voltage_V_96h
        row["temperature_C_96h"] = m96h.temperature_C_96h

        row["iddq_drift_96h_pct"] = _safe_pct_drift(
            row.get("iddq_uA_96h"), row.get("iddq_uA_0h")
        )
        row["leakage_drift_96h_pct"] = _safe_pct_drift(
            row.get("leakage_current_uA_96h"), row.get("leakage_current_uA_0h")
        )
        row["delay_drift_96h_pct"] = _safe_pct_drift(
            row.get("propagation_delay_ns_96h"), row.get("propagation_delay_ns_0h")
        )

    return row


# ---------------------------------------------------------------------------
# Public: strip_forbidden_columns
# ---------------------------------------------------------------------------

def strip_forbidden_columns(row: pd.Series) -> Dict[str, Any]:
    """
    Returns a dict from a dataset row with all ground-truth and identifier
    columns removed so they cannot leak into API responses.
    """
    result = {}
    for col, val in row.items():
        if col in _FORBIDDEN_OUTPUT_COLS:
            continue
        if isinstance(val, float) and math.isnan(val):
            result[col] = None
        else:
            result[col] = val
    return result


# ---------------------------------------------------------------------------
# Public: compute_observed_changes
# ---------------------------------------------------------------------------

# Parameter metadata: (display_name, unit, 0h_col, 24h_col, 96h_col)
_PARAMETER_META = [
    ("IDDQ",              "μA", "iddq_uA_0h",               "iddq_uA_24h",               "iddq_uA_96h"),
    ("Leakage Current",   "μA", "leakage_current_uA_0h",     "leakage_current_uA_24h",     "leakage_current_uA_96h"),
    ("Propagation Delay", "ns", "propagation_delay_ns_0h",   "propagation_delay_ns_24h",   "propagation_delay_ns_96h"),
    ("Voltage",           "V",  "voltage_V_0h",              "voltage_V_24h",              "voltage_V_96h"),
    ("Temperature",       "°C", "temperature_C_0h",          "temperature_C_24h",          "temperature_C_96h"),
]


def compute_observed_changes(
    row: Dict[str, Any],
    has_96h: bool = False,
) -> List[ObservedChange]:
    """
    Generates human-friendly observed parameter changes between available time-points.

    For each physical parameter we report the change between:
      - 0h → 24h (always, when 24h data available)
      - 0h → 96h (additionally, when 96h data available)

    Wording deliberately avoids attributing physical causation.
    """
    changes: List[ObservedChange] = []

    for display_name, unit, col_0h, col_24h, col_96h in _PARAMETER_META:
        v0 = _safe_float(row.get(col_0h))
        v24 = _safe_float(row.get(col_24h))
        v96 = _safe_float(row.get(col_96h)) if has_96h else None

        # 0h → 24h
        if v24 is not None:
            changes.append(
                ObservedChange(
                    parameter=display_name,
                    unit=unit,
                    start_time="0h",
                    end_time="24h",
                    start_value=_round_or_none(v0),
                    end_value=_round_or_none(v24),
                    absolute_change=_round_or_none(_safe_subtract(v24, v0)),
                    percentage_change=_round_or_none(_safe_pct_change(v24, v0)),
                )
            )

        # 0h → 96h
        if has_96h and v96 is not None:
            changes.append(
                ObservedChange(
                    parameter=display_name,
                    unit=unit,
                    start_time="0h",
                    end_time="96h",
                    start_value=_round_or_none(v0),
                    end_value=_round_or_none(v96),
                    absolute_change=_round_or_none(_safe_subtract(v96, v0)),
                    percentage_change=_round_or_none(_safe_pct_change(v96, v0)),
                )
            )

    return changes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_pct_drift(
    end_val: Optional[float], start_val: Optional[float]
) -> Optional[float]:
    """Computes (end - start) / start — the fractional drift formula used in training."""
    if end_val is None or start_val is None:
        return None
    if start_val == 0.0:
        return None
    return (end_val - start_val) / start_val


def _safe_pct_change(
    end_val: Optional[float], start_val: Optional[float]
) -> Optional[float]:
    """Computes percentage change: (end - start) / |start| * 100."""
    if end_val is None or start_val is None:
        return None
    if start_val == 0.0:
        return None
    return ((end_val - start_val) / abs(start_val)) * 100.0


def _safe_subtract(
    a: Optional[float], b: Optional[float]
) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _round_or_none(val: Optional[float], ndigits: int = 4) -> Optional[float]:
    if val is None:
        return None
    return round(val, ndigits)
