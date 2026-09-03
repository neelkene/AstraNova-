"""
Module: backend.schemas
Purpose: Pydantic v2 request/response models for the FastAPI backend.

All sensor value ranges are based on the empirical dataset distributions:
  - IDDQ:               ~80–130 μA (0h baseline), up to ~200 μA post-stress
  - Leakage Current:    ~1.5–4.5 μA
  - Propagation Delay:  ~0.5–2.5 ns
  - Voltage:            ~1.1–1.3 V
  - Temperature:        ~120–130 °C
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScreeningStage(str, Enum):
    """The screening stage determined from available sensor data."""
    INSUFFICIENT = "insufficient"   # Only 0h data — cannot run models
    GATE_24H = "24h"                # 0h + 24h available
    GATE_96H = "96h"                # 0h + 24h + 96h available


class ScreeningDecision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# Sensor Measurement Blocks
# ---------------------------------------------------------------------------

class Measurements0h(BaseModel):
    """Pre-burn-in baseline sensor readings at t=0h."""
    iddq_uA_0h: Optional[float] = Field(
        None, ge=0.0, le=500.0,
        description="Quiescent drain current at 0h (μA)"
    )
    leakage_current_uA_0h: Optional[float] = Field(
        None, ge=0.0, le=50.0,
        description="Leakage current at 0h (μA)"
    )
    propagation_delay_ns_0h: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Propagation delay at 0h (ns)"
    )
    voltage_V_0h: Optional[float] = Field(
        None, ge=0.5, le=2.0,
        description="Supply voltage at 0h (V)"
    )
    temperature_C_0h: Optional[float] = Field(
        None, ge=100.0, le=200.0,
        description="Junction temperature at 0h (°C)"
    )


class Measurements24h(BaseModel):
    """Early burn-in sensor readings at t=24h."""
    iddq_uA_24h: Optional[float] = Field(
        None, ge=0.0, le=500.0,
        description="Quiescent drain current at 24h (μA)"
    )
    leakage_current_uA_24h: Optional[float] = Field(
        None, ge=0.0, le=50.0,
        description="Leakage current at 24h (μA)"
    )
    propagation_delay_ns_24h: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Propagation delay at 24h (ns)"
    )
    voltage_V_24h: Optional[float] = Field(
        None, ge=0.5, le=2.0,
        description="Supply voltage at 24h (V)"
    )
    temperature_C_24h: Optional[float] = Field(
        None, ge=100.0, le=200.0,
        description="Junction temperature at 24h (°C)"
    )


class Measurements96h(BaseModel):
    """Mid burn-in sensor readings at t=96h."""
    iddq_uA_96h: Optional[float] = Field(
        None, ge=0.0, le=500.0,
        description="Quiescent drain current at 96h (μA)"
    )
    leakage_current_uA_96h: Optional[float] = Field(
        None, ge=0.0, le=50.0,
        description="Leakage current at 96h (μA)"
    )
    propagation_delay_ns_96h: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Propagation delay at 96h (ns)"
    )
    voltage_V_96h: Optional[float] = Field(
        None, ge=0.5, le=2.0,
        description="Supply voltage at 96h (V)"
    )
    temperature_C_96h: Optional[float] = Field(
        None, ge=100.0, le=200.0,
        description="Junction temperature at 96h (°C)"
    )


# ---------------------------------------------------------------------------
# Request Model
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Main prediction request. Supports two usage modes:

    Mode 1 — Component Lookup (demo/testing):
        Provide only `component_id`. The backend loads measurements from the
        dataset. Ground-truth labels are NEVER used as model inputs.

    Mode 2 — Raw Measurements (production inference):
        Provide `measurements_0h` and `measurements_24h` (and optionally
        `measurements_96h`). The backend computes drift features and runs
        the appropriate screening gate(s).
    """
    component_id: Optional[str] = Field(
        None,
        pattern=r"^SYN_C\d{5}$",
        description="Dataset component identifier (e.g. SYN_C01216). "
                    "If provided, measurements are loaded from the dataset."
    )
    measurements_0h: Optional[Measurements0h] = Field(
        None, description="Pre-burn-in baseline readings at t=0h"
    )
    measurements_24h: Optional[Measurements24h] = Field(
        None, description="Early burn-in readings at t=24h"
    )
    measurements_96h: Optional[Measurements96h] = Field(
        None, description="Mid burn-in readings at t=96h (optional — enables 96h gate)"
    )

    @model_validator(mode="after")
    def validate_input_mode(self) -> "PredictRequest":
        has_id = self.component_id is not None
        has_raw = self.measurements_0h is not None
        if not has_id and not has_raw:
            raise ValueError(
                "Provide either 'component_id' (dataset lookup) or "
                "'measurements_0h' + 'measurements_24h' (raw inference)."
            )
        if has_id and has_raw:
            raise ValueError(
                "Provide either 'component_id' OR raw measurements — not both."
            )
        if has_raw and self.measurements_24h is None:
            raise ValueError(
                "When providing raw measurements, 'measurements_24h' is required "
                "together with 'measurements_0h'."
            )
        return self


# ---------------------------------------------------------------------------
# Response Sub-Models
# ---------------------------------------------------------------------------

class FeatureImportanceItem(BaseModel):
    """Single feature importance entry (sorted highest → lowest)."""
    feature: str = Field(..., description="Feature column name")
    importance: float = Field(..., description="Relative importance score (Gini / split gain)")


class ModuleAGateResult(BaseModel):
    """Module A classification result for a single screening gate."""
    gate: str = Field(..., description="Screening gate ('24h' or '96h')")
    model_name: str = Field(..., description="Algorithm used (e.g. 'LogisticRegression')")
    prediction: int = Field(..., description="Predicted class (0=Normal, 1=Defective)")
    class_name: str = Field(..., description="Human-readable class ('normal' or 'defective')")
    risk_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model-estimated defect probability ∈ [0, 1]"
    )
    features_used: int = Field(..., description="Number of feature columns fed to the model")
    feature_importances: List[FeatureImportanceItem] = Field(
        default_factory=list,
        description="Top-10 feature importances (empty for models without a valid importance mechanism)"
    )


class ModuleBGateResult(BaseModel):
    """Module B regression result for a single screening gate."""
    gate: str = Field(..., description="Screening gate ('24h' or '96h')")
    model_name: str = Field(..., description="Algorithm used (e.g. 'GradientBoostingRegressor')")
    predicted_iddq_drift_168h: float = Field(
        ..., description="Predicted 168h IDDQ drift as a raw fraction (e.g. 0.098 = 9.8%)"
    )
    predicted_iddq_drift_168h_pct: float = Field(
        ..., description="Predicted 168h IDDQ drift expressed as a percentage (e.g. 9.8)"
    )


class ObservedChange(BaseModel):
    """
    Human-friendly observed measurement change for one parameter.
    Reports the actual change between available time-points.
    Wording deliberately avoids attributing causation.
    """
    parameter: str = Field(..., description="Physical parameter name (e.g. 'IDDQ')")
    unit: str = Field(..., description="Engineering unit (e.g. 'μA')")
    start_time: str = Field(..., description="Reference time-point (e.g. '0h')")
    end_time: str = Field(..., description="Comparison time-point (e.g. '96h')")
    start_value: Optional[float] = Field(None, description="Sensor reading at start_time")
    end_value: Optional[float] = Field(None, description="Sensor reading at end_time")
    absolute_change: Optional[float] = Field(
        None, description="Absolute difference: end_value − start_value"
    )
    percentage_change: Optional[float] = Field(
        None, description="Percentage change: (end − start) / |start| × 100 "
                          "(None when start_value is zero or unavailable)"
    )


class FinalDecision(BaseModel):
    """Combined screening decision synthesised from Module A + Module B outputs."""
    status: ScreeningDecision
    confidence_level: ConfidenceLevel
    reason: str = Field(..., description="Structured explanation of the decision basis")
    recommendation: str = Field(..., description="Actionable engineering recommendation")


class GateResults(BaseModel):
    """Bundled Module A + Module B outputs for a single screening gate."""
    module_a: ModuleAGateResult
    module_b: ModuleBGateResult
    gate_decision: FinalDecision


class MeasurementsSnapshot(BaseModel):
    """Read-back of the sensor values that were used for inference."""
    zero_h: Optional[Dict[str, Optional[float]]] = Field(None, alias="0h")
    twenty_four_h: Optional[Dict[str, Optional[float]]] = Field(None, alias="24h")
    ninety_six_h: Optional[Dict[str, Optional[float]]] = Field(None, alias="96h")

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """
    Complete, frontend-ready prediction response.

    Fields:
      component_id       — The component being evaluated
      screening_stage    — Which gate(s) could be run given available data
      measurements       — Echo of input sensor values used for inference
      observed_changes   — Human-friendly parameter change summaries
      gate_24h           — A24 + B24 results (present if 24h gate was run)
      gate_96h           — A96 + B96 results (present if 96h gate was run)
      final_decision     — Combined PASS / REVIEW / REJECT verdict
    """
    component_id: str
    screening_stage: ScreeningStage
    measurements: MeasurementsSnapshot
    observed_changes: List[ObservedChange] = Field(default_factory=list)
    gate_24h: Optional[GateResults] = None
    gate_96h: Optional[GateResults] = None
    final_decision: Optional[FinalDecision] = None


# ---------------------------------------------------------------------------
# Component Lookup
# ---------------------------------------------------------------------------

class ComponentLookupResponse(BaseModel):
    """
    Measurements for a single component retrieved from the dataset.
    Ground-truth labels (module_a_label, iddq_drift_168h_true, component_type)
    are NEVER included in this response.
    """
    component_id: str
    is_in_locked_test_set: bool
    measurements_0h: Dict[str, Optional[float]]
    measurements_24h: Dict[str, Optional[float]]
    measurements_96h: Dict[str, Optional[float]]
    measurements_168h: Optional[Dict[str, Optional[float]]] = None


class ComponentListItem(BaseModel):
    """Summary row for the component list endpoint."""
    component_id: str
    iddq_uA_0h: Optional[float] = None
    iddq_uA_24h: Optional[float] = None
    iddq_uA_96h: Optional[float] = None


class ComponentListResponse(BaseModel):
    """Paginated list of available component IDs."""
    total_available: int
    returned_count: int
    split_filter: str
    components: List[ComponentListItem]


# ---------------------------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------------------------

class DatasetOverviewResponse(BaseModel):
    """Aggregated dataset statistics for the overview section."""
    total_components: int
    normal_count: int
    normal_pct: float
    drifting_count: int
    drifting_pct: float
    anomalous_count: int
    anomalous_pct: float
    burnin_gates: List[str]


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """API health and model readiness status."""
    status: str = Field(..., description="'ok' when all models are loaded")
    module_a_loaded: bool
    module_b_loaded: bool
    models_detail: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-model load status: a24, a96, b24, b96"
    )

