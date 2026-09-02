"""
Package: src.decision
Purpose: Operational screening decision rules (PASS / REVIEW / REJECT) and sequential early-exit logic.
"""

from src.decision.screening_decision import (
    DecisionConfig,
    DEFAULT_CONFIG,
    make_screening_decision,
    run_screening_pipeline,
    run_sequential_screening,
)

__all__ = [
    "DecisionConfig",
    "DEFAULT_CONFIG",
    "make_screening_decision",
    "run_screening_pipeline",
    "run_sequential_screening",
]
