import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.inference.predict import (
    load_models,
    predict_24h,
    predict_96h,
    run_inference_gate,
)


@dataclass
class DecisionConfig:
    """
    Configurable Operational Decision Thresholds for Burn-In Screening.
    
    Note on Empirical Threshold Design:
    - Normal components in dataset have true 168h drift of ~1.0% (max 2.0%).
    - Drifting components have true 168h drift of ~10.0% (range 5.0% - 15.0%).
    - Anomalous components have true 168h drift of ~29.9% (range 20.0% - 40.0%).
    
    These thresholds are configurable operating points tuned to balance defect escape risk
    (False Negative Rate) against false scrap rate (False Positive Rate).
    """
    # 24h Early Screening Gate Thresholds
    prob_reject_24h: float = 0.75       # Defect probability >= 75% -> early REJECT
    drift_reject_24h: float = 0.12      # Projected 168h drift >= 12.0% -> early REJECT
    prob_pass_24h: float = 0.25         # Defect probability < 25% -> eligible for early PASS
    drift_pass_24h: float = 0.035       # Projected 168h drift <= 3.5% -> eligible for early PASS
    
    # 96h Mid Screening Gate Thresholds (Final Qualification Gate)
    prob_reject_96h: float = 0.60       # Defect probability >= 60% -> REJECT
    drift_reject_96h: float = 0.05      # Projected 168h drift >= 5.0% -> REJECT (catches all true drifters)
    prob_pass_96h: float = 0.30         # Defect probability < 30% -> PASS
    drift_pass_96h: float = 0.03        # Projected 168h drift <= 3.0% -> PASS
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_CONFIG = DecisionConfig()


def make_screening_decision(
    defect_probability: float,
    predicted_168h_drift: float,
    screening_gate: str = "24h",
    config: Optional[DecisionConfig] = None
) -> Dict[str, Any]:
    """
    Evaluates classification probability and forecasted degradation against configurable thresholds
    to assign a transparent, human-readable PASS / REVIEW / REJECT decision.
    
    Decision Rules:
    - 24h Gate:
        * REJECT if defect_probability >= prob_reject_24h OR predicted_168h_drift >= drift_reject_24h
        * PASS (Early Exit) if defect_probability < prob_pass_24h AND predicted_168h_drift < drift_pass_24h
        * REVIEW (Continue Stress) otherwise (borderline or uncertain signal at 24h)
        
    - 96h Gate:
        * REJECT if defect_probability >= prob_reject_96h OR predicted_168h_drift >= drift_reject_96h
        * PASS if defect_probability < prob_pass_96h AND predicted_168h_drift < drift_pass_96h
        * REVIEW otherwise (borderline anomaly requiring engineering manual inspection)
    """
    cfg = config or DEFAULT_CONFIG
    gate_clean = screening_gate.lower().strip()
    is_24h = "24" in gate_clean
    drift_pct = predicted_168h_drift * 100.0

    if is_24h:
        # 24h Gate Decision Logic
        if defect_probability >= cfg.prob_reject_24h or predicted_168h_drift >= cfg.drift_reject_24h:
            decision = "REJECT"
            confidence = "HIGH"
            reason = (
                f"Severe early defect signature detected at 24h: Defect Probability = {defect_probability*100:.1f}% "
                f"(Threshold: {cfg.prob_reject_24h*100:.1f}%) or Projected 168h Drift = {drift_pct:.2f}% "
                f"(Threshold: {cfg.drift_reject_24h*100:.1f}%). Early screening reject to save test resources."
            )
            recommendation = "Eject component from burn-in chamber immediately (early defect triage)."
            
        elif defect_probability < cfg.prob_pass_24h and predicted_168h_drift < cfg.drift_pass_24h:
            decision = "PASS"
            confidence = "HIGH"
            reason = (
                f"Nominal electrical stability confirmed at 24h: Defect Probability = {defect_probability*100:.1f}% "
                f"(< {cfg.prob_pass_24h*100:.1f}%) and Minimal Projected Drift = {drift_pct:.2f}% "
                f"(< {cfg.drift_pass_24h*100:.1f}%). Qualified for early burn-in exit."
            )
            recommendation = "Component qualifies for early burn-in exit (24h early release)."
            
        else:
            decision = "REVIEW"
            confidence = "MEDIUM"
            reason = (
                f"Intermediate degradation signal at 24h: Defect Probability = {defect_probability*100:.1f}% "
                f"(Between {cfg.prob_pass_24h*100:.1f}% and {cfg.prob_reject_24h*100:.1f}%) or "
                f"Projected 168h Drift = {drift_pct:.2f}% (Between {cfg.drift_pass_24h*100:.1f}% and {cfg.drift_reject_24h*100:.1f}%). "
                f"Signal requires continued stress qualification."
            )
            recommendation = "Continue component stress testing to the 96h burn-in screening gate."
            
    else:
        # 96h Gate Decision Logic
        if defect_probability >= cfg.prob_reject_96h or predicted_168h_drift >= cfg.drift_reject_96h:
            decision = "REJECT"
            confidence = "HIGH"
            reason = (
                f"Defect confirmed at 96h qualification gate: Defect Probability = {defect_probability*100:.1f}% "
                f"(Threshold: {cfg.prob_reject_96h*100:.1f}%) or Projected 168h Drift = {drift_pct:.2f}% "
                f"(Exceeds maximum allowable tolerance of {cfg.drift_reject_96h*100:.1f}%)."
            )
            recommendation = "Reject and scrap component. Fails 96h burn-in qualification standard."
            
        elif defect_probability < cfg.prob_pass_96h and predicted_168h_drift < cfg.drift_pass_96h:
            decision = "PASS"
            confidence = "HIGH"
            reason = (
                f"Parametric reliability confirmed at 96h: Defect Probability = {defect_probability*100:.1f}% "
                f"(< {cfg.prob_pass_96h*100:.1f}%) and Safe Projected Degradation = {drift_pct:.2f}% "
                f"(< {cfg.drift_pass_96h*100:.1f}%). Component meets high-reliability standards."
            )
            recommendation = "Pass component and release to production inventory."
            
        else:
            decision = "REVIEW"
            confidence = "LOW"
            reason = (
                f"Borderline parametric behavior at 96h: Defect Probability = {defect_probability*100:.1f}% "
                f"and Projected 168h Drift = {drift_pct:.2f}%. Model outputs fall in the boundary zone."
            )
            recommendation = "Flag for manual engineering review and secondary curve-trace verification."

    return {
        "decision": decision,
        "reason": reason,
        "defect_probability": round(defect_probability, 4),
        "predicted_168h_drift": round(predicted_168h_drift, 6),
        "predicted_168h_drift_pct": round(drift_pct, 3),
        "screening_gate": "24h" if is_24h else "96h",
        "confidence_level": confidence,
        "recommendation": recommendation,
    }


def run_screening_pipeline(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    gate: str = "24h",
    config: Optional[DecisionConfig] = None,
    models: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes end-to-end ML inference and decision logic for one or more components at a specific gate.
    """
    cfg = config or DEFAULT_CONFIG
    inf_results = run_inference_gate(data, gate=gate, models=models)

    single_item = isinstance(inf_results, dict)
    items = [inf_results] if single_item else inf_results

    out_records = []
    for r in items:
        dec = make_screening_decision(
            defect_probability=r["defect_probability"],
            predicted_168h_drift=r["predicted_168h_iddq_drift"],
            screening_gate=gate,
            config=cfg
        )
        # Merge inference metadata with decision output
        combined = {**r, **dec}
        out_records.append(combined)

    return out_records[0] if single_item else out_records


def run_sequential_screening(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    config: Optional[DecisionConfig] = None,
    models: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes the 2-Stage Sequential Screening Workflow:
    
    1. Evaluates component at 24h Gate (A24 + B24).
    2. If 24h Decision is PASS -> Early Exit Qualification.
    3. If 24h Decision is REJECT -> Early Defect Triage (Ejected).
    4. If 24h Decision is REVIEW (Uncertain/Marginal):
       - If 96h measurements are present in the input record:
           Evaluates component at 96h Gate (A96 + B96) for final decision.
       - If 96h measurements are not present:
           Returns 24h REVIEW verdict instructing user to proceed to 96h test.
           
    The system NEVER accesses 96h data during 24h decision evaluation.
    """
    cfg = config or DEFAULT_CONFIG
    model_suite = models or load_models()

    if isinstance(data, dict):
        df_all = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df_all = pd.DataFrame([data.to_dict()])
    elif isinstance(data, pd.DataFrame):
        df_all = data.copy()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    results = []

    for idx in range(len(df_all)):
        row = df_all.iloc[idx : idx + 1]

        # Stage 1: Strictly 24h screening
        res_24 = run_screening_pipeline(row, gate="24h", config=cfg, models=model_suite)
        if isinstance(res_24, list):
            res_24 = res_24[0]

        # Check early exit condition
        if res_24["decision"] in ["PASS", "REJECT"]:
            final_record = {
                "component_index": idx,
                "final_decision": res_24["decision"],
                "final_screening_gate": "24h",
                "early_exit_applied": True,
                "stage_1_24h": res_24,
                "stage_2_96h": None,
                "summary": f"Decided at 24h gate: {res_24['decision']} ({res_24['recommendation']})"
            }
            results.append(final_record)
        else:
            # Stage 2: Requires 96h screening
            has_96h = all(col in row.columns for col in ["iddq_uA_96h", "leakage_current_uA_96h", "propagation_delay_ns_96h", "voltage_V_96h", "temperature_C_96h"])
            
            if has_96h:
                res_96 = run_screening_pipeline(row, gate="96h", config=cfg, models=model_suite)
                if isinstance(res_96, list):
                    res_96 = res_96[0]

                final_record = {
                    "component_index": idx,
                    "final_decision": res_96["decision"],
                    "final_screening_gate": "96h",
                    "early_exit_applied": False,
                    "stage_1_24h": res_24,
                    "stage_2_96h": res_96,
                    "summary": f"Continued to 96h gate: Final {res_96['decision']} ({res_96['recommendation']})"
                }
            else:
                final_record = {
                    "component_index": idx,
                    "final_decision": "REVIEW",
                    "final_screening_gate": "24h",
                    "early_exit_applied": False,
                    "stage_1_24h": res_24,
                    "stage_2_96h": None,
                    "summary": "24h decision is REVIEW; 96h sensor readings pending. Continue stress to 96h gate."
                }
            results.append(final_record)

    return results[0] if (isinstance(data, (dict, pd.Series)) or len(results) == 1) else results


if __name__ == "__main__":
    from src.data.load_data import load_ml_ready_data
    df = load_ml_ready_data()
    
    print("=== Sequential Screening Demo on 3 Diverse Components ===")
    sample_normal = df[df['module_a_label'] == 0].iloc[0]
    sample_defective = df[df['module_a_label'] == 1].iloc[0]
    
    for label, s in [("Normal Component", sample_normal), ("Defective Component", sample_defective)]:
        seq_res = run_sequential_screening(s)
        print(f"\n--- {label} ({s['component_id']}) ---")
        print(f" Final Decision: {seq_res['final_decision']} @ {seq_res['final_screening_gate']}")
        print(f" Summary: {seq_res['summary']}")
        print(f" 24h Defect Prob: {seq_res['stage_1_24h']['defect_probability']*100:.1f}% | 24h Forecast Drift: {seq_res['stage_1_24h']['predicted_168h_drift_pct']:.2f}%")
        if seq_res['stage_2_96h']:
            print(f" 96h Defect Prob: {seq_res['stage_2_96h']['defect_probability']*100:.1f}% | 96h Forecast Drift: {seq_res['stage_2_96h']['predicted_168h_drift_pct']:.2f}%")
