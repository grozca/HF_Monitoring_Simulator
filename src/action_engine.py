"""
src/action_engine.py

Action engine for the Frac Monitoring Training Simulator.

Purpose
-------
This module turns the simulator into an interactive training game. The learner
can take an action during the job, and the module modifies future curves and
scores the decision.

Supported actions
-----------------
    Continue pumping
    Reduce PPA / prepare flush
    Hold PPA
    Increase PPA
    Decrease rate
    Increase rate
    Flush
    Shut down
    Check pumps / stabilize rate
    Verify sensor / compare channels
    Evaluate offset communication

Core math
---------
For an action applied at time t0, define a smooth response function:

    R(t) = 0                                      for t < t0 + delay
    R(t) = 1 - exp(-(t - t0 - delay) / tau)       for t >= t0 + delay

Then apply control changes:

    ppa_new(t)  = ppa_old(t)  * (1 - alpha_ppa * R(t))
    rate_new(t) = rate_old(t) * (1 - alpha_rate * R(t))

For good screenout response:

    net_pressure_new(t) = net_pressure_old(t)
                          - beta * R(t) * max(net_pressure_old(t) - target_net, 0)

For bad actions during screenout:

    net_pressure_new(t) = net_pressure_old(t) + beta_bad * R(t)^gamma

This is semi-physical training logic. It is not a real frac-control algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


OperatorAction = Literal[
    "Continue pumping",
    "Reduce PPA / prepare flush",
    "Hold PPA",
    "Increase PPA",
    "Decrease rate",
    "Increase rate",
    "Flush",
    "Shut down",
    "Check pumps / stabilize rate",
    "Verify sensor / compare channels",
    "Evaluate offset communication",
]

DiagnosisName = Literal[
    "Normal treatment response",
    "Possible screenout",
    "Possible perforation or cluster restriction",
    "Pump or rate-control issue",
    "Possible frac hit / pressure communication",
    "Possible sensor or data quality issue",
    "Unknown",
]


@dataclass(frozen=True)
class ActionConfig:
    # Dynamic response
    response_delay_min: float = 0.50
    response_tau_min: float = 2.00

    # Control magnitudes
    reduce_ppa_fraction: float = 0.50
    increase_ppa_fraction: float = 0.35
    decrease_rate_fraction: float = 0.15
    increase_rate_fraction: float = 0.10
    shutdown_rate_fraction: float = 0.98

    # Pressure/risk response
    good_screenout_pressure_relief_fraction: float = 0.28
    flush_pressure_relief_fraction: float = 0.35
    bad_screenout_pressure_growth_psi: float = 1100.0
    pump_stabilization_strength: float = 0.90
    sensor_smoothing_strength: float = 0.85

    # Scoring
    score_fast_action_bonus: int = 5
    score_dangerous_action_penalty: int = -20


@dataclass
class ActionContext:
    time_min: float
    scenario: str
    diagnosis: DiagnosisName
    selected_action: OperatorAction
    current_pressure_psi: float
    current_rate_bpm: float
    current_bh_ppa: float
    current_net_pressure_psi: float
    current_screenout_risk: float
    current_acceptance_index: float


@dataclass
class ActionResult:
    modified_df: pd.DataFrame
    score_delta: int
    risk_delta: float
    action_quality: str
    explanation: str
    evidence: list[str]
    recommended_action: str


# =============================================================================
# Basic helpers
# =============================================================================


def action_options() -> list[OperatorAction]:
    return [
        "Continue pumping",
        "Reduce PPA / prepare flush",
        "Hold PPA",
        "Increase PPA",
        "Decrease rate",
        "Increase rate",
        "Flush",
        "Shut down",
        "Check pumps / stabilize rate",
        "Verify sensor / compare channels",
        "Evaluate offset communication",
    ]


def _nearest_row(df: pd.DataFrame, time_min: float) -> pd.Series:
    idx = int(np.argmin(np.abs(df["time_min"].to_numpy(float) - float(time_min))))
    return df.iloc[idx]


def _response_curve(time: pd.Series, action_time_min: float, delay_min: float, tau_min: float) -> np.ndarray:
    t = time.to_numpy(float)
    x = t - float(action_time_min) - float(delay_min)
    r = np.where(x <= 0.0, 0.0, 1.0 - np.exp(-x / max(tau_min, 1e-6)))
    return np.clip(r, 0.0, 1.0)


def recommended_action_for_diagnosis(diagnosis: DiagnosisName) -> str:
    table = {
        "Normal treatment response": "Continue pumping",
        "Possible screenout": "Reduce PPA / prepare flush",
        "Possible perforation or cluster restriction": "Hold PPA",
        "Pump or rate-control issue": "Check pumps / stabilize rate",
        "Possible frac hit / pressure communication": "Evaluate offset communication",
        "Possible sensor or data quality issue": "Verify sensor / compare channels",
        "Unknown": "Continue pumping",
    }
    return table.get(diagnosis, "Continue pumping")


def infer_context(
    df: pd.DataFrame,
    time_min: float,
    scenario: str,
    diagnosis: DiagnosisName,
    selected_action: OperatorAction,
) -> ActionContext:
    row = _nearest_row(df, time_min)
    return ActionContext(
        time_min=float(row["time_min"]),
        scenario=scenario,
        diagnosis=diagnosis,
        selected_action=selected_action,
        current_pressure_psi=float(row.get("surface_pressure_psi", np.nan)),
        current_rate_bpm=float(row.get("slurry_rate_bpm", np.nan)),
        current_bh_ppa=float(row.get("bottomhole_ppa", row.get("ppa", 0.0))),
        current_net_pressure_psi=float(row.get("net_pressure_psi", np.nan)),
        current_screenout_risk=float(row.get("screenout_risk", 0.0)),
        current_acceptance_index=float(row.get("acceptance_index", 1.0)),
    )


# =============================================================================
# Decision scoring logic
# =============================================================================


def classify_action_quality(context: ActionContext) -> Tuple[str, str, int, list[str]]:
    """Return quality, recommended action, base score and evidence."""
    diagnosis = context.diagnosis
    action = context.selected_action
    recommended = recommended_action_for_diagnosis(diagnosis)

    evidence = [
        f"Current rate: {context.current_rate_bpm:.1f} bpm",
        f"Bottomhole PPA: {context.current_bh_ppa:.2f}",
        f"Net pressure: {context.current_net_pressure_psi:.0f} psi",
        f"Screenout risk: {100 * context.current_screenout_risk:.0f}%",
        f"Acceptance index: {100 * context.current_acceptance_index:.0f}%",
    ]

    if action == recommended:
        return "safe / recommended", recommended, 10, evidence

    if diagnosis == "Possible screenout":
        if action in {"Flush", "Hold PPA", "Decrease rate"}:
            return "acceptable but not ideal", recommended, 4, evidence
        if action in {"Increase PPA", "Increase rate", "Continue pumping"}:
            return "dangerous for screenout", recommended, -20, evidence

    if diagnosis == "Possible perforation or cluster restriction":
        if action in {"Reduce PPA / prepare flush", "Decrease rate"}:
            return "conservative", recommended, 2, evidence
        if action in {"Increase PPA", "Increase rate"}:
            return "can worsen restriction", recommended, -12, evidence

    if diagnosis == "Pump or rate-control issue":
        if action in {"Decrease rate", "Hold PPA"}:
            return "conservative", recommended, 2, evidence
        if action in {"Increase rate", "Increase PPA"}:
            return "can destabilize pumps", recommended, -12, evidence

    if diagnosis == "Possible frac hit / pressure communication":
        if action in {"Continue pumping", "Hold PPA"}:
            return "incomplete response", recommended, -3, evidence
        if action in {"Increase rate", "Increase PPA"}:
            return "risky during communication", recommended, -12, evidence

    if diagnosis == "Possible sensor or data quality issue":
        if action in {"Continue pumping", "Hold PPA"}:
            return "acceptable only if verified quickly", recommended, 1, evidence
        if action in {"Flush", "Shut down"}:
            return "possibly unnecessary", recommended, -6, evidence

    if diagnosis == "Normal treatment response":
        if action in {"Continue pumping", "Hold PPA"}:
            return "reasonable", recommended, 5, evidence
        if action in {"Flush", "Shut down"}:
            return "unnecessary interruption", recommended, -8, evidence

    return "not recommended", recommended, -5, evidence


def calculate_reaction_bonus(context: ActionContext, event_onset_min: Optional[float], cfg: ActionConfig) -> int:
    if event_onset_min is None:
        return 0
    reaction_time = max(0.0, context.time_min - event_onset_min)
    if reaction_time <= 3.0:
        return cfg.score_fast_action_bonus
    if reaction_time <= 7.0:
        return max(1, cfg.score_fast_action_bonus // 2)
    return 0


# =============================================================================
# Curve modification logic
# =============================================================================


def _ensure_common_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    if "surface_ppa" not in out.columns and "ppa" in out.columns:
        out["surface_ppa"] = out["ppa"]
    if "bottomhole_ppa" not in out.columns and "ppa" in out.columns:
        out["bottomhole_ppa"] = out["ppa"]
    if "screenout_risk" not in out.columns:
        out["screenout_risk"] = 0.0
    if "acceptance_index" not in out.columns:
        out["acceptance_index"] = 1.0
    return out


def _recalculate_basic_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Lightweight recalculation after action changes."""
    out = out.copy()
    ppa = out["surface_ppa"] if "surface_ppa" in out.columns else out.get("ppa", pd.Series(0.0, index=out.index))

    if "slurry_rate_bpm" in out.columns:
        out["clean_rate_bpm"] = out["slurry_rate_bpm"] / (1.0 + 0.045 * ppa)
        if "surface_ppa" in out.columns:
            out["sand_rate_surface_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["surface_ppa"]
        if "bottomhole_ppa" in out.columns:
            out["sand_rate_bh_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["bottomhole_ppa"]

    if "screenout_risk" in out.columns:
        out["screenout_risk"] = out["screenout_risk"].clip(0.0, 1.0)
        out["screenout_risk_pct"] = 100.0 * out["screenout_risk"]

    if "acceptance_index" in out.columns:
        out["acceptance_index"] = out["acceptance_index"].clip(0.0, 1.0)
        out["formation_acceptance_pct"] = 100.0 * out["acceptance_index"]

    if {"closure_pressure_psi", "net_pressure_psi"}.issubset(out.columns):
        out["bhp_psi"] = out["closure_pressure_psi"] + out["net_pressure_psi"]

    if {"bhp_psi", "hydrostatic_psi", "pipe_friction_psi", "perf_friction_psi"}.issubset(out.columns):
        out["surface_pressure_psi"] = (
            out["bhp_psi"] - out["hydrostatic_psi"] + out["pipe_friction_psi"] + out["perf_friction_psi"]
        )

    return out


def apply_action_to_future(df: pd.DataFrame, context: ActionContext, cfg: Optional[ActionConfig] = None) -> pd.DataFrame:
    """Modify future curves after a learner action."""
    cfg = cfg or ActionConfig()
    out = _ensure_common_columns(df)
    future = out["time_min"] >= context.time_min
    r = pd.Series(
        _response_curve(out["time_min"], context.time_min, cfg.response_delay_min, cfg.response_tau_min),
        index=out.index,
    )
    action = context.selected_action

    if action == "Reduce PPA / prepare flush":
        out.loc[future, "surface_ppa"] *= (1.0 - cfg.reduce_ppa_fraction * r.loc[future])
        out.loc[future, "ppa"] = out.loc[future, "surface_ppa"]
        out.loc[future, "bottomhole_ppa"] *= (1.0 - 0.65 * cfg.reduce_ppa_fraction * r.loc[future])
        if "net_pressure_psi" in out.columns:
            target = float(out.loc[future, "net_pressure_psi"].iloc[0])
            excess = (out.loc[future, "net_pressure_psi"] - target).clip(lower=0.0)
            out.loc[future, "net_pressure_psi"] -= cfg.good_screenout_pressure_relief_fraction * r.loc[future] * excess
        out.loc[future, "screenout_risk"] *= (1.0 - 0.45 * r.loc[future])
        out.loc[future, "acceptance_index"] += 0.25 * r.loc[future]

    elif action == "Hold PPA":
        hold = float(_nearest_row(out, context.time_min).get("surface_ppa", 0.0))
        out.loc[future, "surface_ppa"] = (1.0 - r.loc[future]) * out.loc[future, "surface_ppa"] + r.loc[future] * hold
        out.loc[future, "ppa"] = out.loc[future, "surface_ppa"]
        if "net_pressure_psi" in out.columns and context.diagnosis == "Possible screenout":
            target = float(out.loc[future, "net_pressure_psi"].iloc[0])
            excess = (out.loc[future, "net_pressure_psi"] - target).clip(lower=0.0)
            out.loc[future, "net_pressure_psi"] -= 0.12 * r.loc[future] * excess

    elif action == "Increase PPA":
        out.loc[future, "surface_ppa"] *= (1.0 + cfg.increase_ppa_fraction * r.loc[future])
        out.loc[future, "ppa"] = out.loc[future, "surface_ppa"]
        out.loc[future, "bottomhole_ppa"] *= (1.0 + 0.70 * cfg.increase_ppa_fraction * r.loc[future])
        if "net_pressure_psi" in out.columns:
            out.loc[future, "net_pressure_psi"] += cfg.bad_screenout_pressure_growth_psi * (r.loc[future] ** 1.5)
        out.loc[future, "screenout_risk"] += 0.35 * r.loc[future]

    elif action == "Decrease rate":
        out.loc[future, "slurry_rate_bpm"] *= (1.0 - cfg.decrease_rate_fraction * r.loc[future])
        if "pipe_friction_psi" in out.columns:
            out.loc[future, "pipe_friction_psi"] *= (1.0 - 0.30 * r.loc[future])
        if "perf_friction_psi" in out.columns:
            out.loc[future, "perf_friction_psi"] *= (1.0 - 0.35 * r.loc[future])
        if context.diagnosis == "Possible screenout" and "net_pressure_psi" in out.columns:
            out.loc[future, "net_pressure_psi"] *= (1.0 - 0.10 * r.loc[future])

    elif action == "Increase rate":
        out.loc[future, "slurry_rate_bpm"] *= (1.0 + cfg.increase_rate_fraction * r.loc[future])
        if "pipe_friction_psi" in out.columns:
            out.loc[future, "pipe_friction_psi"] *= (1.0 + 0.35 * r.loc[future])
        if "perf_friction_psi" in out.columns:
            out.loc[future, "perf_friction_psi"] *= (1.0 + 0.45 * r.loc[future])
        if context.diagnosis == "Possible screenout" and "net_pressure_psi" in out.columns:
            out.loc[future, "net_pressure_psi"] += 0.35 * cfg.bad_screenout_pressure_growth_psi * r.loc[future]

    elif action == "Flush":
        out.loc[future, "surface_ppa"] *= (1.0 - r.loc[future])
        out.loc[future, "ppa"] = out.loc[future, "surface_ppa"]
        out.loc[future, "bottomhole_ppa"] *= (1.0 - 0.70 * r.loc[future])
        if "net_pressure_psi" in out.columns:
            out.loc[future, "net_pressure_psi"] *= (1.0 - cfg.flush_pressure_relief_fraction * r.loc[future])
        out.loc[future, "screenout_risk"] *= (1.0 - 0.60 * r.loc[future])

    elif action == "Shut down":
        out.loc[future, "slurry_rate_bpm"] *= (1.0 - cfg.shutdown_rate_fraction * r.loc[future])
        out.loc[future, "surface_ppa"] *= (1.0 - r.loc[future])
        out.loc[future, "ppa"] = out.loc[future, "surface_ppa"]
        for col in ["pipe_friction_psi", "perf_friction_psi", "surface_pressure_psi"]:
            if col in out.columns:
                out.loc[future, col] *= (1.0 - 0.80 * r.loc[future])

    elif action == "Check pumps / stabilize rate":
        current_rate = float(_nearest_row(out, context.time_min).get("slurry_rate_bpm", np.nan))
        if np.isfinite(current_rate):
            out.loc[future, "slurry_rate_bpm"] = (
                (1.0 - cfg.pump_stabilization_strength * r.loc[future]) * out.loc[future, "slurry_rate_bpm"]
                + (cfg.pump_stabilization_strength * r.loc[future]) * current_rate
            )

    elif action == "Verify sensor / compare channels":
        if "surface_pressure_psi" in out.columns:
            smooth = out["surface_pressure_psi"].rolling(7, center=True, min_periods=1).median()
            out.loc[future, "surface_pressure_psi"] = (
                (1.0 - cfg.sensor_smoothing_strength * r.loc[future]) * out.loc[future, "surface_pressure_psi"]
                + (cfg.sensor_smoothing_strength * r.loc[future]) * smooth.loc[future]
            )

    elif action == "Evaluate offset communication":
        out.loc[future, "screenout_risk"] *= (1.0 - 0.25 * r.loc[future])
        if "formation_state" in out.columns:
            out.loc[future, "formation_state"] = "Pressure sink / communication"

    # Continue pumping intentionally changes nothing.
    return _recalculate_basic_columns(out)


# =============================================================================
# Main public function
# =============================================================================


def build_action_explanation(
    context: ActionContext,
    quality: str,
    recommended: str,
    score_delta: int,
    risk_delta: float,
    reaction_bonus: int,
) -> str:
    lines = [
        f"Action selected: {context.selected_action}",
        f"Diagnosis: {context.diagnosis}",
        f"Action quality: {quality}",
        f"Recommended action: {recommended}",
        f"Score change: {score_delta:+d}",
        f"Screenout risk change over next minutes: {100 * risk_delta:+.1f} percentage points",
    ]
    if reaction_bonus > 0:
        lines.append(f"Reaction bonus: +{reaction_bonus}")

    if context.diagnosis == "Possible screenout":
        lines.append(
            "Logic: rate is usually stable, bottomhole PPA is high, net pressure is rising, "
            "and formation acceptance is declining. Reducing PPA or preparing flush reduces "
            "solid loading and helps avoid bridging/packing."
        )
    elif context.diagnosis == "Possible perforation or cluster restriction":
        lines.append(
            "Logic: if the pressure increase is dominated by perforation friction, the issue "
            "is near-wellbore/perf restriction rather than fracture storage. Avoid aggressive PPA/rate increases."
        )
    elif context.diagnosis == "Pump or rate-control issue":
        lines.append(
            "Logic: stabilize equipment first. Formation interpretation is unreliable while rate is unstable."
        )
    elif context.diagnosis == "Possible frac hit / pressure communication":
        lines.append(
            "Logic: pressure drop at stable rate suggests communication or pressure sink, not screenout."
        )
    elif context.diagnosis == "Possible sensor or data quality issue":
        lines.append(
            "Logic: verify data quality before changing the job because the pressure signal may be false."
        )
    else:
        lines.append("Logic: normal response means continue monitoring and avoid unnecessary changes.")
    return "\n".join(lines)


def process_player_action(
    df: pd.DataFrame,
    *,
    time_min: float,
    scenario: str,
    diagnosis: DiagnosisName,
    selected_action: OperatorAction,
    event_onset_min: Optional[float] = None,
    config: Optional[ActionConfig] = None,
) -> ActionResult:
    cfg = config or ActionConfig()
    context = infer_context(df, time_min, scenario, diagnosis, selected_action)
    quality, recommended, base_score, evidence = classify_action_quality(context)
    reaction_bonus = calculate_reaction_bonus(context, event_onset_min, cfg)

    modified = apply_action_to_future(df, context, cfg)
    after_time = min(float(modified["time_min"].max()), context.time_min + 8.0)
    baseline_after = _nearest_row(df, after_time)
    after = _nearest_row(modified, after_time)
    baseline_risk = float(baseline_after.get("screenout_risk", context.current_screenout_risk))
    after_risk = float(after.get("screenout_risk", baseline_risk))
    risk_delta = after_risk - baseline_risk

    score_delta = int(base_score + reaction_bonus)
    if diagnosis == "Possible screenout":
        if risk_delta < -0.10:
            score_delta += 5
        elif risk_delta > 0.10:
            score_delta -= 8
    if "dangerous" in quality:
        score_delta += cfg.score_dangerous_action_penalty // 2

    explanation = build_action_explanation(context, quality, recommended, score_delta, risk_delta, reaction_bonus)
    return ActionResult(modified, score_delta, float(risk_delta), quality, explanation, evidence, recommended)


if __name__ == "__main__":
    t = np.arange(0, 60.25, 0.25)
    screenout_growth = 25.0 * np.clip(t - 40.0, 0.0, None) ** 1.5
    df = pd.DataFrame({
        "time_min": t,
        "slurry_rate_bpm": np.where(t < 2, 40*t, 80),
        "ppa": np.clip((t-15)/30*2, 0, 2),
        "surface_ppa": np.clip((t-15)/30*2, 0, 2),
        "bottomhole_ppa": np.clip((t-18)/30*2, 0, 2),
        "net_pressure_psi": 600 + 10*t + screenout_growth,
        "closure_pressure_psi": 6200,
        "hydrostatic_psi": 3900,
        "pipe_friction_psi": 1500,
        "perf_friction_psi": 500,
        "screenout_risk": np.clip((t-38)/20, 0, 1),
        "acceptance_index": 1 - np.clip((t-38)/25, 0, 1),
    })
    df["bhp_psi"] = df["closure_pressure_psi"] + df["net_pressure_psi"]
    df["surface_pressure_psi"] = df["bhp_psi"] - df["hydrostatic_psi"] + df["pipe_friction_psi"] + df["perf_friction_psi"]

    result = process_player_action(
        df,
        time_min=42.0,
        scenario="Screenout",
        diagnosis="Possible screenout",
        selected_action="Reduce PPA / prepare flush",
        event_onset_min=40.0,
    )
    print(result.explanation)
    print(result.modified_df[["time_min", "ppa", "bottomhole_ppa", "net_pressure_psi", "screenout_risk"]].tail())
