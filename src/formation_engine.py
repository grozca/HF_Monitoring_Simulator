"""
Hidden formation-response engine for the Frac Monitoring Training Simulator.

This is not a full GEOS / PKN / KGD / 3D fracture model. It is a practical
training proxy that creates hidden formation variables behind the dashboard.

Training idea:
    Surface PPA is the schedule.
    Bottomhole PPA is what the formation is actually seeing.
    Net pressure + bottomhole PPA + acceptance tell the story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


FormationState = Literal[
    "Normal growth",
    "High net pressure",
    "Low acceptance",
    "Elevated screenout risk",
    "Critical screenout risk",
    "Pressure sink / communication",
]


@dataclass(frozen=True)
class FormationConfig:
    """Configuration for the simplified formation engine."""

    plane_strain_modulus_mpsi: float = 4.0
    target_fracture_height_ft: float = 220.0
    min_width_in: float = 0.03
    max_width_in: float = 1.20
    initial_half_length_ft: float = 25.0
    leakoff_base_bpm: float = 2.0
    proppant_capacity_lb_per_bbl: float = 95.0
    screenout_pressure_slope_threshold_psi_min: float = 80.0
    natural_fracture_sink_strength: float = 0.30
    pressure_adjustment_max_psi: float = 420.0


def _require_any_column(df: pd.DataFrame, candidates: list[str], concept: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column for {concept}. Expected one of: {candidates}")


def _time_steps_min(time_min: pd.Series) -> np.ndarray:
    time = time_min.to_numpy(dtype=float)
    if len(time) <= 1:
        return np.ones(len(time), dtype=float)
    return np.gradient(time)


def _safe_gradient(y: pd.Series, x: pd.Series) -> np.ndarray:
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    if len(yv) < 2:
        return np.zeros_like(yv)
    grad = np.gradient(yv, xv)
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _scenario_progress(time_min: pd.Series, start_fraction: float) -> pd.Series:
    duration = max(float(time_min.max()), 1.0)
    start = start_fraction * duration
    return ((time_min - start) / max(duration - start, 1.0)).clip(lower=0.0, upper=1.0)


def _calculate_cumulative_volume(df: pd.DataFrame) -> pd.Series:
    if "cum_slurry_bbl" in df.columns:
        return df["cum_slurry_bbl"].astype(float)

    required = {"time_min", "slurry_rate_bpm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for cumulative volume: {sorted(missing)}")

    dt = _time_steps_min(df["time_min"])
    cum = np.cumsum(df["slurry_rate_bpm"].to_numpy(dtype=float) * dt)
    return pd.Series(cum, index=df.index, name="cum_slurry_bbl")


def _calculate_cumulative_bottomhole_sand(df: pd.DataFrame) -> pd.Series:
    if "cum_sand_bh_lb" in df.columns:
        return df["cum_sand_bh_lb"].astype(float)

    ppa_col = "bottomhole_ppa" if "bottomhole_ppa" in df.columns else "ppa"
    if "clean_rate_bpm" in df.columns:
        clean_rate = df["clean_rate_bpm"].astype(float)
    else:
        clean_rate = df["slurry_rate_bpm"].astype(float) / (1.0 + 0.045 * df[ppa_col].astype(float))

    sand_rate = clean_rate * 42.0 * df[ppa_col].astype(float)
    dt = _time_steps_min(df["time_min"])
    cum_sand = np.cumsum(sand_rate.to_numpy(dtype=float) * dt)
    return pd.Series(cum_sand, index=df.index, name="cum_sand_bh_lb")


def calculate_fracture_width_proxy(
    net_pressure_psi: pd.Series,
    config: FormationConfig | None = None,
) -> pd.Series:
    """Fracture width proxy in inches."""
    config = config or FormationConfig()
    e_prime_psi = max(config.plane_strain_modulus_mpsi * 1_000_000.0, 1.0)
    net = net_pressure_psi.clip(lower=0.0).astype(float)

    width_ft = net * config.target_fracture_height_ft / e_prime_psi
    width_in = (12.0 * width_ft).clip(lower=config.min_width_in, upper=config.max_width_in)
    return pd.Series(width_in, index=net_pressure_psi.index, name="fracture_width_in")


def calculate_fracture_length_proxy(
    df: pd.DataFrame,
    width_in: pd.Series,
    config: FormationConfig | None = None,
) -> pd.Series:
    """Estimate hydraulic half-length from effective fracture volume."""
    config = config or FormationConfig()
    cum_slurry = _calculate_cumulative_volume(df)
    leakoff = df.get("cumulative_leakoff_bbl", pd.Series(0.0, index=df.index)).astype(float)

    effective_volume_bbl = (cum_slurry - leakoff).clip(lower=0.0)
    width_ft = (width_in.astype(float) / 12.0).clip(lower=config.min_width_in / 12.0)
    fracture_volume_ft3 = effective_volume_bbl * 5.615
    half_length = fracture_volume_ft3 / (2.0 * config.target_fracture_height_ft * width_ft)
    half_length = half_length + config.initial_half_length_ft
    half_length = pd.Series(half_length, index=df.index).rolling(window=5, min_periods=1).mean()
    return half_length.clip(lower=config.initial_half_length_ft).rename("fracture_half_length_ft")


def calculate_leakoff_proxy(
    df: pd.DataFrame,
    fracture_half_length_ft: pd.Series,
    config: FormationConfig | None = None,
) -> pd.DataFrame:
    """Carter-inspired leakoff proxy."""
    config = config or FormationConfig()
    out = df.copy()
    time = out["time_min"].astype(float)

    area_index = np.sqrt(
        (fracture_half_length_ft.clip(lower=1.0) * config.target_fracture_height_ft) / 220_000.0
    )
    time_factor = np.sqrt(time.clip(lower=0.0) + 1.0) / np.sqrt(time.max() + 1.0)
    leakoff_rate = config.leakoff_base_bpm * area_index * time_factor
    leakoff_rate = pd.Series(leakoff_rate, index=out.index, name="leakoff_rate_bpm").clip(lower=0.0)

    cumulative_leakoff = np.cumsum(leakoff_rate.to_numpy(dtype=float) * _time_steps_min(time))
    out["leakoff_rate_bpm"] = leakoff_rate
    out["cumulative_leakoff_bbl"] = cumulative_leakoff
    return out


def calculate_fluid_efficiency(df: pd.DataFrame) -> pd.Series:
    cum_slurry = _calculate_cumulative_volume(df)
    leakoff = df.get("cumulative_leakoff_bbl", pd.Series(0.0, index=df.index)).astype(float)
    efficiency = ((cum_slurry - leakoff) / cum_slurry.replace(0.0, np.nan)).fillna(0.0)
    return efficiency.clip(lower=0.0, upper=1.0).rename("fluid_efficiency")


def calculate_acceptance_index(
    df: pd.DataFrame,
    config: FormationConfig | None = None,
) -> pd.Series:
    """Formation acceptance index. 1.0 = normal acceptance, 0.0 = poor acceptance."""
    config = config or FormationConfig()
    required = {"time_min", "fracture_width_in", "net_pressure_psi"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for acceptance index: {sorted(missing)}")

    time = df["time_min"].astype(float)
    net_slope = pd.Series(_safe_gradient(df["net_pressure_psi"], time), index=df.index)
    bh_ppa = df["bottomhole_ppa"].astype(float) if "bottomhole_ppa" in df.columns else df["ppa"].astype(float)
    width = df["fracture_width_in"].astype(float)

    cum_sand = _calculate_cumulative_bottomhole_sand(df)
    cum_slurry = _calculate_cumulative_volume(df).clip(lower=1.0)
    sand_capacity_lb = cum_slurry * config.proppant_capacity_lb_per_bbl
    sand_loading_ratio = (cum_sand / sand_capacity_lb.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 2.0)

    proppant_gate = _sigmoid(14.0 * (bh_ppa - 0.12))
    ppa_penalty = np.clip((bh_ppa - 0.5) / 2.5, 0.0, 1.0)
    slope_penalty = (
        np.clip(net_slope / config.screenout_pressure_slope_threshold_psi_min, 0.0, 2.0) / 2.0
    ) * proppant_gate
    narrow_width_penalty = np.clip((0.28 - width) / 0.28, 0.0, 1.0) * proppant_gate
    sand_loading_penalty = np.clip((sand_loading_ratio - 0.55) / 0.65, 0.0, 1.0)

    total_penalty = (
        0.25 * ppa_penalty
        + 0.35 * slope_penalty
        + 0.20 * narrow_width_penalty
        + 0.20 * sand_loading_penalty
    )
    acceptance = (1.0 - total_penalty).clip(0.0, 1.0)
    return pd.Series(acceptance, index=df.index, name="acceptance_index")


def calculate_screenout_risk(
    df: pd.DataFrame,
    config: FormationConfig | None = None,
) -> pd.Series:
    """Screenout risk from formation-response variables."""
    config = config or FormationConfig()
    required = {"time_min", "net_pressure_psi", "acceptance_index", "fracture_width_in"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for screenout risk: {sorted(missing)}")

    time = df["time_min"].astype(float)
    net_slope = pd.Series(_safe_gradient(df["net_pressure_psi"], time), index=df.index)
    bh_ppa = df["bottomhole_ppa"].astype(float) if "bottomhole_ppa" in df.columns else df["ppa"].astype(float)
    acceptance = df["acceptance_index"].astype(float)
    width = df["fracture_width_in"].astype(float)

    proppant_gate = _sigmoid(14.0 * (bh_ppa - 0.12))
    ppa_term = np.clip((bh_ppa - 0.35) / 2.0, 0.0, 1.5)
    slope_term = np.clip(net_slope / config.screenout_pressure_slope_threshold_psi_min, -1.0, 3.0) * proppant_gate
    acceptance_term = 1.0 - acceptance
    width_term = np.clip((0.25 - width) / 0.25, 0.0, 1.0) * proppant_gate

    raw = -2.2 + 1.15 * ppa_term + 1.45 * slope_term + 2.00 * acceptance_term + 0.80 * width_term
    risk = _sigmoid(raw)
    return pd.Series(risk, index=df.index, name="screenout_risk").clip(0.0, 1.0)


def classify_formation_state(df: pd.DataFrame) -> pd.Series:
    """Classify the hidden formation state for training feedback."""
    states: list[FormationState] = []
    time = df["time_min"].astype(float)
    net_slope = pd.Series(_safe_gradient(df["net_pressure_psi"], time), index=df.index)

    for idx, row in df.iterrows():
        risk = float(row.get("screenout_risk", 0.0))
        acceptance = float(row.get("acceptance_index", 1.0))
        net_pressure = float(row.get("net_pressure_psi", 0.0))
        slope = float(net_slope.loc[idx])
        bh_ppa = float(row.get("bottomhole_ppa", row.get("ppa", 0.0)))
        surface_ppa = float(row.get("surface_ppa", row.get("ppa", 0.0)))

        if slope < -80.0 and bh_ppa > 0.1 and surface_ppa > 0.05:
            state: FormationState = "Pressure sink / communication"
        elif risk >= 0.80:
            state = "Critical screenout risk"
        elif risk >= 0.55:
            state = "Elevated screenout risk"
        elif acceptance <= 0.45:
            state = "Low acceptance"
        elif net_pressure >= 1800.0:
            state = "High net pressure"
        else:
            state = "Normal growth"

        states.append(state)

    return pd.Series(states, index=df.index, name="formation_state")


def calculate_pressure_adjustment(
    df: pd.DataFrame,
    config: FormationConfig,
    *,
    scenario: str,
    severity: float,
) -> pd.Series:
    """Pressure adjustment used by the hydraulic engine before BHP is finalized."""
    acceptance = df["acceptance_index"].astype(float)
    risk = df["screenout_risk"].astype(float)
    time = df["time_min"].astype(float)
    strength = float(np.clip(severity, 0.5, 2.0))

    adjustment = (1.0 - acceptance).clip(lower=-0.25, upper=0.85) * config.pressure_adjustment_max_psi
    adjustment += risk.clip(0.0, 1.0) * 95.0

    if scenario == "Screenout":
        adjustment += (250.0 + 140.0 * strength) * _scenario_progress(time, 0.64)
    elif scenario in {"Frac Hit", "Natural Fracture Hit"}:
        adjustment -= (180.0 + 80.0 * strength) * _scenario_progress(time, 0.56)

    return pd.Series(adjustment, index=df.index, name="formation_pressure_adjustment_psi")


def apply_formation_engine(
    df: pd.DataFrame,
    config: FormationConfig | None = None,
    *,
    scenario: str = "Normal Job",
    severity: float = 1.0,
) -> pd.DataFrame:
    """Add hidden formation-response variables to a simulated job."""
    config = config or FormationConfig()
    out = df.copy()

    _require_any_column(out, ["time_min"], "time")
    net_col = _require_any_column(out, ["net_pressure_psi", "estimated_net_pressure_psi"], "net pressure")
    if net_col != "net_pressure_psi":
        out["net_pressure_psi"] = out[net_col].astype(float)

    if "cum_slurry_bbl" not in out.columns:
        out["cum_slurry_bbl"] = _calculate_cumulative_volume(out)
    if "cum_sand_bh_lb" not in out.columns:
        out["cum_sand_bh_lb"] = _calculate_cumulative_bottomhole_sand(out)

    out["fracture_width_in"] = calculate_fracture_width_proxy(out["net_pressure_psi"], config)
    out["fracture_half_length_ft"] = calculate_fracture_length_proxy(out, out["fracture_width_in"], config)
    out = calculate_leakoff_proxy(out, out["fracture_half_length_ft"], config)
    out["fracture_half_length_ft"] = calculate_fracture_length_proxy(out, out["fracture_width_in"], config)

    out["fracture_height_ft"] = config.target_fracture_height_ft
    out["fluid_efficiency"] = calculate_fluid_efficiency(out)
    out["acceptance_index"] = calculate_acceptance_index(out, config)
    out["screenout_risk"] = calculate_screenout_risk(out, config)

    out["formation_pressure_adjustment_psi"] = calculate_pressure_adjustment(
        out,
        config,
        scenario=scenario,
        severity=severity,
    )
    out["formation_state"] = classify_formation_state(out)
    out["formation_acceptance_pct"] = 100.0 * out["acceptance_index"]
    out["screenout_risk_pct"] = 100.0 * out["screenout_risk"]

    # Compatibility aliases used by the existing dashboard and physics engine.
    out["fracture_width_proxy_in"] = out["fracture_width_in"]
    out["fracture_length_ft"] = out["fracture_half_length_ft"]
    out["leakoff_bbl"] = out["cumulative_leakoff_bbl"]
    out["natural_fracture_intersection"] = scenario in {"Frac Hit", "Natural Fracture Hit"}
    return out


def summarize_formation_at_time(df: pd.DataFrame, time_min: float) -> dict[str, float | str]:
    """Return formation summary at nearest time."""
    if df.empty:
        return {}

    idx = int(np.argmin(np.abs(df["time_min"].to_numpy(dtype=float) - float(time_min))))
    row = df.iloc[idx]
    return {
        "time_min": float(row["time_min"]),
        "formation_state": str(row.get("formation_state", "Unknown")),
        "fracture_width_in": float(row.get("fracture_width_in", np.nan)),
        "fracture_half_length_ft": float(row.get("fracture_half_length_ft", np.nan)),
        "fracture_height_ft": float(row.get("fracture_height_ft", np.nan)),
        "leakoff_rate_bpm": float(row.get("leakoff_rate_bpm", np.nan)),
        "fluid_efficiency": float(row.get("fluid_efficiency", np.nan)),
        "acceptance_index": float(row.get("acceptance_index", np.nan)),
        "screenout_risk": float(row.get("screenout_risk", np.nan)),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.physics_engine import run_physics_engine
    from src.sand_transport import SandTransportConfig, apply_sand_transport
    from src.treatment_schedule import TreatmentScheduleConfig, generate_treatment_schedule

    schedule = generate_treatment_schedule(
        TreatmentScheduleConfig(duration_min=60.0, dt_min=0.25, target_rate_bpm=80.0, max_ppa=2.0)
    )
    schedule = apply_sand_transport(
        schedule,
        SandTransportConfig(measured_depth_ft=15000.0, wellbore_capacity_bbl_per_ft=0.020),
    )
    simulated = run_physics_engine(
        schedule,
        scenario="Screenout",
        target_rate_bpm=80.0,
        duration_min=60.0,
        severity=1.0,
    )
    print(
        simulated[
            [
                "time_min",
                "bottomhole_ppa",
                "net_pressure_psi",
                "fracture_width_in",
                "fracture_half_length_ft",
                "leakoff_rate_bpm",
                "fluid_efficiency",
                "acceptance_index",
                "screenout_risk",
                "formation_state",
            ]
        ].tail(20)
    )
    print(summarize_formation_at_time(simulated, 45.0))
