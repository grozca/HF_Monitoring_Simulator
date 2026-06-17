from __future__ import annotations

import numpy as np
import pandas as pd

from src.treatment_schedule import GAL_PER_BBL, PROPPANT_BULK_DENSITY_LBM_PER_GAL


SCENARIO_OPTIONS = [
    "Normal Job",
    "Screenout",
    "Perforation Plugging",
    "Pump Issue",
    "Frac Hit",
    "Sensor Error",
]


def _severity(time_min: pd.Series, start_min: float, end_min: float) -> pd.Series:
    return ((time_min - start_min) / max(end_min - start_min, 1.0)).clip(0.0, 1.0)


def _recalculate_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["slurry_rate_bpm"] = out["clean_rate_bpm"] * (
        1.0 + out["ppa"] / PROPPANT_BULK_DENSITY_LBM_PER_GAL
    )
    out["sand_rate_lbm_min"] = out["clean_rate_bpm"] * GAL_PER_BBL * out["ppa"]
    dt_min = out["time_min"].diff().fillna(0.0)
    out["cumulative_sand_lbm"] = (out["sand_rate_lbm_min"] * dt_min).cumsum()
    out["cumulative_slurry_bbl"] = (out["slurry_rate_bpm"] * dt_min).cumsum()
    out["sand_rate_lb_min"] = out["sand_rate_lbm_min"]
    out["cum_sand_lb"] = out["cumulative_sand_lbm"]
    out["cum_slurry_bbl"] = out["cumulative_slurry_bbl"]
    return out


def _base_modifiers(schedule: pd.DataFrame) -> pd.DataFrame:
    out = schedule.copy()
    out["event"] = "Normal"
    out["pipe_friction_multiplier"] = 1.0
    out["perf_friction_multiplier"] = 1.0
    out["net_pressure_adjustment_psi"] = 0.0
    out["surface_pressure_bias_psi"] = 0.0
    return out


def apply_scenario(
    schedule: pd.DataFrame,
    scenario: str,
    severity: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Apply field-style anomalies before pressure calculations."""
    if scenario not in SCENARIO_OPTIONS:
        raise ValueError(f"Unknown scenario: {scenario}")

    out = _base_modifiers(schedule)
    time_min = out["time_min"]
    duration = max(float(time_min.max()), 1.0)
    strength = float(np.clip(severity, 0.5, 2.0))
    rng = np.random.default_rng(seed)

    if scenario == "Normal Job":
        return out

    if scenario == "Screenout":
        start = duration * 0.63
        progression = _severity(time_min, start, duration)
        mask = progression > 0
        out.loc[mask, "event"] = "Screenout Trend"
        out.loc[mask, "perf_friction_multiplier"] = (
            1.0 + (1.0 + 0.65 * strength) * progression[mask] ** 1.35
        )
        out.loc[mask, "net_pressure_adjustment_psi"] = (
            (720.0 + 330.0 * strength) * progression[mask] ** 1.25
        )

    elif scenario == "Perforation Plugging":
        start = duration * 0.47
        progression = _severity(time_min, start, duration)
        mask = progression > 0
        out.loc[mask, "event"] = "Perforation Restriction"
        out.loc[mask, "perf_friction_multiplier"] = 1.0 + (0.75 + 0.35 * strength) * progression[mask]
        out.loc[mask, "net_pressure_adjustment_psi"] = (120.0 + 90.0 * strength) * progression[mask]

    elif scenario == "Pump Issue":
        start = duration * 0.52
        end = duration * 0.72
        progression = _severity(time_min, start, end)
        recovery = 1.0 - _severity(time_min, end, duration) * 0.45
        mask = (time_min >= start) & (time_min <= duration * 0.82)
        oscillation = 1.0 - (0.03 + 0.035 * strength) * np.sin((time_min - start) * 1.55)
        rate_factor = np.where(
            time_min <= end,
            (1.0 - (0.11 + 0.075 * strength) * progression) * oscillation,
            recovery,
        )
        if mask.any():
            rate_factor = pd.Series(rate_factor, index=out.index)
            rate_factor.loc[mask] += rng.normal(0.0, 0.006 * strength, int(mask.sum()))
        out.loc[mask, "event"] = "Pump Rate Loss"
        out.loc[mask, "clean_rate_bpm"] = out.loc[mask, "clean_rate_bpm"] * rate_factor[mask]
        out.loc[mask, "net_pressure_adjustment_psi"] = -110.0 * strength * progression[mask]
        out = _recalculate_rates(out)

    elif scenario == "Frac Hit":
        start = duration * 0.56
        progression = _severity(time_min, start, start + duration * 0.10)
        leakoff_relief = np.exp(-0.085 * (time_min - start).clip(lower=0.0))
        mask = time_min >= start
        out.loc[mask, "event"] = "Frac Hit"
        out.loc[mask, "net_pressure_adjustment_psi"] = -(470.0 + 230.0 * strength) * leakoff_relief[mask]
        out.loc[mask, "pipe_friction_multiplier"] = 1.0 + 0.05 * strength * progression[mask]

    elif scenario == "Sensor Error":
        start = duration * 0.50
        end = duration * 0.62
        mask = (time_min >= start) & (time_min <= end)
        oscillation = np.sin((time_min - start) * np.pi * 1.7)
        noise = pd.Series(0.0, index=out.index)
        if mask.any():
            noise.loc[mask] = rng.normal(0.0, 28.0 * strength, int(mask.sum()))
        out.loc[mask, "event"] = "Pressure Sensor Error"
        out.loc[mask, "surface_pressure_bias_psi"] = (
            (260.0 + 170.0 * strength) * oscillation[mask] + noise[mask]
        )

    return out
