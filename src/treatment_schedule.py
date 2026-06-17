"""
Treatment schedule generator for the Frac Monitoring Training Simulator.

Builds a simple hydraulic-fracturing stage schedule:

    Ramp up -> Pad -> Breakdown / Breakover -> Slurry ramp -> Main proppant -> Flush

Main output columns:

    time_min
    phase
    slurry_rate_bpm
    ppa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd


GAL_PER_BBL = 42.0
PROPPANT_BULK_DENSITY_LBM_PER_GAL = 22.0


PhaseName = Literal[
    "Ramp up",
    "Pad",
    "Breakdown / Breakover",
    "Slurry ramp",
    "Main proppant",
    "Flush",
]

PpaProfile = Literal["linear", "stepped"]


PHASE_ORDER = (
    "Ramp up",
    "Pad",
    "Breakdown / Breakover",
    "Slurry ramp",
    "Main proppant",
    "Flush",
)


@dataclass(frozen=True)
class TreatmentScheduleConfig:
    """Treatment schedule configuration."""

    duration_min: float = 60.0
    dt_min: float = 0.25
    target_rate_bpm: float = 80.0
    max_ppa: float = 2.0
    ppa_nudge: float = 0.0
    ppa_profile: PpaProfile = "stepped"

    ramp_up_fraction: float = 0.04
    pad_fraction: float = 0.18
    breakdown_fraction: float = 0.22
    slurry_ramp_fraction: float = 0.45
    main_prop_fraction: float = 0.85


def validate_schedule_config(config: TreatmentScheduleConfig) -> None:
    if config.duration_min <= 0:
        raise ValueError("duration_min must be > 0.")
    if config.dt_min <= 0:
        raise ValueError("dt_min must be > 0.")
    if config.dt_min > config.duration_min:
        raise ValueError("dt_min cannot be greater than duration_min.")
    if config.target_rate_bpm <= 0:
        raise ValueError("target_rate_bpm must be > 0.")
    if config.max_ppa < 0:
        raise ValueError("max_ppa cannot be negative.")
    if config.ppa_profile not in {"linear", "stepped"}:
        raise ValueError("ppa_profile must be either 'linear' or 'stepped'.")

    fractions = [
        config.ramp_up_fraction,
        config.pad_fraction,
        config.breakdown_fraction,
        config.slurry_ramp_fraction,
        config.main_prop_fraction,
        1.0,
    ]
    if any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("All phase fractions must be in the interval (0, 1].")
    if fractions != sorted(fractions):
        raise ValueError(
            "Phase fractions must be increasing: "
            "ramp_up < pad < breakdown < slurry_ramp < main_prop < flush."
        )


def phase_boundaries(config_or_duration: TreatmentScheduleConfig | float) -> Dict[str, float]:
    """Return phase boundaries in minutes."""
    if isinstance(config_or_duration, TreatmentScheduleConfig):
        config = config_or_duration
        validate_schedule_config(config)
        return {
            "ramp_up": config.ramp_up_fraction * config.duration_min,
            "pad": config.pad_fraction * config.duration_min,
            "breakdown": config.breakdown_fraction * config.duration_min,
            "slurry_ramp": config.slurry_ramp_fraction * config.duration_min,
            "main_prop": config.main_prop_fraction * config.duration_min,
            "flush": config.duration_min,
        }

    duration_min = float(config_or_duration)
    if duration_min <= 0:
        raise ValueError("duration_min must be > 0.")
    return {
        "ramp_up": 0.04 * duration_min,
        "pad": 0.18 * duration_min,
        "breakdown": 0.22 * duration_min,
        "slurry_ramp": 0.45 * duration_min,
        "main_prop": 0.85 * duration_min,
        "flush": duration_min,
    }


def get_phase_at_time(time_min: float, boundaries: Dict[str, float]) -> PhaseName:
    if time_min <= boundaries["ramp_up"]:
        return "Ramp up"
    if time_min <= boundaries["pad"]:
        return "Pad"
    if time_min <= boundaries["breakdown"]:
        return "Breakdown / Breakover"
    if time_min <= boundaries["slurry_ramp"]:
        return "Slurry ramp"
    if time_min <= boundaries["main_prop"]:
        return "Main proppant"
    return "Flush"


def calculate_rate_at_time(time_min: float, target_rate_bpm: float, boundaries: Dict[str, float]) -> float:
    if time_min <= boundaries["ramp_up"]:
        return float(target_rate_bpm * time_min / max(boundaries["ramp_up"], 0.01))
    return float(target_rate_bpm)


def calculate_ppa_at_time(
    time_min: float,
    max_ppa: float,
    boundaries: Dict[str, float],
    ppa_nudge: float = 0.0,
    ppa_profile: PpaProfile = "stepped",
) -> float:
    if ppa_profile == "stepped":
        return calculate_stepped_ppa_at_time(
            time_min,
            max_ppa=max_ppa,
            boundaries=boundaries,
            ppa_nudge=ppa_nudge,
        )

    if time_min <= boundaries["breakdown"]:
        ppa = 0.0
    elif time_min <= boundaries["slurry_ramp"]:
        ppa = np.interp(
            time_min,
            [boundaries["breakdown"], boundaries["slurry_ramp"]],
            [0.25, min(1.0, max_ppa)],
        )
    elif time_min <= boundaries["main_prop"]:
        ppa = np.interp(
            time_min,
            [boundaries["slurry_ramp"], boundaries["main_prop"]],
            [min(1.0, max_ppa), max_ppa],
        )
    else:
        ppa = np.interp(
            time_min,
            [boundaries["main_prop"], boundaries["flush"]],
            [max_ppa, 0.0],
        )
    return float(np.clip(ppa + ppa_nudge, 0.0, 5.0))


def calculate_stepped_ppa_at_time(
    time_min: float,
    max_ppa: float,
    boundaries: Dict[str, float],
    ppa_nudge: float = 0.0,
) -> float:
    """Field-style stepped proppant schedule."""
    if time_min <= boundaries["breakdown"]:
        ppa = 0.0
    elif time_min <= boundaries["slurry_ramp"]:
        span = max(boundaries["slurry_ramp"] - boundaries["breakdown"], 0.01)
        progress = (time_min - boundaries["breakdown"]) / span
        ramp_steps = [0.25, 0.50, min(1.0, max_ppa)]
        ppa = ramp_steps[min(int(progress * len(ramp_steps)), len(ramp_steps) - 1)]
    elif time_min <= boundaries["main_prop"]:
        span = max(boundaries["main_prop"] - boundaries["slurry_ramp"], 0.01)
        progress = (time_min - boundaries["slurry_ramp"]) / span
        main_steps = [
            min(1.5, max_ppa),
            min(2.0, max_ppa),
            min(2.5, max_ppa),
            max_ppa,
        ]
        ppa = main_steps[min(int(progress * len(main_steps)), len(main_steps) - 1)]
    else:
        tail_in_end = boundaries["main_prop"] + 0.18 * (boundaries["flush"] - boundaries["main_prop"])
        ppa = max_ppa if time_min <= tail_in_end else 0.0

    return float(np.clip(ppa + ppa_nudge, 0.0, 5.0))


def generate_treatment_schedule(config: TreatmentScheduleConfig | None = None) -> pd.DataFrame:
    """Generate treatment schedule as a pandas DataFrame."""
    config = config or TreatmentScheduleConfig()
    validate_schedule_config(config)
    boundaries = phase_boundaries(config)
    times = np.round(np.arange(0.0, config.duration_min + config.dt_min, config.dt_min), 4)

    rows = []
    for time_min in times:
        phase = get_phase_at_time(float(time_min), boundaries)
        rows.append(
            {
                "time_min": float(time_min),
                "phase": phase,
                "slurry_rate_bpm": calculate_rate_at_time(
                    float(time_min),
                    config.target_rate_bpm,
                    boundaries,
                ),
                "ppa": calculate_ppa_at_time(
                    float(time_min),
                    max_ppa=config.max_ppa,
                    boundaries=boundaries,
                    ppa_nudge=config.ppa_nudge,
                    ppa_profile=config.ppa_profile,
                ),
            }
        )

    return pd.DataFrame(rows)


def generate_custom_ppa_schedule(
    duration_min: float,
    dt_min: float,
    target_rate_bpm: float,
    ppa_points: Dict[float, float],
) -> pd.DataFrame:
    """Generate a custom PPA schedule using time/PPA control points."""
    if duration_min <= 0:
        raise ValueError("duration_min must be > 0.")
    if dt_min <= 0:
        raise ValueError("dt_min must be > 0.")
    if target_rate_bpm <= 0:
        raise ValueError("target_rate_bpm must be > 0.")
    if not ppa_points:
        raise ValueError("ppa_points cannot be empty.")

    sorted_points = sorted((float(k), float(v)) for k, v in ppa_points.items())
    x = np.array([point[0] for point in sorted_points], dtype=float)
    y = np.array([point[1] for point in sorted_points], dtype=float)

    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    if x[-1] < duration_min:
        x = np.append(x, duration_min)
        y = np.append(y, y[-1])

    times = np.round(np.arange(0.0, duration_min + dt_min, dt_min), 4)
    ppa = np.clip(np.interp(times, x, y), 0.0, 5.0)
    default_config = TreatmentScheduleConfig(
        duration_min=duration_min,
        dt_min=dt_min,
        target_rate_bpm=target_rate_bpm,
        max_ppa=float(np.max(ppa)),
    )
    boundaries = phase_boundaries(default_config)

    return pd.DataFrame(
        {
            "time_min": times.astype(float),
            "phase": [get_phase_at_time(float(t), boundaries) for t in times],
            "slurry_rate_bpm": np.full_like(times, target_rate_bpm, dtype=float),
            "ppa": ppa.astype(float),
        }
    )


def summarize_schedule(df: pd.DataFrame) -> Dict[str, float]:
    """Return high-level schedule summary metrics."""
    required = {"time_min", "slurry_rate_bpm", "ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for schedule summary: {sorted(missing)}")

    dt = np.gradient(df["time_min"].to_numpy())
    slurry_bbl = float(np.sum(df["slurry_rate_bpm"].to_numpy() * dt))
    clean_rate = df["slurry_rate_bpm"].to_numpy() / (1.0 + 0.045 * df["ppa"].to_numpy())
    sand_lb = float(np.sum(clean_rate * GAL_PER_BBL * df["ppa"].to_numpy() * dt))

    return {
        "duration_min": float(df["time_min"].max()),
        "avg_rate_bpm": float(df["slurry_rate_bpm"].mean()),
        "max_rate_bpm": float(df["slurry_rate_bpm"].max()),
        "max_ppa": float(df["ppa"].max()),
        "scheduled_slurry_bbl": slurry_bbl,
        "scheduled_sand_lb": sand_lb,
    }


def build_base_schedule(
    duration_minutes: float = 60.0,
    time_step_seconds: float = 60.0,
    rate_multiplier: float = 1.0,
    ppa_multiplier: float = 1.0,
    target_rate_bpm: float | None = None,
    max_ppa: float | None = None,
    ppa_nudge: float = 0.0,
) -> pd.DataFrame:
    """Compatibility wrapper for earlier app versions."""
    config = TreatmentScheduleConfig(
        duration_min=duration_minutes,
        dt_min=time_step_seconds / 60.0,
        target_rate_bpm=(target_rate_bpm or 80.0) * rate_multiplier,
        max_ppa=(max_ppa or 2.0) * ppa_multiplier,
        ppa_nudge=ppa_nudge,
    )
    schedule = generate_treatment_schedule(config)
    schedule["clean_rate_bpm"] = schedule["slurry_rate_bpm"] / (1.0 + 0.045 * schedule["ppa"])
    schedule["sand_rate_lbm_min"] = schedule["clean_rate_bpm"] * GAL_PER_BBL * schedule["ppa"]
    dt = np.gradient(schedule["time_min"].to_numpy())
    schedule["cumulative_sand_lbm"] = np.cumsum(schedule["sand_rate_lbm_min"].to_numpy() * dt)
    schedule["cumulative_slurry_bbl"] = np.cumsum(schedule["slurry_rate_bpm"].to_numpy() * dt)
    schedule["sand_rate_lb_min"] = schedule["sand_rate_lbm_min"]
    schedule["cum_sand_lb"] = schedule["cumulative_sand_lbm"]
    schedule["cum_slurry_bbl"] = schedule["cumulative_slurry_bbl"]
    return schedule
