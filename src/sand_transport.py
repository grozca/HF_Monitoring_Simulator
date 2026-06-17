"""
Simple proppant transport model for the Frac Monitoring Training Simulator.

The goal is to teach a real field habit: proppant concentration at surface does
not affect the fracture immediately. Slurry has to travel through the wellbore
before sand reaches the perforations.

Expected input columns:
    time_min
    slurry_rate_bpm
    ppa

Added columns include:
    surface_ppa
    sand_lag_min
    bottomhole_ppa
    sand_rate_surface_lb_min
    sand_rate_bh_lb_min
    cum_sand_surface_lb
    cum_sand_bh_lb
    sand_in_wellbore_lb
    sand_in_fracture_lb
    sand_arrived_at_perfs
    flush_started
    flush_arrived_at_perfs
    sand_arrival_time_min
    flush_arrival_time_min
    flush_efficiency
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SandTransportConfig:
    """Configuration for simplified wellbore sand transport."""

    measured_depth_ft: float = 15000.0
    wellbore_capacity_bbl_per_ft: float = 0.020
    min_rate_bpm: float = 1.0
    mixing_efficiency: float = 0.75


def estimate_wellbore_volume_bbl(config: SandTransportConfig) -> float:
    """Estimate the slurry volume from surface to perforations."""
    return config.measured_depth_ft * config.wellbore_capacity_bbl_per_ft


def estimate_constant_lag_min(rate_bpm: float, config: SandTransportConfig) -> float:
    """Estimate lag time from wellbore volume divided by slurry rate."""
    volume_bbl = estimate_wellbore_volume_bbl(config)
    rate = max(float(rate_bpm), config.min_rate_bpm)
    return volume_bbl / rate


def calculate_dynamic_lag_min(
    df: pd.DataFrame,
    config: SandTransportConfig | None = None,
) -> pd.Series:
    """Calculate time-varying sand lag from surface to perforations."""
    config = config or SandTransportConfig()

    if "slurry_rate_bpm" not in df.columns:
        raise ValueError("Missing required column: slurry_rate_bpm")

    volume_bbl = estimate_wellbore_volume_bbl(config)
    rate = df["slurry_rate_bpm"].clip(lower=config.min_rate_bpm)
    return pd.Series(volume_bbl / rate, index=df.index, name="sand_lag_min")


def _interp_with_left_fill(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    left: float = 0.0,
) -> np.ndarray:
    """Interpolate y(x) at x_new. Values before the first sample use left fill."""
    if len(x) == 0:
        return np.array([])
    return np.interp(x_new, x, y, left=left, right=float(y[-1]))


def calculate_bottomhole_ppa(
    df: pd.DataFrame,
    config: SandTransportConfig | None = None,
) -> pd.Series:
    """Calculate bottomhole PPA using a dynamic wellbore lag."""
    config = config or SandTransportConfig()

    required = {"time_min", "ppa", "slurry_rate_bpm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for sand lag: {sorted(missing)}")

    time = df["time_min"].to_numpy(dtype=float)
    surface_ppa = df["ppa"].to_numpy(dtype=float)
    lag_min = calculate_dynamic_lag_min(df, config).to_numpy(dtype=float)

    delayed_time = time - lag_min
    bottomhole = _interp_with_left_fill(delayed_time, time, surface_ppa, left=0.0)

    mixing = float(np.clip(config.mixing_efficiency, 0.01, 1.0))
    smoothed = np.zeros_like(bottomhole)
    for i, value in enumerate(bottomhole):
        smoothed[i] = value if i == 0 else mixing * value + (1.0 - mixing) * smoothed[i - 1]

    return pd.Series(smoothed, index=df.index, name="bottomhole_ppa")


def _time_steps_min(time_min: pd.Series) -> np.ndarray:
    time = time_min.to_numpy(dtype=float)
    if len(time) <= 1:
        return np.ones(len(time), dtype=float)
    return np.gradient(time)


def calculate_sand_rates_and_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate surface and bottomhole sand rates plus inventory in the wellbore."""
    required = {"time_min", "surface_ppa", "bottomhole_ppa", "slurry_rate_bpm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for sand inventory: {sorted(missing)}")

    out = df.copy()

    if "clean_rate_bpm" not in out.columns:
        out["clean_rate_bpm"] = out["slurry_rate_bpm"] / (1.0 + 0.045 * out["surface_ppa"])

    out["sand_rate_surface_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["surface_ppa"]
    out["sand_rate_bh_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["bottomhole_ppa"]

    dt = _time_steps_min(out["time_min"])
    out["cum_sand_surface_lb"] = np.cumsum(out["sand_rate_surface_lb_min"].to_numpy(dtype=float) * dt)
    out["cum_sand_bh_lb"] = np.cumsum(out["sand_rate_bh_lb_min"].to_numpy(dtype=float) * dt)
    out["sand_in_wellbore_lb"] = (out["cum_sand_surface_lb"] - out["cum_sand_bh_lb"]).clip(lower=0.0)
    out["sand_in_fracture_lb"] = out["cum_sand_bh_lb"]

    surface_total = out["cum_sand_surface_lb"].replace(0.0, np.nan)
    out["flush_efficiency"] = (out["cum_sand_bh_lb"] / surface_total).clip(lower=0.0, upper=1.05)
    out["flush_efficiency"] = out["flush_efficiency"].fillna(1.0)

    return out


def detect_sand_and_flush_arrival(df: pd.DataFrame, ppa_threshold: float = 0.05) -> pd.DataFrame:
    """Add Boolean columns for sand and flush arrival at perforations."""
    required = {"surface_ppa", "bottomhole_ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for arrival detection: {sorted(missing)}")

    out = df.copy()
    surface_has_sand = out["surface_ppa"] > ppa_threshold
    bh_has_sand = out["bottomhole_ppa"] > ppa_threshold

    out["sand_arrived_at_perfs"] = bh_has_sand.cummax()
    out["flush_started"] = surface_has_sand.cummax() & (~surface_has_sand)
    out["flush_arrived_at_perfs"] = bh_has_sand.cummax() & (~bh_has_sand)
    return out


def get_first_arrival_times(df: pd.DataFrame) -> dict[str, float | None]:
    """Return first sand and flush arrival times."""
    result: dict[str, float | None] = {
        "first_sand_at_surface_min": None,
        "first_sand_at_perfs_min": None,
        "flush_start_surface_min": None,
        "flush_arrival_perfs_min": None,
    }

    if "surface_ppa" in df.columns:
        mask = df["surface_ppa"] > 0.05
        if mask.any():
            result["first_sand_at_surface_min"] = float(df.loc[mask, "time_min"].iloc[0])

    if "bottomhole_ppa" in df.columns:
        mask = df["bottomhole_ppa"] > 0.05
        if mask.any():
            result["first_sand_at_perfs_min"] = float(df.loc[mask, "time_min"].iloc[0])

    if "flush_started" in df.columns:
        mask = df["flush_started"]
        if mask.any():
            result["flush_start_surface_min"] = float(df.loc[mask, "time_min"].iloc[0])

    if "flush_arrived_at_perfs" in df.columns:
        mask = df["flush_arrived_at_perfs"]
        if mask.any():
            result["flush_arrival_perfs_min"] = float(df.loc[mask, "time_min"].iloc[0])

    return result


def apply_sand_transport(
    df: pd.DataFrame,
    config: SandTransportConfig | None = None,
    *,
    ppa_threshold: float = 0.05,
) -> pd.DataFrame:
    """Add proppant transport columns to a treatment schedule or simulated job."""
    config = config or SandTransportConfig()

    required = {"time_min", "slurry_rate_bpm", "ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for sand transport: {sorted(missing)}")

    out = df.copy()
    out["surface_ppa"] = out["ppa"]
    out["sand_lag_min"] = calculate_dynamic_lag_min(out, config)
    out["bottomhole_ppa"] = calculate_bottomhole_ppa(out, config)

    out = calculate_sand_rates_and_inventory(out)
    out = detect_sand_and_flush_arrival(out, ppa_threshold=ppa_threshold)
    out["ppa_lag_delta"] = out["surface_ppa"] - out["bottomhole_ppa"]

    arrivals = get_first_arrival_times(out)
    out["sand_arrival_time_min"] = arrivals["first_sand_at_perfs_min"]
    out["flush_arrival_time_min"] = arrivals["flush_arrival_perfs_min"]

    out["sand_rate_lbm_min"] = out["sand_rate_surface_lb_min"]
    out["sand_rate_bh_lbm_min"] = out["sand_rate_bh_lb_min"]
    out["cumulative_sand_lbm"] = out["cum_sand_surface_lb"]
    out["cumulative_bh_sand_lbm"] = out["cum_sand_bh_lb"]
    return out


if __name__ == "__main__":
    try:
        from src.treatment_schedule import TreatmentScheduleConfig, generate_treatment_schedule
    except ImportError:
        from treatment_schedule import TreatmentScheduleConfig, generate_treatment_schedule

    schedule = generate_treatment_schedule(
        TreatmentScheduleConfig(
            duration_min=60.0,
            dt_min=0.25,
            target_rate_bpm=80.0,
            max_ppa=2.0,
        )
    )
    sand = apply_sand_transport(schedule)
    print(
        sand[
            [
                "time_min",
                "ppa",
                "bottomhole_ppa",
                "sand_lag_min",
                "sand_rate_surface_lb_min",
                "sand_rate_bh_lb_min",
                "sand_arrived_at_perfs",
                "flush_arrived_at_perfs",
            ]
        ].head(30)
    )
    print(get_first_arrival_times(sand))
