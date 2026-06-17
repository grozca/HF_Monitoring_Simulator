"""
Simplified hydraulic-fracturing physics engine for the Frac Monitoring Training Simulator.

This is a training engine, not a calibrated commercial frac model. It produces
realistic-looking monitoring curves for practicing interpretation:

    Surface Treating Pressure
    Slurry Rate / Clean Rate
    PPA
    Sand Rate
    Pipe Friction
    Perforation Friction
    Estimated BHP
    Net Pressure
    Alarms / automatic diagnosis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd

from src.equipment_engine import EquipmentConfig, apply_equipment_limits
from src.formation_engine import FormationConfig, apply_formation_engine
from src.sand_transport import SandTransportConfig, apply_sand_transport
from src.wellbore_hydraulics import WellboreHydraulicsConfig, apply_wellbore_hydraulics


ScenarioName = Literal[
    "Normal Job",
    "Screenout",
    "Perforation Plugging",
    "Pump Issue",
    "Frac Hit",
    "Sensor Error",
]


@dataclass(frozen=True)
class WellConfig:
    """Basic well and formation configuration."""

    tvd_ft: float = 8500.0
    measured_depth_ft: float = 15000.0
    casing_id_in: float = 5.0
    wellbore_capacity_bbl_per_ft: float = 0.020
    closure_gradient_psi_ft: float = 0.72
    base_fluid_density_ppg: float = 8.6
    pipe_friction_at_target_psi: float = 1450.0
    perf_friction_at_target_psi: float = 450.0
    base_net_pressure_psi: float = 520.0
    surface_pressure_limit_psi: float = 9000.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Empirical tuning knobs for the training simulator."""

    pipe_rate_exponent: float = 1.85
    perf_rate_exponent: float = 2.00
    pipe_ppa_multiplier: float = 0.14
    perf_ppa_multiplier: float = 0.07
    ppa_density_gain_ppg: float = 0.42
    noise_std_psi: float = 12.0
    breakdown_spike_psi: float = 950.0


def get_phase_boundaries(duration_min: float) -> Dict[str, float]:
    return {
        "ramp_up": 0.04 * duration_min,
        "pad": 0.18 * duration_min,
        "breakdown": 0.22 * duration_min,
        "slurry_ramp": 0.45 * duration_min,
        "main_prop": 0.85 * duration_min,
        "flush": duration_min,
    }


def field_hydrostatic_pressure_psi(
    density_ppg: pd.Series | np.ndarray | float,
    tvd_ft: float,
) -> pd.Series | np.ndarray | float:
    """Calculate hydrostatic pressure using field units: psi = 0.052 * ppg * ft."""
    return 0.052 * density_ppg * tvd_ft


def calculate_slurry_properties(
    df: pd.DataFrame,
    base_fluid_density_ppg: float = 8.6,
    ppa_density_gain_ppg: float = 0.42,
) -> pd.DataFrame:
    """Add clean rate, sand rate, cumulative volume/sand, and slurry density."""
    required = {"time_min", "slurry_rate_bpm", "ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for slurry calculations: {sorted(missing)}")

    out = df.copy()
    out["surface_ppa"] = out.get("surface_ppa", out["ppa"]).astype(float)
    out["bottomhole_ppa"] = out.get("bottomhole_ppa", out["surface_ppa"]).astype(float)
    out["formation_ppa"] = out["bottomhole_ppa"]

    out["clean_rate_bpm"] = out["slurry_rate_bpm"] / (1.0 + 0.045 * out["surface_ppa"])
    out["sand_rate_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["surface_ppa"]

    if "sand_rate_bh_lb_min" not in out.columns:
        out["sand_rate_bh_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["bottomhole_ppa"]

    time_values = out["time_min"].to_numpy()
    dt = np.ones(len(time_values), dtype=float) if len(time_values) <= 1 else np.gradient(time_values)
    out["cum_slurry_bbl"] = np.cumsum(out["slurry_rate_bpm"].to_numpy() * dt)
    out["cum_sand_lb"] = np.cumsum(out["sand_rate_lb_min"].to_numpy() * dt)
    out["slurry_density_ppg"] = base_fluid_density_ppg + ppa_density_gain_ppg * out["surface_ppa"]

    out["sand_rate_lbm_min"] = out["sand_rate_lb_min"]
    out["sand_rate_bh_lbm_min"] = out["sand_rate_bh_lb_min"]
    out["cumulative_slurry_bbl"] = out["cum_slurry_bbl"]
    out["cumulative_sand_lbm"] = out["cum_sand_lb"]
    if "cum_sand_bh_lb" not in out.columns:
        out["cum_sand_bh_lb"] = np.cumsum(out["sand_rate_bh_lb_min"].to_numpy() * dt)
    out["cumulative_bh_sand_lbm"] = out["cum_sand_bh_lb"]
    return out


def apply_rate_scenario(
    df: pd.DataFrame,
    scenario: ScenarioName,
    duration_min: float,
    severity: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Modify slurry rate for equipment/rate-driven scenarios."""
    out = df.copy()
    out["event"] = "Normal"
    rng = np.random.default_rng(seed)
    strength = float(np.clip(severity, 0.5, 2.0))

    if scenario == "Pump Issue":
        onset = 0.58 * duration_min
        mask = out["time_min"] >= onset
        t = out.loc[mask, "time_min"].to_numpy()
        if len(t):
            disturbance = 1.0 - (0.10 + 0.07 * strength) * np.sin((t - onset) * 1.7)
            gradual_drop = 1.0 - (0.05 + 0.05 * strength) * ((t - onset) / max(duration_min - onset, 1.0))
            noise = rng.normal(0.0, 0.012 * strength, len(t))
            out.loc[mask, "slurry_rate_bpm"] *= np.clip(
                disturbance * gradual_drop + noise,
                0.55,
                1.08,
            )
            out.loc[mask, "event"] = "Pump Rate Loss"

    return out


def calculate_hydraulics(
    df: pd.DataFrame,
    well: WellConfig | None = None,
    physics: PhysicsConfig | None = None,
    formation: FormationConfig | None = None,
    equipment: EquipmentConfig | None = None,
    *,
    target_rate_bpm: float = 80.0,
    duration_min: float = 60.0,
    scenario: ScenarioName = "Normal Job",
    severity: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Calculate simplified frac-monitoring hydraulic curves."""
    well = well or WellConfig()
    physics = physics or PhysicsConfig()
    required = {"time_min", "slurry_rate_bpm", "ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for hydraulics: {sorted(missing)}")

    out = calculate_slurry_properties(
        df.copy(),
        base_fluid_density_ppg=well.base_fluid_density_ppg,
        ppa_density_gain_ppg=physics.ppa_density_gain_ppg,
    )
    if "event" not in out.columns:
        out["event"] = "Normal"

    rng = np.random.default_rng(seed)
    strength = float(np.clip(severity, 0.5, 2.0))

    rate_ratio = np.clip(out["slurry_rate_bpm"] / max(target_rate_bpm, 1.0), 0.0, 2.0)
    surface_ppa = out["surface_ppa"]
    bottomhole_ppa = out["bottomhole_ppa"]

    out = apply_wellbore_hydraulics(
        out,
        WellboreHydraulicsConfig(
            measured_depth_ft=well.measured_depth_ft,
            tvd_ft=well.tvd_ft,
            casing_id_in=well.casing_id_in,
            wellbore_capacity_bbl_per_ft=well.wellbore_capacity_bbl_per_ft,
            base_fluid_density_ppg=well.base_fluid_density_ppg,
            target_rate_bpm=target_rate_bpm,
            base_pipe_friction_at_target_psi=well.pipe_friction_at_target_psi,
            pipe_rate_exponent=physics.pipe_rate_exponent,
            pipe_ppa_multiplier=physics.pipe_ppa_multiplier,
        ),
    )
    out["perf_friction_psi"] = (
        well.perf_friction_at_target_psi
        * np.power(rate_ratio, physics.perf_rate_exponent)
        * (1.0 + physics.perf_ppa_multiplier * bottomhole_ppa)
    )

    t = out["time_min"].to_numpy()
    boundaries = get_phase_boundaries(duration_min)
    growth_term = 22.0 * np.sqrt(np.maximum(t, 0.0))
    slurry_term = 110.0 * bottomhole_ppa.to_numpy()
    volume_term = 0.035 * np.power(np.maximum(out["cum_slurry_bbl"].to_numpy(), 1.0), 0.74)

    net_pressure = well.base_net_pressure_psi + growth_term + slurry_term + volume_term
    spike_center = boundaries["pad"] + 0.5 * (boundaries["breakdown"] - boundaries["pad"])
    spike_width = max(0.65, 0.012 * duration_min)
    breakdown_spike = physics.breakdown_spike_psi * np.exp(
        -0.5 * np.square((t - spike_center) / spike_width)
    )
    net_pressure += breakdown_spike
    net_pressure += rng.normal(0.0, physics.noise_std_psi, len(out))

    out["closure_pressure_psi"] = well.closure_gradient_psi_ft * well.tvd_ft
    out["tvd_ft"] = well.tvd_ft
    out["measured_depth_ft"] = well.measured_depth_ft
    out["casing_id_in"] = well.casing_id_in
    out["base_fluid_density_ppg"] = well.base_fluid_density_ppg
    out["net_pressure_psi"] = net_pressure

    if scenario == "Screenout":
        onset = 0.68 * duration_min
        mask = (out["time_min"] >= onset) & (out["bottomhole_ppa"] > 0.05)
        tau = (out.loc[mask, "time_min"] - onset) / max(duration_min - onset, 1.0)
        extra_net = (850.0 + 950.0 * strength) * np.power(np.clip(tau, 0, 1), 1.65)
        out.loc[mask, "net_pressure_psi"] += extra_net.to_numpy()
        out.loc[mask, "event"] = "Screenout Trend"

    elif scenario == "Perforation Plugging":
        onset = 0.52 * duration_min
        mask = (out["time_min"] >= onset) & (out["bottomhole_ppa"] > 0.05)
        tau = (out.loc[mask, "time_min"] - onset) / max(duration_min - onset, 1.0)
        extra_perf = (650.0 + 700.0 * strength) * np.power(np.clip(tau, 0, 1), 1.40)
        out.loc[mask, "perf_friction_psi"] += extra_perf.to_numpy()
        out.loc[mask, "event"] = "Perforation Restriction"

    elif scenario == "Frac Hit":
        onset = 0.62 * duration_min
        mask = out["time_min"] >= onset
        tau = (out.loc[mask, "time_min"] - onset) / max(duration_min - onset, 1.0)
        pressure_sink = (500.0 + 350.0 * strength) * (1.0 - np.exp(-8.0 * tau))
        out.loc[mask, "net_pressure_psi"] -= pressure_sink.to_numpy()
        out.loc[mask, "event"] = "Frac Hit"

    out = apply_formation_engine(
        out,
        config=formation,
        scenario=scenario,
        severity=strength,
    )
    out["net_pressure_psi"] += out["formation_pressure_adjustment_psi"]

    out["bhp_psi"] = out["closure_pressure_psi"] + out["net_pressure_psi"]
    out["bhp_gradient_psi_ft"] = out["bhp_psi"] / max(well.tvd_ft, 1.0)
    out["surface_pressure_psi"] = (
        out["bhp_psi"]
        - out["hydrostatic_psi"]
        + out["pipe_friction_psi"]
        + out["perf_friction_psi"]
    )

    if scenario == "Sensor Error":
        onset = 0.45 * duration_min
        mask = out["time_min"] >= onset
        n = int(mask.sum())
        if n:
            sensor_noise = rng.normal(0.0, 25.0, n)
            spike_count = max(2, n // 18)
            spike_indices = rng.choice(np.arange(n), size=spike_count, replace=False)
            sensor_noise[spike_indices] += rng.choice([-1, 1], size=spike_count) * rng.uniform(
                350.0,
                900.0 * strength,
                spike_count,
            )
            out.loc[mask, "surface_pressure_psi"] += sensor_noise
            out.loc[mask, "event"] = "Pressure Sensor Error"

    out = apply_equipment_limits(out, equipment)
    out["treating_pressure_limit_psi"] = out["max_treating_pressure_psi"]

    out["estimated_bhp_psi"] = (
        out["surface_pressure_psi"]
        + out["hydrostatic_psi"]
        - out["pipe_friction_psi"]
        - out["perf_friction_psi"]
    )
    out["estimated_net_pressure_psi"] = out["estimated_bhp_psi"] - out["closure_pressure_psi"]
    return out


def add_diagnostics(
    df: pd.DataFrame,
    *,
    duration_min: float = 60.0,
    trend_window_min: float = 3.0,
) -> pd.DataFrame:
    """Add real-time alarms and automatic diagnosis."""
    required = {
        "time_min",
        "surface_pressure_psi",
        "slurry_rate_bpm",
        "rate_bph",
        "clean_rate_bpm",
        "net_pressure_psi",
        "perf_friction_psi",
        "ppa",
        "event",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for diagnostics: {sorted(missing)}")

    out = df.copy()
    mean_dt = max(float(np.mean(np.diff(out["time_min"]))), 0.01)
    window = max(3, int(round(trend_window_min / mean_dt)))
    dt = out["time_min"].diff(window).replace(0.0, np.nan)

    out["pressure_slope_psi_min"] = out["surface_pressure_psi"].diff(window).div(dt)
    out["rate_slope_bpm_min"] = out["slurry_rate_bpm"].diff(window).div(dt)
    out["net_pressure_slope_psi_min"] = out["net_pressure_psi"].diff(window).div(dt)
    out["perf_friction_slope_psi_min"] = out["perf_friction_psi"].diff(window).div(dt)

    slope_cols = [
        "pressure_slope_psi_min",
        "rate_slope_bpm_min",
        "net_pressure_slope_psi_min",
        "perf_friction_slope_psi_min",
    ]
    out[slope_cols] = out[slope_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    alarms: list[str] = []
    diagnoses: list[str] = []

    for _, row in out.iterrows():
        alarm = "OK"
        diagnosis = "Normal treatment response"
        slurry_on = row.get("bottomhole_ppa", row["ppa"]) > 0.40
        screenout_risk = float(row.get("screenout_risk", 0.0))

        if row["event"] == "Screenout Trend" and slurry_on and (
            row["net_pressure_slope_psi_min"] > 25.0 or row["net_pressure_psi"] > 2050.0
        ):
            alarm = "HIGH PRESSURE TREND"
            diagnosis = "Possible screenout"

        if row["event"] == "Perforation Restriction" and slurry_on and (
            row["perf_friction_slope_psi_min"] > 18.0 or row["perf_friction_psi"] > 900.0
        ):
            alarm = "PERF FRICTION INCREASE"
            diagnosis = "Possible perforation or cluster restriction"

        if row["event"] == "Pump Rate Loss" and abs(row["rate_slope_bpm_min"]) > 2.5:
            alarm = "RATE INSTABILITY"
            diagnosis = "Pump or rate-control issue"

        if (
            row["event"] == "Frac Hit"
            and row["pressure_slope_psi_min"] < -25.0
            and row["time_min"] > 0.35 * duration_min
        ):
            alarm = "PRESSURE DROP"
            diagnosis = "Possible frac hit / pressure communication"

        if row["event"] == "Pressure Sensor Error" and abs(row["pressure_slope_psi_min"]) > 60.0:
            alarm = "DATA QUALITY"
            diagnosis = "Possible sensor or data quality issue"

        if row["surface_pressure_psi"] > row["treating_pressure_limit_psi"]:
            alarm = "PRESSURE LIMIT"
            diagnosis = "Pressure above treating limit"

        if row.get("equipment_status", "Available") == "HHP limited":
            alarm = "HHP LIMIT"
            diagnosis = "Hydraulic horsepower demand exceeds available spread"

        if row.get("equipment_status", "Available") == "Rate limited":
            alarm = "RATE CAPACITY"
            diagnosis = "Requested rate is above equipment capacity"

        if row.get("equipment_status", "Available") == "Pressure limited":
            alarm = "PRESSURE LIMIT"
            diagnosis = "Pressure above treating limit"

        if row["net_pressure_psi"] > 2400.0:
            alarm = "HIGH NET PRESSURE"
            diagnosis = "Possible screenout"

        if slurry_on and screenout_risk > 0.85 and row["net_pressure_slope_psi_min"] > 16.0:
            alarm = "HIGH SCREENOUT RISK"
            diagnosis = "Possible screenout"

        if row["perf_friction_psi"] > 1700.0:
            alarm = "PERF FRICTION HIGH"
            diagnosis = "Possible perforation or cluster restriction"

        alarms.append(alarm)
        diagnoses.append(diagnosis)

    out["alarm"] = alarms
    out["engineer_diagnosis"] = diagnoses

    warning = (
        (out["alarm"] != "OK")
        | (out["event"] != "Normal")
        | (out["net_pressure_psi"] > 2100.0)
        | (out["perf_friction_psi"] > 1350.0)
        | (out.get("screenout_risk", 0.0) > 0.75)
        | (out.get("equipment_status", "Available") == "Near limit")
    )
    critical = (
        out["alarm"].isin(
            [
                "PRESSURE LIMIT",
                "HHP LIMIT",
                "RATE CAPACITY",
                "HIGH NET PRESSURE",
                "HIGH SCREENOUT RISK",
                "PERF FRICTION HIGH",
            ]
        )
        | (out["surface_pressure_psi"] > out["treating_pressure_limit_psi"])
        | (out["net_pressure_psi"] > 2400.0)
        | (out["perf_friction_psi"] > 1700.0)
        | (out.get("hhp_utilization", 0.0) >= 1.0)
    )
    out["diagnostic_status"] = np.select(
        [critical, warning],
        ["Critical", "Warning"],
        default="Normal",
    )

    numeric_cols = [
        "slurry_rate_bpm",
        "clean_rate_bpm",
        "ppa",
        "surface_ppa",
        "bottomhole_ppa",
        "formation_ppa",
        "sand_lag_min",
        "ppa_lag_delta",
        "sand_rate_lb_min",
        "sand_rate_lbm_min",
        "sand_rate_surface_lb_min",
        "sand_rate_bh_lb_min",
        "sand_rate_bh_lbm_min",
        "cum_sand_surface_lb",
        "cum_sand_bh_lb",
        "sand_in_wellbore_lb",
        "sand_in_fracture_lb",
        "cum_slurry_bbl",
        "cum_sand_lb",
        "cumulative_slurry_bbl",
        "cumulative_sand_lbm",
        "cumulative_bh_sand_lbm",
        "measured_depth_ft",
        "tvd_ft",
        "casing_id_in",
        "wellbore_capacity_bbl_per_ft",
        "slurry_density_ppg",
        "hydrostatic_psi",
        "reynolds_proxy",
        "friction_factor_proxy",
        "pipe_friction_psi",
        "perf_friction_psi",
        "surface_pressure_psi",
        "treating_pressure_limit_psi",
        "bhp_psi",
        "bhp_gradient_psi_ft",
        "estimated_bhp_psi",
        "net_pressure_psi",
        "estimated_net_pressure_psi",
        "fracture_width_in",
        "fracture_width_proxy_in",
        "fracture_half_length_ft",
        "fracture_length_ft",
        "fracture_height_ft",
        "leakoff_rate_bpm",
        "cumulative_leakoff_bbl",
        "leakoff_bbl",
        "fluid_efficiency",
        "acceptance_index",
        "formation_acceptance_pct",
        "screenout_risk",
        "screenout_risk_pct",
        "formation_pressure_adjustment_psi",
        "number_of_pumps",
        "pump_hhp",
        "pump_efficiency",
        "available_hhp",
        "max_treating_pressure_psi",
        "rate_capacity_bpm",
        "hhp_required",
        "hhp_utilization",
        "pressure_margin_psi",
        "rate_margin_bpm",
        "pressure_slope_psi_min",
        "rate_slope_bpm_min",
        "net_pressure_slope_psi_min",
        "perf_friction_slope_psi_min",
    ]
    existing_numeric_cols = [col for col in numeric_cols if col in out.columns]
    out[existing_numeric_cols] = out[existing_numeric_cols].round(2)
    return out


def run_physics_engine(
    schedule_df: pd.DataFrame,
    *,
    scenario: ScenarioName = "Normal Job",
    target_rate_bpm: float = 80.0,
    duration_min: float = 60.0,
    severity: float = 1.0,
    seed: int = 7,
    well: WellConfig | None = None,
    physics: PhysicsConfig | None = None,
    sand_transport: SandTransportConfig | None = None,
    formation: FormationConfig | None = None,
    equipment: EquipmentConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: rate scenario -> hydraulics -> diagnostics."""
    well = well or WellConfig()
    df = apply_rate_scenario(
        schedule_df,
        scenario=scenario,
        duration_min=duration_min,
        severity=severity,
        seed=seed,
    )
    df = apply_sand_transport(
        df,
        sand_transport or SandTransportConfig(measured_depth_ft=well.tvd_ft),
    )
    df = calculate_hydraulics(
        df,
        well=well,
        physics=physics,
        formation=formation,
        equipment=equipment,
        target_rate_bpm=target_rate_bpm,
        duration_min=duration_min,
        scenario=scenario,
        severity=severity,
        seed=seed,
    )
    return add_diagnostics(df, duration_min=duration_min)


def hydrostatic_pressure_psi(config: WellConfig) -> float:
    """Compatibility helper for older code paths."""
    return float(field_hydrostatic_pressure_psi(config.base_fluid_density_ppg, config.tvd_ft))


def apply_pressure_model(schedule: pd.DataFrame, config: WellConfig | None = None) -> pd.DataFrame:
    """Compatibility wrapper for earlier app versions."""
    target_rate = float(schedule["slurry_rate_bpm"].max()) if "slurry_rate_bpm" in schedule else 80.0
    duration = float(schedule["time_min"].max()) if "time_min" in schedule else 60.0
    return run_physics_engine(
        schedule,
        target_rate_bpm=target_rate,
        duration_min=duration,
        well=config,
    )
