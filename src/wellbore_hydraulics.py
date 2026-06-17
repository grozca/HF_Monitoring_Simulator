"""Wellbore hydraulic helper calculations for the training simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WellboreHydraulicsConfig:
    """Wellbore geometry and simple hydraulic constants."""

    measured_depth_ft: float = 15000.0
    tvd_ft: float = 8500.0
    casing_id_in: float = 5.0
    wellbore_capacity_bbl_per_ft: float = 0.020
    base_fluid_density_ppg: float = 8.6
    target_rate_bpm: float = 80.0
    base_pipe_friction_at_target_psi: float = 1450.0
    pipe_rate_exponent: float = 1.85
    pipe_ppa_multiplier: float = 0.14


def hydrostatic_pressure_psi(density_ppg: pd.Series | float, tvd_ft: float) -> pd.Series | float:
    """Hydrostatic pressure in field units."""
    return 0.052 * density_ppg * tvd_ft


def calculate_rate_bph(rate_bpm: pd.Series | float) -> pd.Series | float:
    """Convert barrels per minute to barrels per hour."""
    return rate_bpm * 60.0


def calculate_annular_velocity_ft_min(rate_bpm: pd.Series, casing_id_in: float) -> pd.Series:
    """Approximate pipe velocity from rate and casing ID."""
    area_ft2 = np.pi * (max(casing_id_in, 0.1) / 12.0) ** 2 / 4.0
    flow_ft3_min = rate_bpm.astype(float) * 5.615
    return flow_ft3_min / max(area_ft2, 0.001)


def calculate_reynolds_proxy(
    rate_bpm: pd.Series,
    density_ppg: pd.Series,
    casing_id_in: float,
) -> pd.Series:
    """Dimensionless training proxy, not a rigorous Reynolds number."""
    velocity = calculate_annular_velocity_ft_min(rate_bpm, casing_id_in)
    density_factor = density_ppg.astype(float) / 8.6
    id_factor = max(casing_id_in, 0.1) / 5.0
    reynolds = 9000.0 * (velocity / 900.0) * density_factor * id_factor
    return pd.Series(reynolds, index=rate_bpm.index, name="reynolds_proxy").clip(lower=0.0)


def calculate_friction_factor_proxy(reynolds_proxy: pd.Series) -> pd.Series:
    """Simple turbulent-friction trend proxy."""
    reynolds = reynolds_proxy.astype(float).clip(lower=1.0)
    friction = 0.018 + 0.018 * np.power(10000.0 / reynolds, 0.18)
    return pd.Series(friction, index=reynolds_proxy.index, name="friction_factor_proxy").clip(0.012, 0.080)


def calculate_pipe_friction_psi(
    rate_bpm: pd.Series,
    slurry_density_ppg: pd.Series,
    surface_ppa: pd.Series,
    config: WellboreHydraulicsConfig,
) -> pd.Series:
    """Empirical pipe friction with geometry and density modifiers."""
    rate_ratio = np.clip(rate_bpm.astype(float) / max(config.target_rate_bpm, 1.0), 0.0, 2.2)
    density_ratio = slurry_density_ppg.astype(float) / max(config.base_fluid_density_ppg, 0.1)
    id_modifier = np.power(5.0 / max(config.casing_id_in, 0.1), 4.8)
    md_modifier = config.measured_depth_ft / 15000.0
    capacity_modifier = np.power(0.020 / max(config.wellbore_capacity_bbl_per_ft, 0.001), 0.20)

    pipe_friction = (
        config.base_pipe_friction_at_target_psi
        * np.power(rate_ratio, config.pipe_rate_exponent)
        * density_ratio
        * (1.0 + config.pipe_ppa_multiplier * surface_ppa.astype(float))
        * id_modifier
        * md_modifier
        * capacity_modifier
    )
    return pd.Series(pipe_friction, index=rate_bpm.index, name="pipe_friction_psi").clip(lower=0.0)


def apply_wellbore_hydraulics(
    df: pd.DataFrame,
    config: WellboreHydraulicsConfig,
) -> pd.DataFrame:
    """Add wellbore hydraulic helper columns and pipe friction."""
    required = {"slurry_rate_bpm", "slurry_density_ppg", "surface_ppa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for wellbore hydraulics: {sorted(missing)}")

    out = df.copy()
    out["measured_depth_ft"] = config.measured_depth_ft
    out["tvd_ft"] = config.tvd_ft
    out["casing_id_in"] = config.casing_id_in
    out["wellbore_capacity_bbl_per_ft"] = config.wellbore_capacity_bbl_per_ft
    out["rate_bph"] = calculate_rate_bph(out["slurry_rate_bpm"])
    out["hydrostatic_psi"] = hydrostatic_pressure_psi(out["slurry_density_ppg"], config.tvd_ft)
    out["reynolds_proxy"] = calculate_reynolds_proxy(
        out["slurry_rate_bpm"],
        out["slurry_density_ppg"],
        config.casing_id_in,
    )
    out["friction_factor_proxy"] = calculate_friction_factor_proxy(out["reynolds_proxy"])
    out["pipe_friction_psi"] = calculate_pipe_friction_psi(
        out["slurry_rate_bpm"],
        out["slurry_density_ppg"],
        out["surface_ppa"],
        config,
    )
    return out
