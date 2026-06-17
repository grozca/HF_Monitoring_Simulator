"""Surface equipment and hydraulic horsepower calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquipmentConfig:
    """Simplified frac spread limits."""

    number_of_pumps: int = 14
    pump_hhp: float = 2500.0
    pump_efficiency: float = 0.88
    max_treating_pressure_psi: float = 9000.0
    rate_capacity_bpm: float = 105.0


def calculate_available_hhp(config: EquipmentConfig) -> float:
    """Available hydraulic horsepower after efficiency."""
    return config.number_of_pumps * config.pump_hhp * config.pump_efficiency


def calculate_required_hhp(surface_pressure_psi: pd.Series, slurry_rate_bpm: pd.Series) -> pd.Series:
    """HHP ~= pressure * rate / 40.8."""
    hhp = surface_pressure_psi.astype(float) * slurry_rate_bpm.astype(float) / 40.8
    return pd.Series(hhp, index=surface_pressure_psi.index, name="hhp_required").clip(lower=0.0)


def apply_equipment_limits(
    df: pd.DataFrame,
    config: EquipmentConfig | None = None,
) -> pd.DataFrame:
    """Add equipment utilization, pressure margin, and limit status columns."""
    config = config or EquipmentConfig()
    required = {"surface_pressure_psi", "slurry_rate_bpm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for equipment calculations: {sorted(missing)}")

    out = df.copy()
    available_hhp = calculate_available_hhp(config)
    out["number_of_pumps"] = int(config.number_of_pumps)
    out["pump_hhp"] = float(config.pump_hhp)
    out["pump_efficiency"] = float(config.pump_efficiency)
    out["available_hhp"] = float(available_hhp)
    out["max_treating_pressure_psi"] = float(config.max_treating_pressure_psi)
    out["rate_capacity_bpm"] = float(config.rate_capacity_bpm)
    out["hhp_required"] = calculate_required_hhp(out["surface_pressure_psi"], out["slurry_rate_bpm"])
    out["hhp_utilization"] = (out["hhp_required"] / max(available_hhp, 1.0)).clip(lower=0.0)
    out["pressure_margin_psi"] = config.max_treating_pressure_psi - out["surface_pressure_psi"]
    out["rate_margin_bpm"] = config.rate_capacity_bpm - out["slurry_rate_bpm"]

    pressure_limited = out["pressure_margin_psi"] <= 0.0
    hhp_limited = out["hhp_utilization"] >= 1.0
    rate_limited = out["rate_margin_bpm"] <= 0.0
    warning = (
        (out["pressure_margin_psi"] < 500.0)
        | (out["hhp_utilization"] > 0.90)
        | (out["rate_margin_bpm"] < 5.0)
    )

    out["equipment_status"] = np.select(
        [pressure_limited, hhp_limited, rate_limited, warning],
        ["Pressure limited", "HHP limited", "Rate limited", "Near limit"],
        default="Available",
    )
    return out
