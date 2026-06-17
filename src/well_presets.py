"""Field-style well presets for the frac monitoring simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WellPreset:
    """Preset values used to initialize a training simulation."""

    name: str
    duration_min: float
    target_rate_bpm: float
    max_ppa: float
    tvd_ft: float
    measured_depth_ft: float
    casing_id_in: float
    base_fluid_density_ppg: float
    closure_gradient_psi_ft: float
    wellbore_capacity_bbl_per_ft: float
    mixing_efficiency: float
    number_of_pumps: int
    pump_hhp: float
    pump_efficiency: float
    max_treating_pressure_psi: float
    rate_capacity_bpm: float
    description: str


WELL_PRESETS: dict[str, WellPreset] = {
    "Custom": WellPreset(
        name="Custom",
        duration_min=60.0,
        target_rate_bpm=80.0,
        max_ppa=2.0,
        tvd_ft=8500.0,
        measured_depth_ft=15000.0,
        casing_id_in=5.0,
        base_fluid_density_ppg=8.6,
        closure_gradient_psi_ft=0.72,
        wellbore_capacity_bbl_per_ft=0.020,
        mixing_efficiency=0.75,
        number_of_pumps=14,
        pump_hhp=2500.0,
        pump_efficiency=0.88,
        max_treating_pressure_psi=9000.0,
        rate_capacity_bpm=105.0,
        description="Manual controls for sandbox training.",
    ),
    "Permian Horizontal Stage": WellPreset(
        name="Permian Horizontal Stage",
        duration_min=70.0,
        target_rate_bpm=90.0,
        max_ppa=3.0,
        tvd_ft=9500.0,
        measured_depth_ft=18500.0,
        casing_id_in=5.0,
        base_fluid_density_ppg=8.8,
        closure_gradient_psi_ft=0.78,
        wellbore_capacity_bbl_per_ft=0.021,
        mixing_efficiency=0.75,
        number_of_pumps=16,
        pump_hhp=2500.0,
        pump_efficiency=0.88,
        max_treating_pressure_psi=9500.0,
        rate_capacity_bpm=105.0,
        description="High-rate horizontal stage with field-style sand lag.",
    ),
    "Delaware High Pressure Stage": WellPreset(
        name="Delaware High Pressure Stage",
        duration_min=75.0,
        target_rate_bpm=85.0,
        max_ppa=2.5,
        tvd_ft=10500.0,
        measured_depth_ft=20500.0,
        casing_id_in=4.78,
        base_fluid_density_ppg=9.1,
        closure_gradient_psi_ft=0.84,
        wellbore_capacity_bbl_per_ft=0.022,
        mixing_efficiency=0.70,
        number_of_pumps=16,
        pump_hhp=2500.0,
        pump_efficiency=0.86,
        max_treating_pressure_psi=10000.0,
        rate_capacity_bpm=100.0,
        description="Higher TVD and closure gradient; pressure discipline matters.",
    ),
    "Training Mini Stage": WellPreset(
        name="Training Mini Stage",
        duration_min=50.0,
        target_rate_bpm=65.0,
        max_ppa=2.0,
        tvd_ft=7500.0,
        measured_depth_ft=11500.0,
        casing_id_in=5.5,
        base_fluid_density_ppg=8.6,
        closure_gradient_psi_ft=0.70,
        wellbore_capacity_bbl_per_ft=0.018,
        mixing_efficiency=0.80,
        number_of_pumps=10,
        pump_hhp=2250.0,
        pump_efficiency=0.86,
        max_treating_pressure_psi=8500.0,
        rate_capacity_bpm=80.0,
        description="Shorter stage for quick diagnostic practice.",
    ),
}


def get_preset(name: str) -> WellPreset:
    """Return a preset by name, falling back to Custom."""
    return WELL_PRESETS.get(name, WELL_PRESETS["Custom"])
