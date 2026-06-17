"""
src/calibration_engine.py

Calibration engine for the Frac Monitoring Training Simulator.

What it does
------------
Compares simulated frac curves against reference data from Excel/CSV/base model.
It aligns time, optionally estimates time lag, calculates engineering metrics,
and can tune simulator parameters with a derivative-free coordinate search.

Core math
---------
Residual:
    e_i = y_sim(t_i) - y_ref(t_i)

Weighted RMSE:
    RMSE_w = sqrt( sum(w_i * e_i^2) / sum(w_i) )

Normalized RMSE:
    NRMSE = RMSE / (P95(y_ref) - P5(y_ref))

Slope error:
    slope_error = RMSE( d(y_sim)/dt - d(y_ref)/dt )

Lag estimate:
    lag* = argmax_lag corr( y_sim(t + lag), y_ref(t) )

Objective:
    J(theta) = sum(variable_weight * metric_weight * metric_value)

Where theta is a set of simulator parameters such as:
    pipe_friction_at_target_psi
    perf_friction_at_target_psi
    closure_gradient_psi_ft
    base_net_pressure_psi
    scenario_severity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class CalibrationConfig:
    time_col: str = "time_min"
    dt_min: float = 0.25
    max_lag_min: float = 8.0
    apply_lag_correction: bool = True
    smoothing_window: int = 1
    min_points: int = 10
    fallback_pressure_scale_psi: float = 1000.0


@dataclass(frozen=True)
class CalibrationWeights:
    variable_weights: Dict[str, float] = field(default_factory=lambda: {
        "surface_pressure_psi": 1.00,
        "bhp_psi": 0.80,
        "net_pressure_psi": 1.00,
        "slurry_rate_bpm": 0.25,
        "ppa": 0.25,
        "bottomhole_ppa": 0.35,
        "pipe_friction_psi": 0.45,
        "perf_friction_psi": 0.45,
        "screenout_risk": 0.30,
    })
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "nrmse": 1.00,
        "bias_norm": 0.30,
        "slope_nrmse": 0.80,
        "corr_penalty": 0.50,
        "max_abs_norm": 0.20,
    })


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float
    step: float


@dataclass
class CalibrationResult:
    best_params: Dict[str, float]
    best_objective: float
    history: pd.DataFrame
    metrics: pd.DataFrame
    aligned: pd.DataFrame
    lag_min: float
    notes: list[str] = field(default_factory=list)


# =============================================================================
# Column mapping / standardization
# =============================================================================

STANDARD_ALIASES: Dict[str, list[str]] = {
    "time_min": ["time", "time min", "time_min", "minutes", "min", "elapsed time", "t"],
    "surface_pressure_psi": [
        "surface pressure", "surface_pressure", "surface pressure psi", "treating pressure",
        "treating_pressure", "treating pressure psi", "stp", "pressure", "pressure psi",
        "surf pressure",
    ],
    "bhp_psi": ["bhp", "bottomhole pressure", "bottomhole_pressure", "bhp psi"],
    "net_pressure_psi": ["net pressure", "net_pressure", "net pressure psi", "net p"],
    "slurry_rate_bpm": ["slurry rate", "slurry_rate", "rate", "rate bpm", "pump rate", "bpm"],
    "clean_rate_bpm": ["clean rate", "clean_rate", "clean bpm"],
    "ppa": ["ppa", "surface ppa", "surf ppa", "prop conc", "proppant concentration", "lb/gal"],
    "bottomhole_ppa": ["bottomhole ppa", "bh ppa", "bottomhole_ppa", "ppa at perfs"],
    "pipe_friction_psi": ["pipe friction", "pipe_friction", "wellbore friction", "tubing friction"],
    "perf_friction_psi": ["perf friction", "perforation friction", "perf_friction"],
    "screenout_risk": ["screenout risk", "risk", "screenout_risk"],
}


def _clean_name(name: Any) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ").replace("\n", " ")
    return " ".join(s.split())


def auto_map_columns(df: pd.DataFrame, aliases: Optional[Dict[str, list[str]]] = None) -> Dict[str, str]:
    """Return mapping {standard_column: raw_column}."""
    aliases = aliases or STANDARD_ALIASES
    raw = {_clean_name(c): c for c in df.columns}
    mapping: Dict[str, str] = {}

    for standard, names in aliases.items():
        candidates = [_clean_name(n) for n in names]
        for c in candidates:
            if c in raw:
                mapping[standard] = raw[c]
                break
        if standard not in mapping:
            for raw_clean, raw_original in raw.items():
                if any(c in raw_clean for c in candidates):
                    mapping[standard] = raw_original
                    break
    return mapping


def _coerce_numeric_if_possible(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    converted = pd.to_numeric(series, errors="coerce")
    valid_fraction = float(converted.notna().mean()) if len(series) else 0.0
    if pd.api.types.is_numeric_dtype(series) or valid_fraction >= 0.50:
        return converted
    return series


def _coerce_time_minutes(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        time_values = pd.to_datetime(series, errors="coerce")
        return (time_values - time_values.min()).dt.total_seconds() / 60.0

    numeric_time = pd.to_numeric(series, errors="coerce")
    if numeric_time.notna().sum() > 0:
        return numeric_time

    parsed_time = pd.to_datetime(series, errors="coerce")
    if parsed_time.notna().sum() > 0:
        return (parsed_time - parsed_time.min()).dt.total_seconds() / 60.0

    return numeric_time


def standardize_dataframe(
    df: pd.DataFrame,
    column_map: Optional[Mapping[str, str]] = None,
    required: Sequence[str] = ("time_min",),
) -> pd.DataFrame:
    """Rename reference/simulation columns to the simulator standard."""
    if df.empty:
        raise ValueError("Cannot standardize an empty DataFrame.")

    column_map = dict(column_map or auto_map_columns(df))
    rename = {raw: std for std, raw in column_map.items() if raw in df.columns}
    out = df.rename(columns=rename).copy()

    for col in out.columns:
        out[col] = _coerce_numeric_if_possible(out[col])

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Auto map found: {column_map}")

    time_col = required[0] if required else "time_min"
    if time_col in out.columns:
        out[time_col] = _coerce_time_minutes(out[time_col])
        out = out.dropna(subset=[time_col])

    return out.sort_values(time_col).reset_index(drop=True)


# =============================================================================
# Alignment and interpolation
# =============================================================================


def _smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s.astype(float)
    return s.astype(float).rolling(window=window, center=True, min_periods=1).mean()


def make_time_grid(sim_df: pd.DataFrame, ref_df: pd.DataFrame, cfg: CalibrationConfig, lag_min: float = 0.0) -> np.ndarray:
    sim_t = sim_df[cfg.time_col].to_numpy(float) + lag_min
    ref_t = ref_df[cfg.time_col].to_numpy(float)
    t0 = max(np.nanmin(sim_t), np.nanmin(ref_t))
    t1 = min(np.nanmax(sim_t), np.nanmax(ref_t))
    if t1 <= t0:
        raise ValueError(f"No overlapping time interval for lag={lag_min:.2f} min.")
    return np.arange(t0, t1 + 0.5 * cfg.dt_min, cfg.dt_min)


def interpolate_to_grid(
    df: pd.DataFrame,
    variables: Sequence[str],
    grid: np.ndarray,
    cfg: CalibrationConfig,
    prefix: str,
    lag_min: float = 0.0,
) -> pd.DataFrame:
    t = df[cfg.time_col].to_numpy(float) + lag_min
    out = pd.DataFrame({cfg.time_col: grid})
    for v in variables:
        if v not in df.columns:
            continue
        y = _smooth(df[v], cfg.smoothing_window).to_numpy(float)
        out[f"{prefix}{v}"] = np.interp(grid, t, y, left=np.nan, right=np.nan)
    return out


def align_simulation_to_reference(
    sim_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    variables: Sequence[str],
    cfg: Optional[CalibrationConfig] = None,
    lag_min: float = 0.0,
) -> pd.DataFrame:
    cfg = cfg or CalibrationConfig()
    grid = make_time_grid(sim_df, ref_df, cfg, lag_min=lag_min)
    sim = interpolate_to_grid(sim_df, variables, grid, cfg, prefix="sim_", lag_min=lag_min)
    ref = interpolate_to_grid(ref_df, variables, grid, cfg, prefix="ref_", lag_min=0.0)
    aligned = sim.merge(ref, on=cfg.time_col).dropna().reset_index(drop=True)
    if len(aligned) < cfg.min_points:
        raise ValueError(f"Not enough aligned points: {len(aligned)} < {cfg.min_points}")
    return aligned


# =============================================================================
# Lag estimation
# =============================================================================


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def estimate_lag_by_correlation(
    sim_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    variable: str = "surface_pressure_psi",
    cfg: Optional[CalibrationConfig] = None,
) -> float:
    """Find lag that maximizes correlation between simulated and reference curve."""
    cfg = cfg or CalibrationConfig()
    if variable not in sim_df.columns or variable not in ref_df.columns:
        return 0.0

    best_lag, best_corr = 0.0, -np.inf
    lags = np.arange(-cfg.max_lag_min, cfg.max_lag_min + 0.5 * cfg.dt_min, cfg.dt_min)
    for lag in lags:
        try:
            al = align_simulation_to_reference(sim_df, ref_df, [variable], cfg, lag_min=float(lag))
        except ValueError:
            continue
        corr = _safe_corr(al[f"sim_{variable}"].to_numpy(float), al[f"ref_{variable}"].to_numpy(float))
        if corr > best_corr:
            best_lag, best_corr = float(lag), corr
    return best_lag


# =============================================================================
# Metrics and objective
# =============================================================================


def robust_range(y: np.ndarray, fallback: float = 1.0) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return fallback
    r = float(np.percentile(y, 95) - np.percentile(y, 5))
    return r if np.isfinite(r) and r > 1e-9 else fallback


def calculate_variable_metrics(aligned: pd.DataFrame, variable: str, cfg: Optional[CalibrationConfig] = None) -> Dict[str, float]:
    cfg = cfg or CalibrationConfig()
    s_col, r_col = f"sim_{variable}", f"ref_{variable}"
    if s_col not in aligned.columns or r_col not in aligned.columns:
        raise ValueError(f"Missing aligned columns for {variable}")

    t = aligned[cfg.time_col].to_numpy(float)
    sim = aligned[s_col].to_numpy(float)
    ref = aligned[r_col].to_numpy(float)
    e = sim - ref

    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    max_abs = float(np.max(np.abs(e)))
    scale = robust_range(ref, fallback=cfg.fallback_pressure_scale_psi)

    dsim = np.gradient(sim, t)
    dref = np.gradient(ref, t)
    de = dsim - dref
    slope_rmse = float(np.sqrt(np.mean(de ** 2)))
    slope_scale = robust_range(dref, fallback=scale / max(t[-1] - t[0], 1.0))

    corr = _safe_corr(sim, ref)
    slope_corr = _safe_corr(dsim, dref)
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))

    return {
        "variable": variable,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_abs": max_abs,
        "scale": scale,
        "nrmse": rmse / scale,
        "bias_norm": abs(bias) / scale,
        "max_abs_norm": max_abs / scale,
        "slope_rmse": slope_rmse,
        "slope_nrmse": slope_rmse / max(slope_scale, 1e-9),
        "corr": corr,
        "slope_corr": slope_corr,
        "corr_penalty": 1.0 - max(corr, -1.0),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0,
        "n_points": len(aligned),
    }


def calculate_metrics_table(aligned: pd.DataFrame, variables: Sequence[str], cfg: Optional[CalibrationConfig] = None) -> pd.DataFrame:
    rows = []
    for v in variables:
        if f"sim_{v}" in aligned.columns and f"ref_{v}" in aligned.columns:
            rows.append(calculate_variable_metrics(aligned, v, cfg))
    return pd.DataFrame(rows)


def calculate_objective(metrics: pd.DataFrame, weights: Optional[CalibrationWeights] = None) -> float:
    weights = weights or CalibrationWeights()
    if metrics.empty:
        return float("inf")

    total, total_w = 0.0, 0.0
    for _, row in metrics.iterrows():
        v = str(row["variable"])
        vw = weights.variable_weights.get(v, 0.0)
        if vw <= 0:
            continue
        for metric, mw in weights.metric_weights.items():
            if metric in row and np.isfinite(float(row[metric])):
                total += vw * mw * float(row[metric])
                total_w += vw * mw
    return total / total_w if total_w > 0 else float("inf")


# =============================================================================
# Pressure component fit
# =============================================================================


def fit_surface_pressure_components(
    sim_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    cfg: Optional[CalibrationConfig] = None,
    ref_pressure_col: str = "surface_pressure_psi",
) -> Dict[str, float]:
    """
    Fit linear correction:

        P_ref ~= c0 + c_bhp*BHP - c_hydro*Hydrostatic + c_pipe*PipeFriction + c_perf*PerfFriction

    Use this to see whether the simulator is over/under-estimating a component.
    """
    cfg = cfg or CalibrationConfig()
    needed = ["bhp_psi", "hydrostatic_psi", "pipe_friction_psi", "perf_friction_psi"]
    if ref_pressure_col not in ref_df.columns:
        raise ValueError(f"Reference lacks {ref_pressure_col}")
    for c in needed:
        if c not in sim_df.columns:
            raise ValueError(f"Simulation lacks {c}")

    grid = make_time_grid(sim_df, ref_df, cfg)
    ref = interpolate_to_grid(ref_df, [ref_pressure_col], grid, cfg, prefix="ref_")
    sim = interpolate_to_grid(sim_df, needed, grid, cfg, prefix="sim_")
    data = ref.merge(sim, on=cfg.time_col).dropna()

    y = data[f"ref_{ref_pressure_col}"].to_numpy(float)
    X = np.column_stack([
        np.ones(len(data)),
        data["sim_bhp_psi"].to_numpy(float),
        -data["sim_hydrostatic_psi"].to_numpy(float),
        data["sim_pipe_friction_psi"].to_numpy(float),
        data["sim_perf_friction_psi"].to_numpy(float),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "intercept": float(beta[0]),
        "bhp_coefficient": float(beta[1]),
        "hydrostatic_coefficient": float(beta[2]),
        "pipe_friction_coefficient": float(beta[3]),
        "perf_friction_coefficient": float(beta[4]),
    }


# =============================================================================
# Parameter calibration
# =============================================================================


def _clip_params(params: Dict[str, float], specs: Sequence[ParameterSpec]) -> Dict[str, float]:
    bounds = {s.name: (s.lower, s.upper) for s in specs}
    out = dict(params)
    for name, value in out.items():
        if name in bounds:
            lo, hi = bounds[name]
            out[name] = float(np.clip(value, lo, hi))
    return out


def run_calibration_once(
    model_runner: Callable[[Dict[str, float]], pd.DataFrame],
    params: Dict[str, float],
    reference_df: pd.DataFrame,
    variables: Sequence[str],
    cfg: Optional[CalibrationConfig] = None,
    weights: Optional[CalibrationWeights] = None,
    lag_variable: str = "surface_pressure_psi",
) -> Tuple[float, pd.DataFrame, pd.DataFrame, float]:
    cfg = cfg or CalibrationConfig()
    weights = weights or CalibrationWeights()
    sim_df = model_runner(params)

    lag = 0.0
    if cfg.apply_lag_correction and lag_variable in variables:
        lag = estimate_lag_by_correlation(sim_df, reference_df, variable=lag_variable, cfg=cfg)

    aligned = align_simulation_to_reference(sim_df, reference_df, variables, cfg, lag_min=lag)
    metrics = calculate_metrics_table(aligned, variables, cfg)
    objective = calculate_objective(metrics, weights)
    return objective, metrics, aligned, lag


def coordinate_search_calibration(
    model_runner: Callable[[Dict[str, float]], pd.DataFrame],
    reference_df: pd.DataFrame,
    parameter_specs: Sequence[ParameterSpec],
    variables: Sequence[str],
    cfg: Optional[CalibrationConfig] = None,
    weights: Optional[CalibrationWeights] = None,
    max_iterations: int = 40,
    min_step_fraction: float = 0.03,
    lag_variable: str = "surface_pressure_psi",
) -> CalibrationResult:
    """Simple derivative-free calibration; no SciPy required."""
    cfg = cfg or CalibrationConfig()
    weights = weights or CalibrationWeights()

    params = _clip_params({s.name: s.initial for s in parameter_specs}, parameter_specs)
    steps = {s.name: s.step for s in parameter_specs}
    min_steps = {s.name: abs(s.step) * min_step_fraction for s in parameter_specs}
    notes: list[str] = []

    best_obj, best_metrics, best_aligned, best_lag = run_calibration_once(
        model_runner, params, reference_df, variables, cfg, weights, lag_variable
    )

    hist = [{"iteration": 0, "objective": best_obj, "lag_min": best_lag, **params}]

    for it in range(1, max_iterations + 1):
        improved = False
        for spec in parameter_specs:
            name = spec.name
            current = params[name]
            for sign in (+1, -1):
                trial = dict(params)
                trial[name] = current + sign * steps[name]
                trial = _clip_params(trial, parameter_specs)
                try:
                    obj, metrics, aligned, lag = run_calibration_once(
                        model_runner, trial, reference_df, variables, cfg, weights, lag_variable
                    )
                except Exception as exc:
                    notes.append(f"iter={it}, {name} trial failed: {exc}")
                    continue
                if obj < best_obj:
                    params, best_obj, best_metrics, best_aligned, best_lag = trial, obj, metrics, aligned, lag
                    improved = True
        hist.append({"iteration": it, "objective": best_obj, "lag_min": best_lag, **params})
        if not improved:
            for k in steps:
                steps[k] *= 0.5
        if all(abs(steps[k]) <= min_steps[k] for k in steps):
            notes.append(f"Stopped at iteration {it}: minimum step size reached.")
            break

    return CalibrationResult(params, float(best_obj), pd.DataFrame(hist), best_metrics, best_aligned, float(best_lag), notes)


# =============================================================================
# Convenience / reporting
# =============================================================================


def create_default_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec("pipe_friction_at_target_psi", 1450.0, 500.0, 3500.0, 250.0),
        ParameterSpec("perf_friction_at_target_psi", 450.0, 50.0, 2000.0, 150.0),
        ParameterSpec("closure_gradient_psi_ft", 0.72, 0.55, 0.95, 0.03),
        ParameterSpec("base_net_pressure_psi", 520.0, 50.0, 2500.0, 150.0),
        ParameterSpec("scenario_severity", 1.0, 0.2, 3.0, 0.2),
    ]


def build_calibration_summary(result: CalibrationResult) -> str:
    lines = ["CALIBRATION SUMMARY", "=" * 80]
    lines.append(f"Best objective: {result.best_objective:.5f}")
    lines.append(f"Estimated lag: {result.lag_min:.2f} min")
    lines.append("\nBest parameters:")
    for k, v in result.best_params.items():
        lines.append(f"  - {k}: {v:.6g}")
    lines.append("\nMetrics:")
    for _, r in result.metrics.iterrows():
        lines.append(
            f"  - {r['variable']}: RMSE={r['rmse']:.2f}, NRMSE={r['nrmse']:.4f}, "
            f"Bias={r['bias']:.2f}, Corr={r['corr']:.3f}, Slope NRMSE={r['slope_nrmse']:.4f}"
        )
    if result.notes:
        lines.append("\nNotes:")
        lines.extend([f"  - {n}" for n in result.notes])
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test with synthetic reference data.
    t = np.arange(0, 60.25, 0.25)
    ref = pd.DataFrame({
        "time_min": t,
        "surface_pressure_psi": 5000 + 22*t + 700*np.exp(-0.5*((t-12)/0.8)**2),
        "slurry_rate_bpm": np.where(t < 2, 40*t, 80),
        "ppa": np.clip((t-15)/30*2, 0, 2),
    })

    def runner(p: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame({
            "time_min": t,
            "surface_pressure_psi": p["mult"]*ref["surface_pressure_psi"] + p["offset"],
            "slurry_rate_bpm": ref["slurry_rate_bpm"],
            "ppa": ref["ppa"],
        })

    specs = [ParameterSpec("mult", 0.90, 0.80, 1.10, 0.04), ParameterSpec("offset", 0.0, -500, 500, 100)]
    result = coordinate_search_calibration(runner, ref, specs, ["surface_pressure_psi", "slurry_rate_bpm", "ppa"], max_iterations=12)
    print(build_calibration_summary(result))
