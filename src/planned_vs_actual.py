from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GAL_PER_BBL = 42.0
EVENT_COLUMNS = ["time_min", "event_type", "severity", "message", "evidence"]


@dataclass(frozen=True)
class PlannedActualConfig:
    rate_tolerance_pct: float = 3.0
    rate_warning_pct: float = 5.0
    ppa_tolerance: float = 0.15
    ppa_warning: float = 0.30
    pressure_tolerance_psi: float = 300.0
    pressure_warning_psi: float = 600.0
    net_pressure_tolerance_psi: float = 250.0
    net_pressure_warning_psi: float = 500.0
    cum_sand_tolerance_pct: float = 5.0
    min_planned_rate_bpm: float = 1.0
    clean_rate_ppa_factor: float = 0.045
    surface_pressure_envelope_psi: float = 500.0
    bhp_envelope_psi: float = 500.0
    net_pressure_envelope_psi: float = 300.0
    frac_hit_net_pressure_drop_slope_psi_min: float = -60.0


def _cfg(config: PlannedActualConfig | None) -> PlannedActualConfig:
    return config or PlannedActualConfig()


def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_min" not in out.columns:
        out["time_min"] = np.arange(len(out), dtype=float)
    out["time_min"] = pd.to_numeric(out["time_min"], errors="coerce").ffill().fillna(0.0)
    return out


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    result = pd.to_numeric(numerator, errors="coerce").div(denom)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _interp_numeric(source: pd.DataFrame, target_time: pd.Series, col: str) -> pd.Series:
    if col not in source.columns or source.empty:
        return pd.Series(np.nan, index=target_time.index, dtype=float)

    src = _ensure_time(source[["time_min", col]].copy()).dropna(subset=["time_min"])
    src[col] = pd.to_numeric(src[col], errors="coerce")
    src = src.dropna(subset=[col]).sort_values("time_min")
    if src.empty:
        return pd.Series(np.nan, index=target_time.index, dtype=float)
    if len(src) == 1:
        return pd.Series(float(src[col].iloc[0]), index=target_time.index, dtype=float)

    values = np.interp(
        target_time.to_numpy(dtype=float),
        src["time_min"].to_numpy(dtype=float),
        src[col].to_numpy(dtype=float),
    )
    return pd.Series(values, index=target_time.index, dtype=float)


def _smoothed_actual(actual_df: pd.DataFrame, target_time: pd.Series, col: str) -> pd.Series:
    if col not in actual_df.columns:
        return pd.Series(np.nan, index=target_time.index, dtype=float)

    source = _ensure_time(actual_df[["time_min", col]].copy()).sort_values("time_min")
    source[col] = (
        pd.to_numeric(source[col], errors="coerce")
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )
    return _interp_numeric(source, target_time, col)


def _ensure_actual_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "surface_ppa": ("surface_ppa", "ppa"),
        "bottomhole_ppa": ("bottomhole_ppa", "ppa"),
        "sand_rate_surface_lb_min": ("sand_rate_surface_lb_min", "sand_rate_lb_min", "sand_rate_lbm_min"),
        "sand_rate_bh_lb_min": ("sand_rate_bh_lb_min", "sand_rate_bh_lbm_min"),
        "cum_sand_surface_lb": ("cum_sand_surface_lb", "cum_sand_lb", "cumulative_sand_lbm"),
        "cum_sand_bh_lb": ("cum_sand_bh_lb", "cumulative_bh_sand_lbm"),
        "cum_slurry_bbl": ("cum_slurry_bbl", "cumulative_slurry_bbl"),
    }
    for canonical, candidates in aliases.items():
        if canonical in out.columns:
            continue
        source = _first_col(out, candidates)
        if source:
            out[canonical] = out[source]
    return out


def calculate_planned_clean_rate(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.Series:
    """Calculate planned clean rate from slurry rate and PPA."""
    cfg = _cfg(config)
    rate = _numeric_series(df, "planned_rate_bpm")
    ppa = _numeric_series(df, "planned_ppa")
    return rate / (1.0 + cfg.clean_rate_ppa_factor * ppa)


def calculate_planned_sand_rate(df: pd.DataFrame) -> pd.Series:
    """Calculate planned surface sand rate in lb/min."""
    clean_rate = _numeric_series(df, "planned_clean_rate_bpm")
    ppa = _numeric_series(df, "planned_ppa")
    return clean_rate * GAL_PER_BBL * ppa


def calculate_cumulative_from_rate(time_min: pd.Series, rate: pd.Series) -> pd.Series:
    """Integrate a rate curve against time using gradient-based time steps."""
    time = pd.to_numeric(time_min, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = pd.to_numeric(rate, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return pd.Series(dtype=float)
    if len(values) == 1:
        dt = np.array([0.0])
    else:
        dt = np.gradient(time)
        dt = np.where(np.isfinite(dt) & (dt >= 0.0), dt, 0.0)
    return pd.Series(np.cumsum(values * dt), index=rate.index, dtype=float)


def add_pressure_envelopes(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.DataFrame:
    """Add high/low planned pressure envelopes when planned pressure curves exist."""
    cfg = _cfg(config)
    out = df.copy()
    envelope_specs = (
        ("planned_surface_pressure_psi", "planned_surface_pressure_low_psi", "planned_surface_pressure_high_psi", cfg.surface_pressure_envelope_psi),
        ("planned_bhp_psi", "planned_bhp_low_psi", "planned_bhp_high_psi", cfg.bhp_envelope_psi),
        ("planned_net_pressure_psi", "planned_net_pressure_low_psi", "planned_net_pressure_high_psi", cfg.net_pressure_envelope_psi),
    )
    for base_col, low_col, high_col, width in envelope_specs:
        if base_col not in out.columns:
            continue
        base = pd.to_numeric(out[base_col], errors="coerce")
        out[low_col] = base - float(width)
        out[high_col] = base + float(width)
    return out


def build_planned_from_schedule(
    schedule_df: pd.DataFrame,
    actual_df: pd.DataFrame | None = None,
    config: PlannedActualConfig | None = None,
) -> pd.DataFrame:
    """Build a normalized planned-treatment dataframe from a schedule or plan run."""
    cfg = _cfg(config)
    out = _ensure_time(schedule_df).copy()

    rate_col = _first_col(out, ("planned_rate_bpm", "slurry_rate_bpm", "rate_bpm"))
    clean_col = _first_col(out, ("planned_clean_rate_bpm", "clean_rate_bpm"))
    ppa_col = _first_col(out, ("planned_ppa", "ppa", "surface_ppa"))
    stage_col = _first_col(out, ("planned_stage_name", "stage_name", "phase"))

    if rate_col:
        out["planned_rate_bpm"] = pd.to_numeric(out[rate_col], errors="coerce").fillna(0.0)
    elif clean_col:
        clean = pd.to_numeric(out[clean_col], errors="coerce").fillna(0.0)
        ppa = pd.to_numeric(out[ppa_col], errors="coerce").fillna(0.0) if ppa_col else 0.0
        out["planned_rate_bpm"] = clean * (1.0 + cfg.clean_rate_ppa_factor * ppa)
    else:
        out["planned_rate_bpm"] = 0.0

    out["planned_ppa"] = (
        pd.to_numeric(out[ppa_col], errors="coerce").fillna(0.0) if ppa_col else 0.0
    )
    out["planned_clean_rate_bpm"] = calculate_planned_clean_rate(out, cfg)
    out["planned_sand_rate_lb_min"] = calculate_planned_sand_rate(out)
    out["planned_cum_slurry_bbl"] = calculate_cumulative_from_rate(
        out["time_min"], out["planned_rate_bpm"]
    )
    out["planned_cum_sand_lb"] = calculate_cumulative_from_rate(
        out["time_min"], out["planned_sand_rate_lb_min"]
    )

    if "planned_bottomhole_ppa" in out.columns:
        out["planned_bottomhole_ppa"] = pd.to_numeric(
            out["planned_bottomhole_ppa"], errors="coerce"
        ).fillna(out["planned_ppa"])
    elif "bottomhole_ppa" in out.columns:
        out["planned_bottomhole_ppa"] = pd.to_numeric(
            out["bottomhole_ppa"], errors="coerce"
        ).fillna(out["planned_ppa"])
    else:
        out["planned_bottomhole_ppa"] = out["planned_ppa"]

    pressure_map = {
        "planned_surface_pressure_psi": "surface_pressure_psi",
        "planned_bhp_psi": "bhp_psi",
        "planned_net_pressure_psi": "net_pressure_psi",
    }
    actual = _ensure_time(actual_df).copy() if actual_df is not None else None
    for planned_col, actual_col in pressure_map.items():
        if planned_col in out.columns:
            out[planned_col] = pd.to_numeric(out[planned_col], errors="coerce")
        elif actual_col in out.columns:
            out[planned_col] = pd.to_numeric(out[actual_col], errors="coerce")
        elif actual is not None and planned_col in actual.columns:
            out[planned_col] = _interp_numeric(actual, out["time_min"], planned_col)
        elif actual is not None and actual_col in actual.columns:
            out[planned_col] = _smoothed_actual(actual, out["time_min"], actual_col)
        else:
            out[planned_col] = np.nan

    out = add_pressure_envelopes(out, cfg)

    if stage_col:
        out["planned_stage_name"] = out[stage_col].astype(str)
    else:
        out["planned_stage_name"] = "Planned"

    if "planned_step_id" in out.columns:
        out["planned_step_id"] = pd.to_numeric(out["planned_step_id"], errors="coerce").fillna(0).astype(int)
    else:
        changed = (
            out["planned_stage_name"].ne(out["planned_stage_name"].shift())
            | out["planned_rate_bpm"].round(3).ne(out["planned_rate_bpm"].round(3).shift())
            | out["planned_ppa"].round(3).ne(out["planned_ppa"].round(3).shift())
        )
        out["planned_step_id"] = changed.cumsum().astype(int)

    keep_cols = [
        "time_min",
        "planned_rate_bpm",
        "planned_clean_rate_bpm",
        "planned_ppa",
        "planned_bottomhole_ppa",
        "planned_sand_rate_lb_min",
        "planned_cum_slurry_bbl",
        "planned_cum_sand_lb",
        "planned_surface_pressure_psi",
        "planned_surface_pressure_low_psi",
        "planned_surface_pressure_high_psi",
        "planned_bhp_psi",
        "planned_bhp_low_psi",
        "planned_bhp_high_psi",
        "planned_net_pressure_psi",
        "planned_net_pressure_low_psi",
        "planned_net_pressure_high_psi",
        "planned_stage_name",
        "planned_step_id",
    ]
    return out[[col for col in keep_cols if col in out.columns]].copy()


def merge_planned_actual(actual_df: pd.DataFrame, planned_df: pd.DataFrame) -> pd.DataFrame:
    """Merge planned curves onto actual execution time using interpolation."""
    actual = _ensure_actual_aliases(_ensure_time(actual_df)).copy().sort_values("time_min")
    planned = _ensure_time(planned_df).copy().sort_values("time_min")

    if planned.empty:
        return actual

    out = actual.drop(columns=[c for c in planned.columns if c != "time_min" and c in actual.columns], errors="ignore")
    target_time = out["time_min"]

    text_cols = ["planned_stage_name"]
    for col in planned.columns:
        if col == "time_min":
            continue
        if col in text_cols:
            nearest = pd.merge_asof(
                out[["time_min"]].sort_values("time_min"),
                planned[["time_min", col]].sort_values("time_min"),
                on="time_min",
                direction="nearest",
            )
            out[col] = nearest[col].to_numpy()
            continue

        if pd.api.types.is_numeric_dtype(planned[col]):
            out[col] = _interp_numeric(planned, target_time, col)
        else:
            nearest = pd.merge_asof(
                out[["time_min"]].sort_values("time_min"),
                planned[["time_min", col]].sort_values("time_min"),
                on="time_min",
                direction="nearest",
            )
            out[col] = nearest[col].to_numpy()

    return out.sort_values("time_min").reset_index(drop=True)


def calculate_deviation_metrics(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.DataFrame:
    """Add rate, proppant, sand and pressure deviation metrics."""
    cfg = _cfg(config)
    out = _ensure_actual_aliases(df).copy()

    out["rate_error_bpm"] = _numeric_series(out, "slurry_rate_bpm") - _numeric_series(out, "planned_rate_bpm")
    planned_rate = _numeric_series(out, "planned_rate_bpm")
    rate_denominator = planned_rate.where(planned_rate.abs() >= cfg.min_planned_rate_bpm, np.nan)
    out["rate_error_pct"] = 100.0 * _safe_divide(out["rate_error_bpm"], rate_denominator)

    out["ppa_error"] = _numeric_series(out, "surface_ppa") - _numeric_series(out, "planned_ppa")
    out["bh_ppa_error"] = _numeric_series(out, "bottomhole_ppa") - _numeric_series(out, "planned_bottomhole_ppa")

    out["sand_rate_error_lb_min"] = (
        _numeric_series(out, "sand_rate_surface_lb_min")
        - _numeric_series(out, "planned_sand_rate_lb_min")
    )
    out["cum_sand_error_lb"] = (
        _numeric_series(out, "cum_sand_surface_lb")
        - _numeric_series(out, "planned_cum_sand_lb")
    )
    out["cum_sand_error_pct"] = 100.0 * _safe_divide(
        out["cum_sand_error_lb"], _numeric_series(out, "planned_cum_sand_lb")
    )

    out["surface_pressure_error_psi"] = (
        _numeric_series(out, "surface_pressure_psi")
        - _numeric_series(out, "planned_surface_pressure_psi")
    )
    out["bhp_error_psi"] = _numeric_series(out, "bhp_psi") - _numeric_series(out, "planned_bhp_psi")
    out["net_pressure_error_psi"] = (
        _numeric_series(out, "net_pressure_psi")
        - _numeric_series(out, "planned_net_pressure_psi")
    )
    return out


def _classify_abs(series: pd.Series, ok_limit: float, watch_limit: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").abs().fillna(0.0)
    return pd.Series(
        np.select(
            [values <= ok_limit, values <= watch_limit],
            ["OK", "WATCH"],
            default="OFF PLAN",
        ),
        index=series.index,
    )


def classify_execution_status(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.DataFrame:
    """Classify row-level plan adherence for each monitoring channel."""
    cfg = _cfg(config)
    out = df.copy()
    out["rate_status"] = _classify_abs(out.get("rate_error_pct", 0.0), cfg.rate_tolerance_pct, cfg.rate_warning_pct)
    out["ppa_status"] = _classify_abs(out.get("ppa_error", 0.0), cfg.ppa_tolerance, cfg.ppa_warning)
    out["pressure_status"] = _classify_abs(
        out.get("surface_pressure_error_psi", 0.0),
        cfg.pressure_tolerance_psi,
        cfg.pressure_warning_psi,
    )
    out["sand_status"] = _classify_abs(
        out.get("cum_sand_error_pct", 0.0),
        cfg.cum_sand_tolerance_pct,
        2.0 * cfg.cum_sand_tolerance_pct,
    )

    status_cols = ["rate_status", "ppa_status", "pressure_status", "sand_status"]
    any_off = out[status_cols].eq("OFF PLAN").any(axis=1)
    any_watch = out[status_cols].eq("WATCH").any(axis=1)
    out["overall_execution_status"] = np.select(
        [any_off, any_watch],
        ["OFF PLAN", "WATCH"],
        default="ON PLAN",
    )
    return out


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(col, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _event_row(time_min: float, event_type: str, severity: str, message: str, evidence: str) -> dict[str, object]:
    return {
        "time_min": round(float(time_min), 2),
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def detect_planned_actual_events(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.DataFrame:
    """Detect debounced planned-vs-actual monitoring events."""
    cfg = _cfg(config)
    out = _ensure_time(df).copy()
    if out.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    for col in ("net_pressure_slope_psi_min", "perf_friction_slope_psi_min"):
        if col not in out.columns:
            source = "net_pressure_psi" if col.startswith("net") else "perf_friction_psi"
            if source in out.columns and len(out) > 1:
                out[col] = np.gradient(
                    pd.to_numeric(out[source], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                    pd.to_numeric(out["time_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                )
            else:
                out[col] = 0.0

    events: list[dict[str, object]] = []
    last_event_time: dict[str, float] = {}
    cooldown_min = 2.0

    def add_event(row: pd.Series, event_type: str, severity: str, message: str, evidence: str) -> None:
        time_min = _row_float(row, "time_min")
        last_time = last_event_time.get(event_type)
        if last_time is not None and time_min - last_time < cooldown_min:
            return
        events.append(_event_row(time_min, event_type, severity, message, evidence))
        last_event_time[event_type] = time_min

    for _, row in out.iterrows():
        rate_error = _row_float(row, "rate_error_pct")
        ppa_error = _row_float(row, "ppa_error")
        surface_error = _row_float(row, "surface_pressure_error_psi")
        net_error = _row_float(row, "net_pressure_error_psi")
        bhp_error = _row_float(row, "bhp_error_psi")
        cum_sand_error = _row_float(row, "cum_sand_error_pct")
        bh_ppa = _row_float(row, "bottomhole_ppa")
        net_slope = _row_float(row, "net_pressure_slope_psi_min")
        perf_slope = _row_float(row, "perf_friction_slope_psi_min")
        pipe_slope = _row_float(row, "pipe_friction_slope_psi_min")
        pressure_slope = _row_float(row, "pressure_slope_psi_min")
        screenout_risk = _row_float(row, "screenout_risk")

        rate_off = abs(rate_error) > cfg.rate_warning_pct
        perf_restriction = (
            surface_error > cfg.pressure_warning_psi
            and perf_slope > 60.0
            and net_error < cfg.net_pressure_warning_psi
        )
        perf_context_active = (
            "POSSIBLE_PERF_RESTRICTION_FROM_PLAN" in last_event_time
            and surface_error > cfg.pressure_warning_psi
        )
        sensor_issue = (
            surface_error > cfg.pressure_warning_psi
            and abs(bhp_error) < cfg.pressure_tolerance_psi
            and abs(net_error) < cfg.net_pressure_tolerance_psi
            and abs(pressure_slope) > 60.0
            and perf_slope <= 25.0
            and abs(pipe_slope) <= 25.0
            and not perf_context_active
        )
        possible_screenout = (
            abs(rate_error) < cfg.rate_tolerance_pct
            and bh_ppa > 0.5
            and net_error > cfg.net_pressure_warning_psi
            and net_slope > 80.0
            and abs(pipe_slope) < 60.0
            and perf_slope < 60.0
            and screenout_risk > 0.55
        )
        pressure_under_plan = (
            surface_error < -cfg.pressure_warning_psi
            and abs(rate_error) < cfg.rate_tolerance_pct
        )
        possible_frac_hit = (
            abs(rate_error) < cfg.rate_tolerance_pct
            and abs(ppa_error) < cfg.ppa_warning
            and surface_error < -cfg.pressure_warning_psi
            and net_error < -cfg.net_pressure_warning_psi
            and net_slope <= cfg.frac_hit_net_pressure_drop_slope_psi_min
        )

        if rate_off:
            add_event(
                row,
                "RATE_OFF_PLAN",
                "WARNING",
                "Actual slurry rate is outside the planned tolerance.",
                f"rate_error={rate_error:.1f}%",
            )

        if abs(ppa_error) > cfg.ppa_warning:
            add_event(
                row,
                "PPA_OFF_PLAN",
                "WARNING",
                "Surface proppant concentration is outside the planned schedule.",
                f"ppa_error={ppa_error:.2f} lb/gal",
            )

        if cum_sand_error < -cfg.cum_sand_tolerance_pct:
            add_event(
                row,
                "SAND_UNDER_DELIVERY",
                "WARNING",
                "Cumulative sand delivery is under the treatment plan.",
                f"cum_sand_error={cum_sand_error:.1f}%",
            )
        elif cum_sand_error > cfg.cum_sand_tolerance_pct:
            add_event(
                row,
                "SAND_OVER_DELIVERY",
                "WARNING",
                "Cumulative sand delivery is above the treatment plan.",
                f"cum_sand_error={cum_sand_error:.1f}%",
            )

        if pressure_under_plan:
            add_event(
                row,
                "PRESSURE_UNDER_PLAN",
                "WATCH",
                "Actual surface pressure is below planned pressure while rate is close to plan.",
                f"Surface pressure error = {surface_error:.0f} psi, rate error = {rate_error:+.1f}%",
            )

        if possible_frac_hit:
            add_event(
                row,
                "POSSIBLE_FRAC_HIT_FROM_PLAN",
                "WARNING",
                "Actual pressure and net pressure fell below plan while rate and PPA remained close to plan. Possible frac hit / pressure communication.",
                (
                    f"Rate error = {rate_error:+.1f}%, PPA error = {ppa_error:+.2f}, "
                    f"surface pressure error = {surface_error:.0f} psi, net pressure error = {net_error:.0f} psi"
                ),
            )
            continue

        if perf_restriction:
            add_event(
                row,
                "POSSIBLE_PERF_RESTRICTION_FROM_PLAN",
                "CRITICAL",
                "Pressure is over plan while perforation friction is rising faster than net pressure.",
                f"surface_error={surface_error:.0f} psi, perf_slope={perf_slope:.0f} psi/min",
            )
            continue

        if sensor_issue:
            add_event(
                row,
                "POSSIBLE_SENSOR_DATA_ISSUE",
                "WARNING",
                "Surface pressure is over plan without support from BHP, net pressure or friction.",
                f"surface_error={surface_error:.0f} psi, bhp_error={bhp_error:.0f} psi",
            )
            continue

        if possible_screenout:
            add_event(
                row,
                "POSSIBLE_SCREENOUT_FROM_PLAN",
                "CRITICAL",
                "Rate is stable, bottomhole PPA is high and net pressure is rising over plan.",
                f"net_error={net_error:.0f} psi, net_slope={net_slope:.0f} psi/min, risk={screenout_risk:.2f}",
            )
            continue

        if surface_error > cfg.pressure_warning_psi and not rate_off:
            add_event(
                row,
                "PRESSURE_OVER_PLAN",
                "WARNING",
                "Surface pressure is above the planned pressure envelope.",
                f"surface_error={surface_error:.0f} psi",
            )

        if net_error > cfg.net_pressure_warning_psi and not rate_off:
            add_event(
                row,
                "NET_PRESSURE_OVER_PLAN",
                "WARNING",
                "Net pressure is above the planned pressure response.",
                f"net_error={net_error:.0f} psi",
            )

    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def calculate_plan_compliance_score(
    df: pd.DataFrame,
    config: PlannedActualConfig | None = None,
) -> pd.Series:
    """Return a row-level plan compliance score from 0 to 100."""
    cfg = _cfg(config)
    rate_penalty = np.minimum(_numeric_series(df, "rate_error_pct").abs() / cfg.rate_warning_pct, 2.0)
    ppa_penalty = np.minimum(_numeric_series(df, "ppa_error").abs() / cfg.ppa_warning, 2.0)
    pressure_penalty = np.minimum(
        _numeric_series(df, "surface_pressure_error_psi").abs() / cfg.pressure_warning_psi,
        2.0,
    )
    sand_penalty = np.minimum(
        _numeric_series(df, "cum_sand_error_pct").abs() / (2.0 * cfg.cum_sand_tolerance_pct),
        2.0,
    )
    raw_penalty = (
        0.25 * rate_penalty
        + 0.25 * ppa_penalty
        + 0.30 * pressure_penalty
        + 0.20 * sand_penalty
    )
    score = 100.0 * np.maximum(0.0, 1.0 - raw_penalty / 2.0)
    return pd.Series(score, index=df.index).round(1)


def _current_row(df: pd.DataFrame, current_time_min: float | None = None) -> pd.Series | None:
    if df.empty:
        return None
    out = _ensure_time(df)
    if current_time_min is None:
        return out.iloc[-1]
    idx = int(np.argmin(np.abs(out["time_min"].to_numpy(dtype=float) - float(current_time_min))))
    return out.iloc[idx]


def _recent_events(
    event_log: pd.DataFrame,
    current_time_min: float | None,
    *,
    lookback_min: float = 5.0,
) -> pd.DataFrame:
    if event_log is None or event_log.empty or "event_type" not in event_log.columns:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = event_log.copy()
    events["time_min"] = pd.to_numeric(events.get("time_min", np.nan), errors="coerce")
    if current_time_min is None:
        return events
    current = float(current_time_min)
    return events[(events["time_min"] <= current) & (events["time_min"] >= current - lookback_min)].copy()


def _events_to_time(event_log: pd.DataFrame, current_time_min: float | None) -> pd.DataFrame:
    if event_log is None or event_log.empty or "event_type" not in event_log.columns:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = event_log.copy()
    events["time_min"] = pd.to_numeric(events.get("time_min", np.nan), errors="coerce")
    if current_time_min is None:
        return events
    return events[events["time_min"] <= float(current_time_min)].copy()


def _has_event(events: pd.DataFrame, event_type: str) -> bool:
    return not events.empty and events["event_type"].astype(str).eq(event_type).any()


def _latest_event_time(events: pd.DataFrame, event_type: str) -> float | None:
    if events.empty or "time_min" not in events.columns:
        return None
    mask = events["event_type"].astype(str).eq(event_type)
    if not mask.any():
        return None
    return float(events.loc[mask, "time_min"].max())


def _base_evidence(row: pd.Series) -> list[str]:
    return [
        f"Rate error: {_row_float(row, 'rate_error_pct'):+.1f}%",
        f"PPA error: {_row_float(row, 'ppa_error'):+.2f} lb/gal",
        f"Surface pressure error: {_row_float(row, 'surface_pressure_error_psi'):+,.0f} psi",
        f"Net pressure error: {_row_float(row, 'net_pressure_error_psi'):+,.0f} psi",
        f"Plan compliance score: {_row_float(row, 'plan_compliance_score', 100.0):.0f}/100",
    ]


def build_planned_actual_interpretation(
    df: pd.DataFrame,
    event_log: pd.DataFrame,
    current_time_min: float | None = None,
    config: PlannedActualConfig | None = None,
) -> dict:
    """Build a trainee-facing interpretation from planned-vs-actual signals."""
    cfg = _cfg(config)
    row = _current_row(df, current_time_min)
    if row is None:
        return {
            "status": "INFO",
            "title": "Planned-vs-actual unavailable",
            "summary": "No execution data is available for interpretation.",
            "evidence": ["No rows available in planned-vs-actual dataframe."],
            "recommended_action": "Verify simulation data",
        }

    current_time = _row_float(row, "time_min") if current_time_min is None else float(current_time_min)
    recent_events = _recent_events(event_log, current_time)
    history_events = _events_to_time(event_log, current_time)
    severity_series = (
        recent_events["severity"]
        if "severity" in recent_events.columns
        else pd.Series("", index=recent_events.index, dtype=str)
    )
    warning_events = recent_events[severity_series.astype(str).str.upper().isin(["WARNING", "CRITICAL"])]
    execution_status = str(row.get("overall_execution_status", "ON PLAN")).upper()
    rate_status = str(row.get("rate_status", "OK")).upper()
    ppa_status = str(row.get("ppa_status", "OK")).upper()

    screenout_now = (
        _row_float(row, "screenout_risk") > 0.55
        and _row_float(row, "net_pressure_error_psi") > cfg.net_pressure_warning_psi
        and abs(_row_float(row, "rate_error_pct")) < cfg.rate_tolerance_pct
        and _row_float(row, "bottomhole_ppa") > 0.5
    )
    frac_hit_now = (
        _row_float(row, "surface_pressure_error_psi") < -cfg.pressure_warning_psi
        and _row_float(row, "net_pressure_error_psi") < -cfg.net_pressure_warning_psi
        and abs(_row_float(row, "rate_error_pct")) < cfg.rate_tolerance_pct
        and abs(_row_float(row, "ppa_error")) < cfg.ppa_warning
    )

    if _has_event(recent_events, "POSSIBLE_SCREENOUT_FROM_PLAN") or _has_event(history_events, "POSSIBLE_SCREENOUT_FROM_PLAN") or screenout_now:
        evidence = [
            f"Rate error: {_row_float(row, 'rate_error_pct'):+.1f}%",
            f"Bottomhole PPA: {_row_float(row, 'bottomhole_ppa'):.2f} lb/gal",
            f"Net pressure error: {_row_float(row, 'net_pressure_error_psi'):+,.0f} psi",
            f"Net pressure slope: {_row_float(row, 'net_pressure_slope_psi_min'):+.0f} psi/min",
            f"Screenout risk: {_row_float(row, 'screenout_risk'):.2f}",
        ]
        if "acceptance_index" in row:
            evidence.append(f"Acceptance index: {_row_float(row, 'acceptance_index'):.2f}")
        return {
            "status": "CRITICAL" if _row_float(row, "screenout_risk") > 0.75 else "WARNING",
            "title": "Possible screenout trend",
            "summary": "Rate is close to plan and bottomhole PPA is high, but actual net pressure is rising above plan. Friction does not fully explain the pressure increase.",
            "evidence": evidence,
            "recommended_action": "Reduce PPA / prepare flush",
        }

    if _has_event(recent_events, "POSSIBLE_FRAC_HIT_FROM_PLAN") or _has_event(history_events, "POSSIBLE_FRAC_HIT_FROM_PLAN") or frac_hit_now:
        event_time = _latest_event_time(history_events, "POSSIBLE_FRAC_HIT_FROM_PLAN")
        evidence = [
            f"Surface pressure error: {_row_float(row, 'surface_pressure_error_psi'):+,.0f} psi",
            f"Net pressure error: {_row_float(row, 'net_pressure_error_psi'):+,.0f} psi",
            f"Rate error: {_row_float(row, 'rate_error_pct'):+.1f}%",
            f"PPA error: {_row_float(row, 'ppa_error'):+.2f} lb/gal",
        ]
        if event_time is not None:
            evidence.append(f"Event time: {event_time:.2f} min")
        return {
            "status": "WARNING",
            "title": "Possible frac hit / pressure communication",
            "summary": "Actual pressure and net pressure fell below plan while rate and PPA remained close to plan. This suggests communication with an offset well, natural fracture, or pressure sink.",
            "evidence": evidence,
            "recommended_action": "Evaluate offset communication",
        }

    if _has_event(recent_events, "POSSIBLE_PERF_RESTRICTION_FROM_PLAN") or _has_event(history_events, "POSSIBLE_PERF_RESTRICTION_FROM_PLAN"):
        return {
            "status": "WARNING",
            "title": "Possible perforation / cluster restriction",
            "summary": "Surface pressure is above plan and perforation friction is increasing, while net pressure does not explain the full pressure increase.",
            "evidence": [
                f"Surface pressure error: {_row_float(row, 'surface_pressure_error_psi'):+,.0f} psi",
                f"Perf friction slope: {_row_float(row, 'perf_friction_slope_psi_min'):+.0f} psi/min",
                f"Net pressure error: {_row_float(row, 'net_pressure_error_psi'):+,.0f} psi",
            ],
            "recommended_action": "Hold PPA and investigate restriction",
        }

    if execution_status == "OFF PLAN" and ("OFF PLAN" in {rate_status, ppa_status}):
        return {
            "status": "WARNING",
            "title": "Execution is off treatment plan",
            "summary": "Actual execution is deviating from planned rate or proppant schedule. Do not interpret pressure as formation response until operational deviation is understood.",
            "evidence": _base_evidence(row),
            "recommended_action": "Check pumps/blender/proppant delivery",
        }

    if execution_status == "OFF PLAN":
        return {
            "status": "WARNING",
            "title": "Execution is outside planned envelope",
            "summary": "One or more monitoring channels are outside the planned envelope. Separate operational deviation from formation response before making a treatment decision.",
            "evidence": _base_evidence(row),
            "recommended_action": "Verify rate, pressure, friction and sensor channels",
        }

    if execution_status == "WATCH":
        return {
            "status": "WATCH",
            "title": "Small deviation from planned treatment",
            "summary": "One or more operational variables are drifting from plan. Check rate, PPA, sand and pressure before diagnosing formation behavior.",
            "evidence": _base_evidence(row),
            "recommended_action": "Continue monitoring / verify operational channels",
        }

    if execution_status == "ON PLAN" and warning_events.empty:
        return {
            "status": "ON PLAN",
            "title": "Execution tracking treatment program",
            "summary": "Rate, PPA, sand delivery and pressure are within planned tolerance. Continue monitoring.",
            "evidence": _base_evidence(row),
            "recommended_action": "Continue pumping",
        }

    return {
        "status": "INFO",
        "title": "Monitoring planned-vs-actual response",
        "summary": "Recent events are present, but the current row does not match a specific interpretation rule.",
        "evidence": _base_evidence(row),
        "recommended_action": "Review event log and trend plots",
    }


def apply_planned_vs_actual(
    actual_df: pd.DataFrame,
    planned_df: pd.DataFrame | None = None,
    config: PlannedActualConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply planned-vs-actual enrichment and return the event log."""
    cfg = _cfg(config)
    actual = _ensure_actual_aliases(_ensure_time(actual_df)).copy()
    plan_source = planned_df.copy() if planned_df is not None else actual.copy()
    planned = build_planned_from_schedule(plan_source, actual_df=actual, config=cfg)

    enriched = merge_planned_actual(actual, planned)
    enriched = add_pressure_envelopes(enriched, cfg)
    enriched = calculate_deviation_metrics(enriched, cfg)
    enriched = classify_execution_status(enriched, cfg)
    enriched["plan_compliance_score"] = calculate_plan_compliance_score(enriched, cfg)
    event_log = detect_planned_actual_events(enriched, cfg)
    enriched.attrs["planned_actual_event_log"] = event_log
    return enriched, event_log


if __name__ == "__main__":
    time_min = pd.Series(np.arange(0.0, 12.5, 0.5))
    planned = pd.DataFrame(
        {
            "time_min": time_min,
            "phase": np.where(time_min < 3, "Pad", np.where(time_min < 10, "Main proppant", "Flush")),
            "slurry_rate_bpm": 80.0,
            "ppa": np.where(time_min < 3, 0.0, np.where(time_min < 10, 2.0, 0.0)),
            "bottomhole_ppa": np.where(time_min < 4, 0.0, np.where(time_min < 11, 2.0, 0.0)),
            "surface_pressure_psi": 5600.0 + 25.0 * time_min,
            "bhp_psi": 7800.0 + 20.0 * time_min,
            "net_pressure_psi": 1400.0 + 20.0 * time_min,
        }
    )
    planned["clean_rate_bpm"] = planned["slurry_rate_bpm"] / (1.0 + 0.045 * planned["ppa"])
    planned["sand_rate_surface_lb_min"] = planned["clean_rate_bpm"] * GAL_PER_BBL * planned["ppa"]
    planned["cum_sand_surface_lb"] = calculate_cumulative_from_rate(
        planned["time_min"], planned["sand_rate_surface_lb_min"]
    )

    actual = planned.copy()
    actual.loc[actual["time_min"] >= 7.0, "surface_pressure_psi"] -= 850.0
    actual.loc[actual["time_min"] >= 7.0, "bhp_psi"] -= 700.0
    actual.loc[actual["time_min"] >= 7.0, "net_pressure_psi"] -= 700.0
    actual["screenout_risk"] = 0.2
    actual["acceptance_index"] = 0.85
    actual["net_pressure_slope_psi_min"] = np.gradient(actual["net_pressure_psi"], actual["time_min"])
    actual["perf_friction_psi"] = 800.0
    actual["perf_friction_slope_psi_min"] = 0.0
    actual["pipe_friction_slope_psi_min"] = 0.0

    enriched_df, events = apply_planned_vs_actual(actual, planned)
    interpretation = build_planned_actual_interpretation(enriched_df, events, current_time_min=7.0)
    assert "PRESSURE_UNDER_PLAN" in set(events["event_type"])
    assert "POSSIBLE_FRAC_HIT_FROM_PLAN" in set(events["event_type"])
    title = str(interpretation["title"]).lower()
    assert "frac hit" in title or "pressure communication" in title
    print(enriched_df[["time_min", "overall_execution_status", "plan_compliance_score"]].tail())
    print(events)
    print(interpretation)
