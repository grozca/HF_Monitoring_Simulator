from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


NO_MAPPING = "(not mapped)"


class ExcelMode(Enum):
    """Type of Excel file detected from its headers."""

    SIMULATION = "simulation"
    CALIBRATION = "calibration"
    MINIFRAC = "minifrac"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StandardColumn:
    name: str
    label: str
    required_for_curve: bool = False


STANDARD_COLUMNS = [
    StandardColumn("time_min", "Time, min", required_for_curve=True),
    StandardColumn("phase", "Phase"),
    StandardColumn("slurry_rate_bpm", "Slurry rate, bpm"),
    StandardColumn("clean_rate_bpm", "Clean rate, bpm"),
    StandardColumn("surface_pressure_psi", "Surface pressure, psi"),
    StandardColumn("ppa", "PPA / proppant concentration"),
    StandardColumn("sand_rate_lb_min", "Sand rate, lb/min"),
    StandardColumn("pipe_friction_psi", "Pipe friction, psi"),
    StandardColumn("perf_friction_psi", "Perforation friction, psi"),
    StandardColumn("bhp_psi", "BHP, psi"),
    StandardColumn("net_pressure_psi", "Net pressure, psi"),
    StandardColumn("alarm", "Alarm"),
    StandardColumn("engineer_diagnosis", "Engineer diagnosis"),
]


ALIAS_PATTERNS: dict[str, list[str]] = {
    "time_min": [
        "time min", "time_min", "time from start min", "t min", "t_min",
        "time", "tiempo min",
    ],
    "phase": ["phase", "stage", "fase"],
    "slurry_rate_bpm": [
        "rate_bpm", "slurry rate bpm", "slurry_rate_bpm",
        "bh injection rate bpm", "qi_liq bpm", "qi_liq_bpm",
        "injection rate bpm", "rate bpm", "treating rate bpm",
        "pump rate bpm", "stp",
    ],
    "clean_rate_bpm": [
        "clean fluid bbl min", "clean rate bpm", "clean_rate_bpm",
        "clean fluid bpm", "clean_fluid_bbl_min",
    ],
    "surface_pressure_psi": [
        "surface_pressure_psi", "surface pressure psi",
        "surface treating pressure psi", "treating pressure psi",
        "stp psi", "wellhead pressure psi", "wh pressure psi",
        "surface pressure", "sp psi",
    ],
    "ppa": [
        "proppant_ppa", "ppa", "prop conc", "proppant concentration",
        "cadd lbm gal", "cadd_lbm_gal", "lbm gal", "proppant ppa",
        "sand concentration", "prop conc lbm gal",
    ],
    "sand_rate_lb_min": [
        "sand_rate_lb_min", "sand rate lb min", "sand rate lbm min",
        "proppant rate", "sand rate",
    ],
    "pipe_friction_psi": ["pipe_friction_psi", "pipe friction psi"],
    "perf_friction_psi": [
        "perf_friction_psi", "perf friction psi", "perforation friction psi",
    ],
    "bhp_psi": [
        "bhp_psi", "bhp psi", "bh pressure psi", "bottomhole pressure psi",
        "bottom hole pressure psi",
    ],
    "net_pressure_psi": [
        "net_pressure_psi", "net pressure psi", "net p", "pnet psi",
        "pnet_psi", "pnet", "net pressure",
    ],
    "alarm": ["alarm", "alerta"],
    "engineer_diagnosis": [
        "engineer_diagnosis", "engineer diagnosis", "diagnosis", "diagnostico",
    ],
}


_CALIBRATION_SIGNATURE_COLS = [
    "xf ft", "xf_ft", "wave in", "wave_in", "wdry wwet", "wdry_wwet",
    "pnet psi", "pnet_psi", "qi_liq bpm", "qi_liq_bpm",
    "cum liq gal", "cum_liq_gal", "cum prop lbm", "cum_prop_lbm",
]

CALIBRATION_ALIAS_PATTERNS: dict[str, list[str]] = {
    "time_min": ["t min", "t_min", "time min", "time_min"],
    "rate_bpm": ["qi_liq bpm", "qi_liq_bpm", "rate bpm", "injection rate bpm"],
    "ppa": ["cadd lbm gal", "cadd_lbm_gal", "ppa", "prop conc"],
    "cum_liquid_gal": ["cum liq gal", "cum_liq_gal"],
    "cum_prop_lbm": ["cum prop lbm", "cum_prop_lbm"],
    "xf_ft": ["xf ft", "xf_ft", "half length ft", "fracture half length ft"],
    "width_in": ["wave in", "wave_in", "width in", "fracture width in"],
    "width_ratio": ["wdry wwet", "wdry_wwet"],
    "net_pressure_psi": ["pnet psi", "pnet_psi", "pnet", "net pressure psi"],
}


_MINIFRAC_SIGNATURE_COLS = [
    "g func", "g-func", "g_func", "g function", "g-function",
    "g_function", "shut in pressure psi", "shut-in pressure psi",
    "static pressure psi", "isip psi", "straight line",
]

MINIFRAC_ALIAS_PATTERNS: dict[str, list[str]] = {
    "time_min": ["time min", "time_min", "t min", "time from start min"],
    "rate_bpm": ["bh injection rate bpm", "injection rate bpm", "rate bpm"],
    "bhp_psi": ["bh pressure psi", "bottomhole pressure psi", "bhp psi"],
    "g_function": ["g func", "g-func", "g_func", "g function", "g-function"],
    "shutin_psi": [
        "shut in pressure psi", "shut-in pressure psi", "static pressure psi",
        "isip psi",
    ],
    "straight_line_psi": ["straight line", "press from fit", "linear fit"],
}


def normalize_column_name(name: object) -> str:
    """Normalize a source header for matching."""
    text = str(name).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_excel_mode(columns: list[object]) -> ExcelMode:
    """Classify Excel data as simulation, calibration, minifrac, or unknown."""
    normalized_cols = {normalize_column_name(col) for col in columns}

    if any(normalize_column_name(sig) in normalized_cols for sig in _MINIFRAC_SIGNATURE_COLS):
        return ExcelMode.MINIFRAC

    if any(normalize_column_name(sig) in normalized_cols for sig in _CALIBRATION_SIGNATURE_COLS):
        return ExcelMode.CALIBRATION

    has_time = any(
        normalize_column_name(col) in {"time min", "time", "t min"}
        for col in columns
    )
    has_signal = any(
        any(normalize_column_name(alias) in normalized_cols for alias in ALIAS_PATTERNS[key])
        for key in ("slurry_rate_bpm", "surface_pressure_psi", "bhp_psi", "ppa")
    )
    if has_time and has_signal:
        return ExcelMode.SIMULATION

    return ExcelMode.UNKNOWN


def _column_score(source_column: object, aliases: list[str]) -> int:
    normalized = normalize_column_name(source_column)
    if not normalized or normalized.startswith("unnamed"):
        return 0

    best_score = 0
    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if len(normalized_alias) < 2:
            continue
        if normalized == normalized_alias:
            best_score = max(best_score, 100)
        elif normalized_alias and normalized_alias in normalized:
            best_score = max(best_score, 80)
    return best_score


def suggest_column_mapping(
    columns: list[object],
    alias_patterns: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """Return best-guess mapping from internal names to source headers."""
    patterns = alias_patterns or ALIAS_PATTERNS
    standard_names = list(patterns.keys()) if alias_patterns else [col.name for col in STANDARD_COLUMNS]
    mapping: dict[str, str | None] = {}
    used_sources: set[str] = set()

    for name in standard_names:
        aliases = patterns.get(name, [])
        scored = sorted(
            (
                (_column_score(source, aliases), str(source))
                for source in columns
                if str(source) not in used_sources
            ),
            reverse=True,
        )
        if scored and scored[0][0] >= 65:
            mapping[name] = scored[0][1]
            used_sources.add(scored[0][1])
        else:
            mapping[name] = None

    return mapping


def suggest_column_mapping_with_scores(
    columns: list[object],
) -> dict[str, tuple[str | None, int]]:
    """Return mapping plus confidence score for simulation columns."""
    mapping: dict[str, tuple[str | None, int]] = {}
    used_sources: set[str] = set()

    for standard in STANDARD_COLUMNS:
        aliases = ALIAS_PATTERNS.get(standard.name, [])
        scored = sorted(
            (
                (_column_score(source, aliases), str(source))
                for source in columns
                if str(source) not in used_sources
            ),
            reverse=True,
        )
        if scored and scored[0][0] >= 65:
            score, source = scored[0]
            mapping[standard.name] = (source, score)
            used_sources.add(source)
        else:
            mapping[standard.name] = (None, 0)

    return mapping


def mapping_options(columns: list[object]) -> list[str]:
    return [NO_MAPPING, *[str(column) for column in columns]]


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    selected: dict[str, pd.Series] = {}
    for standard_name, source_name in mapping.items():
        if not source_name or source_name == NO_MAPPING:
            continue
        if source_name not in df.columns:
            continue
        selected[standard_name] = df[source_name]

    out = pd.DataFrame(selected)
    out = _coerce_known_numeric_columns(out)
    out = _add_alias_columns(out)
    return out


def _coerce_known_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = {
        "time_min", "slurry_rate_bpm", "clean_rate_bpm",
        "surface_pressure_psi", "ppa", "sand_rate_lb_min",
        "pipe_friction_psi", "perf_friction_psi", "bhp_psi", "net_pressure_psi",
    }
    for column in numeric_columns.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _add_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sand_rate_lb_min" in out.columns and "sand_rate_lbm_min" not in out.columns:
        out["sand_rate_lbm_min"] = out["sand_rate_lb_min"]
    if "sand_rate_lbm_min" in out.columns and "sand_rate_lb_min" not in out.columns:
        out["sand_rate_lb_min"] = out["sand_rate_lbm_min"]
    return out


def derive_missing_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived columns after mapping when the source has enough data."""
    out = df.copy()

    if "clean_rate_bpm" not in out.columns and {"slurry_rate_bpm", "ppa"}.issubset(out.columns):
        out["clean_rate_bpm"] = out["slurry_rate_bpm"] / (1.0 + 0.045 * out["ppa"].fillna(0.0))

    if "sand_rate_lb_min" not in out.columns and {"clean_rate_bpm", "ppa"}.issubset(out.columns):
        out["sand_rate_lb_min"] = out["clean_rate_bpm"] * 42.0 * out["ppa"].fillna(0.0)

    if "sand_rate_lb_min" in out.columns:
        out["sand_rate_lbm_min"] = out["sand_rate_lb_min"]

    if "phase" not in out.columns and "time_min" in out.columns:
        out["phase"] = "Imported"

    return out


def mapping_readiness(mapped_df: pd.DataFrame) -> dict[str, object]:
    available = set(mapped_df.columns)
    has_time = "time_min" in available and mapped_df["time_min"].notna().any()
    curve_columns = [
        "slurry_rate_bpm", "surface_pressure_psi", "ppa",
        "bhp_psi", "net_pressure_psi", "pipe_friction_psi", "perf_friction_psi",
    ]
    available_curves = [
        column for column in curve_columns
        if column in available and mapped_df[column].notna().any()
    ]
    return {
        "has_time": has_time,
        "available_curves": available_curves,
        "can_plot": bool(has_time and available_curves),
        "mapped_columns": sorted(available),
        "usable_rows": int(mapped_df.dropna(how="all").shape[0]),
    }


def clean_mapped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_missing_standard_columns(df)
    if "time_min" in out.columns:
        out = out[out["time_min"].notna()].copy()
        out = out.sort_values("time_min")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def suggest_calibration_mapping(columns: list[object]) -> dict[str, str | None]:
    """Suggest mapping for HF2D/PKN fracture-geometry output."""
    return suggest_column_mapping(columns, alias_patterns=CALIBRATION_ALIAS_PATTERNS)


def apply_calibration_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Apply an HF2D/PKN design mapping."""
    selected: dict[str, pd.Series] = {}
    for internal, source in mapping.items():
        if not source or source == NO_MAPPING or source not in df.columns:
            continue
        selected[internal] = df[source]

    out = pd.DataFrame(selected)
    numeric_columns = {
        "time_min", "rate_bpm", "ppa", "cum_liquid_gal", "cum_prop_lbm",
        "xf_ft", "width_in", "width_ratio", "net_pressure_psi",
    }
    for column in numeric_columns.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "time_min" in out.columns:
        out = out[out["time_min"].notna()].sort_values("time_min")
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def suggest_minifrac_mapping(columns: list[object]) -> dict[str, str | None]:
    """Suggest mapping for minifrac/DFIT shut-in data."""
    return suggest_column_mapping(columns, alias_patterns=MINIFRAC_ALIAS_PATTERNS)


def apply_minifrac_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Apply a minifrac/DFIT mapping."""
    selected: dict[str, pd.Series] = {}
    for internal, source in mapping.items():
        if not source or source == NO_MAPPING or source not in df.columns:
            continue
        selected[internal] = df[source]

    out = pd.DataFrame(selected)
    numeric_columns = {"time_min", "rate_bpm", "bhp_psi", "g_function", "shutin_psi", "straight_line_psi"}
    for column in numeric_columns.intersection(out.columns):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "time_min" in out.columns:
        out = out[out["time_min"].notna()].sort_values("time_min")
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def compute_g_function(
    df: pd.DataFrame,
    tp_min: float | None = None,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Compute the Nolte g-function for shut-in data.

    The implemented default is the common Carter leakoff / PKN alpha=0.5 form:
    g(DtD) = (4/3) * [(1 + DtD)^1.5 - DtD^1.5 - 1].
    """
    out = df.copy()
    if "time_min" not in out.columns:
        raise ValueError("compute_g_function requires a time_min column")

    if tp_min is None:
        if "rate_bpm" in out.columns:
            pumping_mask = out["rate_bpm"].fillna(0.0) > 0.1
            tp_min = float(out.loc[pumping_mask, "time_min"].max()) if pumping_mask.any() else float(out["time_min"].min())
        else:
            tp_min = float(out["time_min"].min())

    time = out["time_min"].to_numpy(dtype=float)
    delta_t = np.maximum(time - float(tp_min), 0.0)
    delta_t_d = delta_t / max(float(tp_min), 1e-9)

    if abs(alpha - 0.5) > 1e-9:
        # Keep alpha in the signature for future extension; alpha=0.5 is the
        # only formula implemented intentionally for now.
        raise NotImplementedError("Only alpha=0.5 Nolte g-function is currently implemented.")

    g_function = (4.0 / 3.0) * ((1.0 + delta_t_d) ** 1.5 - delta_t_d ** 1.5 - 1.0)
    out["delta_t_min"] = delta_t
    out["delta_t_dimensionless"] = delta_t_d
    out["g_function"] = g_function
    return out


def fit_closure_pressure(
    df: pd.DataFrame,
    g_col: str = "g_function",
    bhp_col: str = "bhp_psi",
    include_mask: pd.Series | None = None,
) -> dict[str, object]:
    """Fit BHP = Pc + m*g(t) and return closure-pressure diagnostics."""
    if g_col not in df.columns or bhp_col not in df.columns:
        raise ValueError(f"fit_closure_pressure requires {g_col!r} and {bhp_col!r} columns")

    if include_mask is None:
        include_mask = (
            (df[g_col] > 0.0)
            & df[g_col].notna()
            & df[bhp_col].notna()
            & (df[bhp_col] > 0.0)
        )

    g_vals = df.loc[include_mask, g_col].to_numpy(dtype=float)
    bhp_vals = df.loc[include_mask, bhp_col].to_numpy(dtype=float)

    if len(g_vals) < 2:
        empty = pd.Series(np.nan, index=df.index, dtype=float)
        return {
            "closure_pressure_psi": float("nan"),
            "slope_psi_per_g": float("nan"),
            "r_squared": float("nan"),
            "straight_line_psi": empty.rename("straight_line_psi"),
            "deviation_psi": empty.rename("deviation_psi"),
        }

    design = np.column_stack([np.ones_like(g_vals), g_vals])
    coeffs, *_ = np.linalg.lstsq(design, bhp_vals, rcond=None)
    closure_pressure, slope = float(coeffs[0]), float(coeffs[1])

    predicted_subset = design @ coeffs
    ss_res = float(np.sum((bhp_vals - predicted_subset) ** 2))
    ss_tot = float(np.sum((bhp_vals - bhp_vals.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    g_all = df[g_col].to_numpy(dtype=float)
    straight_line = closure_pressure + slope * g_all
    deviation = df[bhp_col].to_numpy(dtype=float) - straight_line

    return {
        "closure_pressure_psi": closure_pressure,
        "slope_psi_per_g": slope,
        "r_squared": r_squared,
        "straight_line_psi": pd.Series(straight_line, index=df.index, name="straight_line_psi"),
        "deviation_psi": pd.Series(deviation, index=df.index, name="deviation_psi"),
    }
