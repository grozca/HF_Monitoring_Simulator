from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


NO_MAPPING = "(not mapped)"


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
        "time min",
        "time_min",
        "time from start min",
        "t min",
    ],
    "phase": [
        "phase",
        "stage",
    ],
    "slurry_rate_bpm": [
        "rate_bpm",
        "slurry rate bpm",
        "slurry_rate_bpm",
        "bh injection rate bpm",
        "qi_liq bpm",
        "injection rate bpm",
        "rate bpm",
    ],
    "clean_rate_bpm": [
        "clean fluid bbl min",
        "clean rate bpm",
        "clean_rate_bpm",
        "clean fluid bpm",
    ],
    "surface_pressure_psi": [
        "surface_pressure_psi",
        "surface pressure psi",
        "surface treating pressure psi",
        "treating pressure psi",
    ],
    "ppa": [
        "proppant_ppa",
        "ppa",
        "prop conc",
        "proppant concentration",
        "cadd lbm gal",
        "lbm gal",
    ],
    "sand_rate_lb_min": [
        "sand_rate_lb_min",
        "sand rate lb min",
        "sand rate lbm min",
        "proppant rate",
    ],
    "pipe_friction_psi": [
        "pipe_friction_psi",
        "pipe friction psi",
    ],
    "perf_friction_psi": [
        "perf_friction_psi",
        "perf friction psi",
        "perforation friction psi",
    ],
    "bhp_psi": [
        "bhp_psi",
        "bhp psi",
        "bh pressure psi",
        "bottomhole pressure psi",
        "bottom hole pressure psi",
    ],
    "net_pressure_psi": [
        "net_pressure_psi",
        "net pressure psi",
        "net p",
        "pnet psi",
        "pnet psi",
    ],
    "alarm": [
        "alarm",
    ],
    "engineer_diagnosis": [
        "engineer_diagnosis",
        "engineer diagnosis",
        "diagnosis",
    ],
}


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _column_score(source_column: object, aliases: list[str]) -> int:
    normalized = normalize_column_name(source_column)
    if not normalized or normalized.startswith("unnamed"):
        return 0

    best_score = 0
    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if len(normalized_alias) < 3:
            continue
        if normalized == normalized_alias:
            best_score = max(best_score, 100)
        elif normalized_alias and normalized_alias in normalized:
            best_score = max(best_score, 80)
    return best_score


def suggest_column_mapping(columns: list[object]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
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
            mapping[standard.name] = scored[0][1]
            used_sources.add(scored[0][1])
        else:
            mapping[standard.name] = None

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
        "time_min",
        "slurry_rate_bpm",
        "clean_rate_bpm",
        "surface_pressure_psi",
        "ppa",
        "sand_rate_lb_min",
        "pipe_friction_psi",
        "perf_friction_psi",
        "bhp_psi",
        "net_pressure_psi",
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
        "slurry_rate_bpm",
        "surface_pressure_psi",
        "ppa",
        "bhp_psi",
        "net_pressure_psi",
        "pipe_friction_psi",
        "perf_friction_psi",
    ]
    available_curves = [
        column
        for column in curve_columns
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
