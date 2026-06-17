from __future__ import annotations

import numpy as np
import pandas as pd


def _trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 2:
        out["pressure_slope_psi_min"] = 0.0
        out["rate_slope_bpm_min"] = 0.0
        out["net_pressure_slope_psi_min"] = 0.0
        out["perf_friction_slope_psi_min"] = 0.0
        return out

    mean_dt = float(out["time_min"].diff().dropna().mean())
    window = max(1, int(round(3.0 / max(mean_dt, 0.01))))
    dt = out["time_min"].diff(window).replace(0.0, np.nan)

    out["pressure_slope_psi_min"] = out["surface_pressure_psi"].diff(window).div(dt)
    out["rate_slope_bpm_min"] = out["clean_rate_bpm"].diff(window).div(dt)
    out["net_pressure_slope_psi_min"] = out["net_pressure_psi"].diff(window).div(dt)
    out["perf_friction_slope_psi_min"] = out["perf_friction_psi"].diff(window).div(dt)

    trend_cols = [
        "pressure_slope_psi_min",
        "rate_slope_bpm_min",
        "net_pressure_slope_psi_min",
        "perf_friction_slope_psi_min",
    ]
    out[trend_cols] = out[trend_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = _trend_columns(df)

    alarms: list[str] = []
    diagnoses: list[str] = []

    for _, row in out.iterrows():
        alarm = "OK"
        diagnosis = "Normal treatment response"

        rate_stable = abs(row["rate_slope_bpm_min"]) < 3.0
        slurry_on = row["ppa"] > 0.4
        abnormal_event = row["event"] != "Normal"

        if row["event"] == "Screenout Trend" and slurry_on and (
            row["net_pressure_slope_psi_min"] > 25.0 or row["net_pressure_psi"] > 2050.0
        ):
            alarm = "HIGH PRESSURE TREND"
            diagnosis = "Possible screenout"

        if (
            row["event"] == "Perforation Restriction"
            and slurry_on
            and (row["perf_friction_slope_psi_min"] > 18.0 or row["perf_friction_psi"] > 900.0)
        ):
            alarm = "PERF FRICTION INCREASE"
            diagnosis = "Possible perforation or cluster restriction"

        if row["event"] == "Pump Rate Loss" and abs(row["rate_slope_bpm_min"]) > 2.5:
            alarm = "RATE INSTABILITY"
            diagnosis = "Pump or rate-control issue"

        if (
            row["pressure_slope_psi_min"] < -25.0
            and row["time_min"] > out["time_min"].max() * 0.35
            and row["event"] == "Frac Hit"
        ):
            alarm = "PRESSURE DROP"
            diagnosis = "Possible frac hit / pressure communication"

        if row["event"] == "Pressure Sensor Error" and abs(row["pressure_slope_psi_min"]) > 60.0:
            alarm = "DATA QUALITY"
            diagnosis = "Possible sensor or data quality issue"

        if row["surface_pressure_psi"] > row["treating_pressure_limit_psi"]:
            alarm = "PRESSURE LIMIT"
            diagnosis = "Pressure above treating limit"

        if row["net_pressure_psi"] > 2200.0:
            alarm = "HIGH NET PRESSURE"
            diagnosis = "Possible screenout"

        if row["perf_friction_psi"] > 1650.0:
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
    )
    critical = (
        out["alarm"].isin(["PRESSURE LIMIT", "HIGH NET PRESSURE", "PERF FRICTION HIGH"])
        | (out["surface_pressure_psi"] > out["treating_pressure_limit_psi"])
        | (out["net_pressure_psi"] > 2200.0)
        | (out["perf_friction_psi"] > 1650.0)
    )
    out["diagnostic_status"] = np.select(
        [critical, warning],
        ["Critical", "Warning"],
        default="Normal",
    )

    numeric_cols = [
        "pressure_slope_psi_min",
        "rate_slope_bpm_min",
        "net_pressure_slope_psi_min",
        "perf_friction_slope_psi_min",
    ]
    out[numeric_cols] = out[numeric_cols].round(2)
    return out


def evaluate_window(df: pd.DataFrame, current_minute: int | float | None = None) -> list[dict[str, str]]:
    if current_minute is None:
        window = df
    else:
        window = df[df["time_min"] <= current_minute]
    if window.empty:
        window = df.head(1)

    row = window.iloc[-1]
    alerts: list[dict[str, str]] = []

    if row.diagnostic_status == "Critical":
        alerts.append(
            {
                "severity": "Critical",
                "message": f"{row.alarm}: {row.engineer_diagnosis}.",
            }
        )
    elif row.diagnostic_status == "Warning":
        alerts.append(
            {
                "severity": "Warning",
                "message": f"{row.alarm}: {row.engineer_diagnosis}.",
            }
        )

    if row.event != "Normal":
        alerts.append(
            {
                "severity": "Info",
                "message": f"Scenario marker: {row.event}.",
            }
        )

    if row.net_pressure_psi > 1900.0 and row.alarm not in {"HIGH NET PRESSURE", "HIGH PRESSURE TREND"}:
        alerts.append(
            {
                "severity": "Warning",
                "message": "Net pressure is building above the normal trend.",
            }
        )

    if row.perf_friction_psi > 1150.0 and row.alarm not in {"PERF FRICTION HIGH", "PERF FRICTION INCREASE"}:
        alerts.append(
            {
                "severity": "Warning",
                "message": "Perf friction is increasing; watch for plugging.",
            }
        )

    if not alerts:
        alerts.append(
            {
                "severity": "Info",
                "message": "Job response is inside the expected training envelope.",
            }
        )

    return alerts[:4]
