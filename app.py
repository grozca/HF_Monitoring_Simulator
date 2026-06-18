from __future__ import annotations

from datetime import datetime
from html import escape
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.action_engine import action_options, process_player_action
from src.calibration_engine import (
    CalibrationConfig,
    align_simulation_to_reference,
    calculate_metrics_table,
    calculate_objective,
    estimate_lag_by_correlation,
    fit_surface_pressure_components,
    standardize_dataframe,
)
from src.data_loader import (
    describe_raw_file,
    get_sheet_columns,
    get_sheet_names,
    list_raw_files,
    load_table,
    preview_table,
    summarize_dataframe,
)
from src.column_mapping import (
    NO_MAPPING,
    STANDARD_COLUMNS,
    apply_column_mapping,
    clean_mapped_dataframe,
    mapping_options,
    mapping_readiness,
    suggest_column_mapping,
)
from src.equipment_engine import EquipmentConfig
from src.physics_engine import WellConfig, run_physics_engine
from src.planned_vs_actual import apply_planned_vs_actual, build_planned_actual_interpretation
from src.plots import (
    make_field_visual,
    make_formation_response_plot,
    make_planned_actual_ppa_plot,
    make_planned_actual_pressure_plot,
    make_planned_actual_rate_plot,
    make_planned_actual_sand_plot,
    make_pressure_decomposition_plot,
    make_sand_transport_plot,
    make_timeseries_plot,
)
from src.sand_transport import SandTransportConfig
from src.scenario_engine import SCENARIO_OPTIONS
from src.scoring import DIAGNOSIS_OPTIONS, SCENARIO_INFO
from src.treatment_schedule import TreatmentScheduleConfig, generate_treatment_schedule
from src.well_presets import WELL_PRESETS, get_preset


BRAND_NAME = "GROZFRAC"


st.set_page_config(
    page_title=f"{BRAND_NAME} Frac Monitoring Simulator",
    page_icon=None,
    layout="wide",
)


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --ui-font: 'JetBrains Mono', 'Courier New', Consolas, monospace;
    }

    html, body, [class*="css"], .stApp {
        background-color: #0a0c0f !important;
        color: #e2e8f0 !important;
        font-family: var(--ui-font) !important;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 1.4rem;
        max-width: 1680px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-radius: 3px;
        padding: 0.5rem 0.7rem;
        min-height: 72px;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.60rem !important;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    [data-testid="stMetricValue"] {
        color: #22d3ee !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebarHeader"] {
        min-height: 2.2rem;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.80rem !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #94a3b8 !important;
        letter-spacing: 0.08rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #64748b;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 2.05rem;
        font-weight: 700;
    }
    .side-panel {
        background: #0a0c0f;
        border: 1px solid #1e293b;
        border-radius: 3px;
        padding: 0.55rem 0.65rem;
        margin: 0.45rem 0 0.8rem 0;
    }
    .side-panel-title {
        color: #475569;
        font-size: 0.58rem;
        letter-spacing: 0.13rem;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .side-status-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #111827;
        padding-top: 0.38rem;
        margin-top: 0.25rem;
        color: #64748b;
        font-size: 0.68rem;
        letter-spacing: 0.03rem;
    }
    .side-status-line b {
        color: #22d3ee;
        font-weight: 700;
    }
    .side-subtle {
        color: #64748b;
        font-size: 0.65rem;
        line-height: 1.35;
        margin: 0.2rem 0 0.55rem 0;
    }
    /* Slider track */
    [data-testid="stSlider"] > div > div > div {
        background: #1e3a5f !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #1d4ed8 !important;
        color: #93c5fd !important;
        border-radius: 3px !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.70rem !important;
        letter-spacing: 0.08rem !important;
        text-transform: uppercase !important;
        padding: 0.35rem 0.9rem !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #e0f2fe !important;
        background: rgba(56, 189, 248, 0.06) !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        background: #0d1117;
        border-bottom: 1px solid #1e293b;
        gap: 0;
    }
    [data-testid="stTabs"] button[role="tab"] {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.10rem !important;
        text-transform: uppercase !important;
        color: #475569 !important;
        border-radius: 0 !important;
        padding: 0.5rem 1.0rem !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: transparent !important;
    }

    /* ── Alerts / status boxes ── */
    [data-testid="stAlert"] {
        border-radius: 3px !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.80rem !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.75rem !important;
    }

    /* ── Charts ── */
    [data-testid="stPlotlyChart"] {
        margin-bottom: 0.8rem;
    }

    /* ── Header bar ── */
    .groz-header {
        background: #111418;
        border-bottom: 2px solid #1a3a5c;
        padding: 0.42rem 0.85rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .groz-header-left,
    .groz-header-right {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        flex-wrap: wrap;
        min-width: 0;
    }
    .groz-logo {
        font-size: 0.88rem;
        font-weight: 700;
        color: #f59e0b;
        letter-spacing: 0.28rem;
    }
    .groz-meta {
        color: #64748b;
        font-size: 0.68rem;
        letter-spacing: 0.02rem;
        overflow-wrap: anywhere;
    }
    .groz-meta b { color: #38bdf8; }
    .groz-score {
        background: #0a0c0f;
        border: 1px solid #1e3a5f;
        color: #38bdf8;
        padding: 0.22rem 0.65rem;
        border-radius: 3px;
        font-size: 0.66rem;
        letter-spacing: 0.05rem;
        white-space: nowrap;
    }
    .groz-alarm {
        padding: 0.18rem 0.65rem;
        border-radius: 3px;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.09rem;
        white-space: nowrap;
    }
    .groz-alarm.ok       { background:#052e16; border:1px solid #16a34a; color:#4ade80; }
    .groz-alarm.warn     { background:#431407; border:1px solid #92400e; color:#f59e0b; }
    .groz-alarm.critical { background:#450a0a; border:1px solid #b91c1c; color:#f87171; }

    /* ── KPI strip ── */
    .kpi-strip {
        background: #0d1117;
        border-bottom: 1px solid #1e293b;
        border-top: 1px solid #1e293b;
        display: flex;
        flex-wrap: wrap;
        gap: 0;
        margin-bottom: 0.7rem;
    }
    .kpi-cell {
        padding: 0.45rem 0.85rem;
        border-right: 1px solid #1e293b;
        min-width: 100px;
        flex: 1;
    }
    .kpi-cell:last-child { border-right: none; }
    .kpi-label {
        font-size: 0.58rem;
        letter-spacing: 0.13rem;
        color: #475569;
        text-transform: uppercase;
        margin-bottom: 0.18rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .kpi-value {
        font-size: 1.15rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        line-height: 1.15;
    }
    .kpi-unit { font-size: 0.60rem; color: #64748b; }
    .kpi-cyan    { color: #22d3ee; }
    .kpi-green   { color: #4ade80; }
    .kpi-amber   { color: #f59e0b; }
    .kpi-red     { color: #f87171; }
    .kpi-white   { color: #f1f5f9; }
    .kpi-muted   { color: #64748b; }

    /* Phase badge inside KPI strip */
    .phase-badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #2563eb;
        color: #93c5fd;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        padding: 0.15rem 0.6rem;
        border-radius: 2px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .phase-badge.flush   { background:#0c2d1a; border-color:#166534; color:#4ade80; }
    .phase-badge.slurry  { background:#431407; border-color:#92400e; color:#f59e0b; }
    .phase-badge.pad     { background:#1e3a5f; border-color:#2563eb; color:#93c5fd; }

    /* ── Diagnostics sidebar cards ── */
    .diag-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .diag-section {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-radius: 3px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.5rem;
    }
    .diag-grid .diag-section { margin-bottom: 0; }
    .diag-section-title {
        font-size: 0.58rem;
        letter-spacing: 0.13rem;
        color: #475569;
        text-transform: uppercase;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.25rem;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .diag-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.18rem 0;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        color: #94a3b8;
        border-bottom: 1px solid #111418;
    }
    .diag-row:last-child { border-bottom: none; }
    .diag-row .dl { color: #64748b; font-size: 0.65rem; }
    .status-pill {
        padding: 0.08rem 0.45rem;
        border-radius: 2px;
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.05rem;
    }
    .pill-green  { background:#052e16; color:#4ade80; border:1px solid #166534; }
    .pill-amber  { background:#431407; color:#f59e0b; border:1px solid #92400e; }
    .pill-red    { background:#450a0a; color:#f87171; border:1px solid #b91c1c; }
    .pill-gray   { background:#1e293b; color:#64748b; border:1px solid #334155; }

    /* ── Nav bar ── */
    .nav-bar {
        background: #0d1117;
        border-bottom: 1px solid #1e293b;
        padding: 0.35rem 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.68rem;
        color: #475569;
    }
    .nav-label { color: #475569; font-size: 0.62rem; letter-spacing: 0.08rem; text-transform: uppercase; }
    .timebar-head {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-bottom: 0;
        border-radius: 3px 3px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        padding: 0.42rem 0.75rem 0.18rem 0.75rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .timebar-kicker {
        color: #64748b;
        font-size: 0.58rem;
        letter-spacing: 0.14rem;
        text-transform: uppercase;
    }
    .timebar-value {
        color: #22d3ee;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05rem;
    }
    .timebar-state {
        color: #4ade80;
        margin-left: 0.55rem;
    }
    [data-testid="stSlider"] {
        background: #0d1117;
        border-left: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 0 0 3px 3px;
        padding: 0.15rem 0.75rem 0.55rem 0.75rem;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: #22d3ee !important;
        border: 2px solid #0a0c0f !important;
        box-shadow: 0 0 0 1px #38bdf8, 0 0 12px rgba(34, 211, 238, 0.34) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
        }
        .groz-header {
            align-items: flex-start;
            gap: 0.45rem;
        }
        .groz-header-left,
        .groz-header-right {
            width: 100%;
            gap: 0.45rem 0.65rem;
        }
        .groz-logo {
            flex-basis: 100%;
            letter-spacing: 0.20rem;
        }
        .kpi-cell {
            flex: 1 1 33.333%;
            min-width: 0;
            border-bottom: 1px solid #1e293b;
        }
        .diag-grid {
            grid-template-columns: 1fr;
        }
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
        }
        [data-testid="stTabs"] button[role="tab"] {
            flex: 0 0 auto;
            white-space: nowrap;
        }
    }

    @media (max-width: 640px) {
        .groz-score,
        .groz-alarm {
            width: 100%;
            text-align: center;
        }
        .kpi-cell {
            flex-basis: 50%;
            padding: 0.42rem 0.58rem;
        }
        .kpi-label {
            font-size: 0.54rem;
            letter-spacing: 0.08rem;
        }
        .kpi-value {
            font-size: 0.98rem;
            overflow-wrap: anywhere;
        }
        .phase-badge {
            max-width: 100%;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .diag-row {
            gap: 0.75rem;
        }
    }

    /* dividers */
    hr { border-color: #1e293b !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ──────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "score": 0,
        "attempts": 0,
        "current_time": 0.0,
        "was_playing": False,
        "sidebar_playing": False,
        "replay_start": time.time(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def try_load_reference_excel() -> str:
    candidates = [
        Path("data/raw/synthetic_frac_realtime_sample.xlsx"),
        Path("data/raw/HF2D.xls"),
        Path("data/raw/MF.xls"),
    ]
    existing = [path.name for path in candidates if path.exists()]
    if not existing:
        return "Synthetic simulator active. No Excel files detected in data/raw."
    return "Reference files detected: " + ", ".join(existing)


# ── Simulation ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_job(
    scenario: str,
    duration_min: float,
    dt_min: float,
    target_rate_bpm: float,
    max_ppa: float,
    ppa_nudge: float,
    ppa_profile: str,
    severity: float,
    tvd_ft: float,
    measured_depth_ft: float,
    casing_id_in: float,
    wellbore_capacity_bbl_per_ft: float,
    mixing_efficiency: float,
    base_fluid_density_ppg: float,
    closure_gradient_psi_ft: float,
    number_of_pumps: int,
    pump_hhp: float,
    pump_efficiency: float,
    max_treating_pressure_psi: float,
    rate_capacity_bpm: float,
    seed: int,
):
    schedule = generate_treatment_schedule(
        TreatmentScheduleConfig(
            duration_min=duration_min,
            dt_min=dt_min,
            target_rate_bpm=target_rate_bpm,
            max_ppa=max_ppa,
            ppa_nudge=ppa_nudge,
            ppa_profile=ppa_profile,
        )
    )
    well = WellConfig(
        tvd_ft=tvd_ft,
        measured_depth_ft=measured_depth_ft,
        casing_id_in=casing_id_in,
        wellbore_capacity_bbl_per_ft=wellbore_capacity_bbl_per_ft,
        closure_gradient_psi_ft=closure_gradient_psi_ft,
        base_fluid_density_ppg=base_fluid_density_ppg,
        surface_pressure_limit_psi=max_treating_pressure_psi,
    )
    sand_transport = SandTransportConfig(
        measured_depth_ft=measured_depth_ft,
        wellbore_capacity_bbl_per_ft=wellbore_capacity_bbl_per_ft,
        mixing_efficiency=mixing_efficiency,
    )
    equipment = EquipmentConfig(
        number_of_pumps=number_of_pumps,
        pump_hhp=pump_hhp,
        pump_efficiency=pump_efficiency,
        max_treating_pressure_psi=max_treating_pressure_psi,
        rate_capacity_bpm=rate_capacity_bpm,
    )
    actual_df = run_physics_engine(
        schedule,
        scenario=scenario,
        target_rate_bpm=target_rate_bpm,
        duration_min=duration_min,
        severity=severity,
        seed=seed,
        well=well,
        sand_transport=sand_transport,
        equipment=equipment,
    )
    planned_reference_df = run_physics_engine(
        schedule,
        scenario="Normal Job",
        target_rate_bpm=target_rate_bpm,
        duration_min=duration_min,
        severity=severity,
        seed=seed,
        well=well,
        sand_transport=sand_transport,
        equipment=equipment,
    )
    enriched_df, event_log = apply_planned_vs_actual(actual_df, planned_reference_df)
    enriched_df.attrs["planned_actual_event_log"] = event_log
    return enriched_df


# ── Sidebar controls ───────────────────────────────────────────────────────────
def sidebar_controls() -> dict[str, float | int | str | bool]:
    with st.sidebar:
        st.header("Simulation Controls")
        preset_name = st.selectbox("Well preset", list(WELL_PRESETS.keys()))
        preset = get_preset(preset_name)
        preset_key = preset_name.lower().replace(" ", "_")
        st.caption(preset.description)

        scenario = st.selectbox("Scenario", SCENARIO_OPTIONS)
        duration_min = st.slider("Job duration, min", 40.0, 90.0, preset.duration_min, 5.0,
                                  key=f"duration:{preset_key}")
        dt_min = st.select_slider("Time step, min", options=[0.10, 0.25, 0.50, 1.00], value=0.25)

        st.markdown(
            '<div class="side-panel">'
            '<div class="side-panel-title">Operation Controls</div>'
            '<div class="side-subtle">Replay the treatment from the current clock position.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        op_cols = st.columns(3)
        with op_cols[0]:
            play_label = "PAUSE" if st.session_state.sidebar_playing else "PLAY"
            if st.button(play_label, key="sidebar_play_button"):
                st.session_state.sidebar_playing = not bool(st.session_state.sidebar_playing)
                st.session_state.was_playing = False
                st.rerun()
        with op_cols[1]:
            if st.button("RESET", key="sidebar_reset_button"):
                st.session_state.current_time = 0.0
                st.session_state.sidebar_playing = False
                st.session_state.was_playing = False
                st.rerun()
        with op_cols[2]:
            if st.button("STEP", key="sidebar_step_button"):
                st.session_state.current_time = min(
                    float(duration_min),
                    float(st.session_state.current_time) + float(dt_min),
                )
                st.session_state.sidebar_playing = False
                st.session_state.was_playing = False
                st.rerun()

        replay_speed = st.slider("Replay speed", 1, 10, 3, 1, key="sidebar_replay_speed")
        st.markdown(
            f'<div class="side-status-line"><span>Clock</span>'
            f'<b>{float(st.session_state.current_time):.2f} / {float(duration_min):.0f} min</b></div>',
            unsafe_allow_html=True,
        )

        target_rate_bpm = st.slider("Target slurry rate, bpm", 40.0, 110.0, preset.target_rate_bpm, 1.0,
                                     key=f"rate:{preset_key}")
        max_ppa = st.slider("Max PPA, lb/gal", 0.5, 5.0, preset.max_ppa, 0.25,
                             key=f"ppa:{preset_key}")
        ppa_profile_label = st.selectbox("PPA schedule", ["Field steps", "Linear ramp"], index=0,
                                          key=f"ppa_profile:{preset_key}")
        ppa_nudge = st.slider("Manual PPA nudge, lb/gal", -0.50, 0.75, 0.0, 0.05)
        severity = st.slider("Scenario severity", 0.5, 2.0, 1.0, 0.1)

        st.divider()
        st.subheader("Well / Formation")
        tvd_ft = st.slider("TVD, ft", 5000.0, 14000.0, preset.tvd_ft, 250.0,
                            key=f"tvd:{preset_key}")
        fluid_density = st.slider("Base fluid density, ppg", 8.2, 9.8, preset.base_fluid_density_ppg, 0.1,
                                   key=f"density:{preset_key}")
        closure_gradient = st.slider("Closure gradient, psi/ft", 0.55, 0.95, preset.closure_gradient_psi_ft, 0.01,
                                      key=f"closure:{preset_key}")

        st.divider()
        st.subheader("Sand Transport")
        measured_depth_ft = st.slider("Measured depth to perfs, ft", 8000.0, 24000.0, preset.measured_depth_ft, 250.0,
                                       key=f"md:{preset_key}")
        casing_id = st.slider("Casing ID, in", 3.50, 6.50, preset.casing_id_in, 0.05,
                               key=f"casing:{preset_key}")
        wellbore_capacity = st.slider("Wellbore capacity, bbl/ft", 0.006, 0.035, preset.wellbore_capacity_bbl_per_ft, 0.001,
                                       key=f"capacity:{preset_key}")
        mixing_efficiency = st.slider("Mixing efficiency", 0.50, 1.00, preset.mixing_efficiency, 0.05,
                                       key=f"mixing:{preset_key}")

        st.divider()
        st.subheader("Equipment / HHP")
        number_of_pumps = st.slider("Number of pumps", 4, 24, int(preset.number_of_pumps), 1,
                                     key=f"pumps:{preset_key}")
        pump_hhp = st.slider("Pump HHP", 1500.0, 3000.0, preset.pump_hhp, 50.0,
                              key=f"pump_hhp:{preset_key}")
        pump_efficiency = st.slider("Pump efficiency", 0.70, 0.98, preset.pump_efficiency, 0.01,
                                     key=f"pump_eff:{preset_key}")
        max_treating_pressure = st.slider("Max treating pressure, psi", 6000.0, 12000.0, preset.max_treating_pressure_psi, 250.0,
                                           key=f"max_pressure:{preset_key}")
        rate_capacity = st.slider("Rate capacity, bpm", 50.0, 130.0, preset.rate_capacity_bpm, 1.0,
                                   key=f"rate_capacity:{preset_key}")

        st.divider()
        seed = st.number_input("Random seed", value=7, step=1)
        st.info(try_load_reference_excel())

    return {
        "preset_name": preset_name,
        "scenario": scenario,
        "duration_min": duration_min,
        "dt_min": dt_min,
        "target_rate_bpm": target_rate_bpm,
        "max_ppa": max_ppa,
        "ppa_nudge": ppa_nudge,
        "ppa_profile": "stepped" if ppa_profile_label == "Field steps" else "linear",
        "severity": severity,
        "tvd_ft": tvd_ft,
        "measured_depth_ft": measured_depth_ft,
        "casing_id": casing_id,
        "wellbore_capacity": wellbore_capacity,
        "mixing_efficiency": mixing_efficiency,
        "fluid_density": fluid_density,
        "closure_gradient": closure_gradient,
        "number_of_pumps": number_of_pumps,
        "pump_hhp": pump_hhp,
        "pump_efficiency": pump_efficiency,
        "max_treating_pressure": max_treating_pressure,
        "rate_capacity": rate_capacity,
        "seed": int(seed),
        "playing": bool(st.session_state.sidebar_playing),
        "replay_speed": replay_speed,
    }


# ── Formatters ─────────────────────────────────────────────────────────────────
def fmt_pressure(v: float) -> str: return f"{v:,.0f}"
def fmt_rate(v: float) -> str:     return f"{v:.1f}"
def fmt_ppa(v: float) -> str:      return f"{v:.2f}"
def fmt_grad(v: float) -> str:     return f"{v:.2f}"
def fmt_min(v: float, active: bool = True) -> str:
    return "N/A" if (not active or pd.isna(v)) else f"{v:.2f}"


# ── Header bar ─────────────────────────────────────────────────────────────────
def render_console_header(row: pd.Series, controls: dict) -> None:
    diagnostic_status = str(row.get("diagnostic_status", "Normal"))
    alarm_class  = {"Normal": "ok", "Warning": "warn"}.get(diagnostic_status, "critical")
    alarm_label  = {"Normal": "● ALARM OK", "Warning": "⚠ WARNING"}.get(diagnostic_status, "✖ CRITICAL")

    now        = datetime.now().strftime("%m/%d/%Y  %H:%M:%S")
    well_name  = escape(str(controls.get("preset_name", "Training Well")).upper())
    scenario   = escape(str(controls.get("scenario", "Normal Job")).upper())
    phase      = escape(str(row.get("phase", "Unknown")).upper())
    tvd        = float(row.get("tvd_ft", controls.get("tvd_ft", 0.0)))
    score      = int(st.session_state.get("score", 0))
    attempts   = int(st.session_state.get("attempts", 0))

    st.markdown(
        f"""
        <div class="groz-header">
            <div class="groz-header-left">
                <span class="groz-logo">{BRAND_NAME}</span>
                <span class="groz-meta">WELL: <b>{well_name}</b></span>
                <span class="groz-meta">SCENARIO: <b>{scenario}</b></span>
                <span class="groz-meta">PHASE: <b>{phase}</b></span>
                <span class="groz-meta">TVD: <b>{tvd:,.0f} ft</b></span>
            </div>
            <div class="groz-header-right">
                <span class="groz-meta">{now}</span>
                <span class="groz-score">SCORE: {score:+d}&nbsp;&nbsp;|&nbsp;&nbsp;ATTEMPTS: {attempts}</span>
                <span class="groz-alarm {alarm_class}">{alarm_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Replay logic ───────────────────────────────────────────────────────────────
def advance_replay(duration_min: float, dt_min: float, playing: bool, replay_speed: int) -> None:
    if playing and not st.session_state.was_playing:
        st.session_state.replay_start = time.time() - (
            st.session_state.current_time / max(replay_speed, 1)
        )
    if playing:
        elapsed = (time.time() - st.session_state.replay_start) * replay_speed
        stepped_time = round(elapsed / dt_min) * dt_min
        st.session_state.current_time = min(duration_min, stepped_time)
    st.session_state.current_time = min(float(st.session_state.current_time), duration_min)
    st.session_state.was_playing = playing


# ── Navigation bar ─────────────────────────────────────────────────────────────
def render_navigation(duration_min: float, dt_min: float, playing: bool) -> None:
    nav_cols = st.columns([4, 1, 1])
    with nav_cols[0]:
        current_time = float(min(st.session_state.current_time, duration_min))
        state_label = "LIVE" if playing else "MANUAL"
        st.markdown(
            f"""
            <div class="timebar-head">
                <span class="timebar-kicker">Stage Clock</span>
                <span class="timebar-value">{current_time:.2f} / {duration_min:.0f} min
                    <span class="timebar-state">{state_label}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if playing:
            st.slider(
                "Stage clock, min", 0.0, float(duration_min),
                current_time, float(dt_min), disabled=True,
                label_visibility="collapsed",
                key="operation_time_slider_disabled",
            )
        else:
            st.session_state.current_time = st.slider(
                "Stage clock, min", 0.0, float(duration_min),
                current_time, float(dt_min),
                label_visibility="collapsed",
                key="operation_time_slider",
            )
    with nav_cols[1]:
        if st.button("↺  Reset"):
            st.session_state.current_time = 0.0
            st.rerun()
    with nav_cols[2]:
        if st.button("▶  Advance"):
            st.session_state.current_time = min(float(duration_min),
                float(st.session_state.current_time) + float(dt_min))
            st.rerun()


# ── KPI strip (replaces the old st.metric grid) ────────────────────────────────
def _kpi(label: str, value: str, unit: str, color_class: str = "kpi-cyan") -> str:
    return (
        f'<div class="kpi-cell">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value {color_class}">{value}'
        f'    <span class="kpi-unit">{unit}</span>'
        f'  </div>'
        f'</div>'
    )


def _phase_badge(phase: str) -> str:
    p = phase.lower()
    cls = "flush" if "flush" in p else ("slurry" if "slurry" in p else "pad")
    return (
        f'<div class="kpi-cell">'
        f'  <div class="kpi-label">Phase</div>'
        f'  <div class="phase-badge {cls}">{phase.upper()}</div>'
        f'</div>'
    )


def _risk_color(risk_pct: float) -> str:
    if risk_pct >= 60:
        return "kpi-red"
    if risk_pct >= 30:
        return "kpi-amber"
    return "kpi-green"


def render_kpi_strip(row: pd.Series) -> None:
    """Top KPI strip — primary numbers a field operator stares at."""
    risk_pct = float(row.get("screenout_risk_pct", 100.0 * row.get("screenout_risk", 0.0)))
    hhp_util = 100.0 * float(row.get("hhp_utilization", 0.0))

    cells = "".join([
        _phase_badge(str(row.get("phase", "—"))),
        _kpi("Slurry Rate", fmt_rate(row["slurry_rate_bpm"]), "bpm", "kpi-green"),
        _kpi("Surf Pressure", fmt_pressure(row["surface_pressure_psi"]), "psi", "kpi-cyan"),
        _kpi("BH Pressure", fmt_pressure(row["bhp_psi"]), "psi", "kpi-red"),
        _kpi("Net Pressure", fmt_pressure(row["net_pressure_psi"]), "psi", "kpi-amber"),
        _kpi("Surf PPA", fmt_ppa(row["surface_ppa"]), "lb/gal",
             "kpi-green" if row["surface_ppa"] > 0.05 else "kpi-muted"),
        _kpi("BH PPA", fmt_ppa(row["bottomhole_ppa"]), "lb/gal",
             "kpi-amber" if row["bottomhole_ppa"] > 0.05 else "kpi-muted"),
        _kpi("BH Gradient", fmt_grad(row["bhp_gradient_psi_ft"]), "psi/ft", "kpi-white"),
        _kpi("HHP Util", f"{hhp_util:.0f}", "%",
             "kpi-green" if hhp_util < 75 else "kpi-amber"),
        _kpi("Screenout Risk", f"{risk_pct:.0f}", "%", _risk_color(risk_pct)),
    ])
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


# ── Diagnostic panel (replaces old sand_cols / equipment_cols / formation_cols) ─
def render_field_visual_kpis(row: pd.Series) -> None:
    """Compact KPI row above the subsurface field visual."""
    sand_at_perfs = bool(row.get("sand_arrived_at_perfs", row.get("bottomhole_ppa", 0.0) > 0.05))
    surface_ppa = float(row.get("surface_ppa", 0.0))
    risk_pct = float(row.get("screenout_risk_pct", 100.0 * row.get("screenout_risk", 0.0)))
    half_length = float(row.get("fracture_half_length_ft", row.get("fracture_length_ft", 0.0)))
    frac_height = float(row.get("fracture_height_ft", 0.0))

    cells = "".join([
        _phase_badge(str(row.get("phase", "-"))),
        _kpi("Surf Press", fmt_pressure(row.get("surface_pressure_psi", 0.0)), "psi", "kpi-cyan"),
        _kpi("BH Press", fmt_pressure(row.get("bhp_psi", 0.0)), "psi", "kpi-red"),
        _kpi("Surface PPA", fmt_ppa(surface_ppa), "lb/gal",
             "kpi-amber" if surface_ppa > 0.05 else "kpi-muted"),
        _kpi("Sand @ Perfs", "YES" if sand_at_perfs else "NO", "",
             "kpi-green" if sand_at_perfs else "kpi-muted"),
        _kpi("Half-Length", f"{half_length:,.0f}", "ft",
             "kpi-amber" if half_length > 0 else "kpi-muted"),
        _kpi("Frac Height", f"{frac_height:,.0f}", "ft",
             "kpi-white" if frac_height > 0 else "kpi-muted"),
        _kpi("Risk", f"{risk_pct:.0f}", "%", _risk_color(risk_pct)),
    ])
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def render_diagnostic_panel(row: pd.Series) -> None:
    """Three compact HTML cards: sand transport, equipment, formation geometry."""

    def _pill(val: bool | str, true_label: str = "YES", false_label: str = "NO") -> str:
        ok = val if isinstance(val, bool) else str(val).lower() in ("yes", "true", "1")
        cls = "pill-green" if ok else "pill-gray"
        lbl = true_label if ok else false_label
        return f'<span class="status-pill {cls}">{lbl}</span>'

    def _val(v: str, color: str = "#94a3b8") -> str:
        return f'<span style="color:{color};font-size:0.78rem;">{v}</span>'

    sand_lag    = fmt_min(row["sand_lag_min"], row["slurry_rate_bpm"] > 1.0)
    press_marg  = float(row.get("pressure_margin_psi", 0.0))
    marg_color  = "#4ade80" if press_marg > 2000 else ("#f59e0b" if press_marg > 500 else "#f87171")
    equip_stat  = str(row.get("equipment_status", "Available"))
    equip_pill  = "pill-green" if equip_stat.lower() == "available" else "pill-amber"

    sand_html = f"""
    <div class="diag-section">
      <div class="diag-section-title">Sand Transport</div>
      <div class="diag-row"><span class="dl">Sand lag</span>{_val(f'{sand_lag} min')}</div>
      <div class="diag-row"><span class="dl">Sand @ perfs</span>{_pill(bool(row['sand_arrived_at_perfs']))}</div>
      <div class="diag-row"><span class="dl">Flush @ perfs</span>{_pill(bool(row['flush_arrived_at_perfs']))}</div>
      <div class="diag-row"><span class="dl">Sand in WB</span>{_val(f"{row['sand_in_wellbore_lb']:,.0f} lb")}</div>
    </div>
    """

    equip_html = f"""
    <div class="diag-section">
      <div class="diag-section-title">Equipment / HHP</div>
      <div class="diag-row"><span class="dl">Rate</span>{_val(f"{row.get('rate_bph',0):,.0f} BPH")}</div>
      <div class="diag-row"><span class="dl">HHP required</span>{_val(f"{row.get('hhp_required',0):,.0f} HHP")}</div>
      <div class="diag-row"><span class="dl">Press margin</span>{_val(f"{press_marg:,.0f} psi", marg_color)}</div>
      <div class="diag-row"><span class="dl">Equipment</span>
        <span class="status-pill {equip_pill}">{equip_stat.upper()}</span></div>
    </div>
    """

    form_html = f"""
    <div class="diag-section">
      <div class="diag-section-title">Fracture Geometry</div>
      <div class="diag-row"><span class="dl">Formation</span>{_val(str(row.get('formation_state','—')))}</div>
      <div class="diag-row"><span class="dl">Frac width</span>{_val(f"{row.get('fracture_width_in',0):.2f} in")}</div>
      <div class="diag-row"><span class="dl">Half-length</span>{_val(f"{row.get('fracture_half_length_ft',0):,.0f} ft")}</div>
      <div class="diag-row"><span class="dl">Acceptance</span>{_val(f"{row.get('formation_acceptance_pct',0):.0f}%", '#4ade80')}</div>
    </div>
    """

    # Auto-diagnosis banner
    diag_msg = str(row.get("engineer_diagnosis", "Normal treatment response"))
    alarm_msg = str(row.get("alarm", "OK"))
    diag_status = str(row.get("diagnostic_status", "Normal"))
    diag_color = {"Normal": "#052e16", "Warning": "#431407"}.get(diag_status, "#450a0a")
    diag_border = {"Normal": "#16a34a", "Warning": "#92400e"}.get(diag_status, "#b91c1c")
    diag_text   = {"Normal": "#4ade80", "Warning": "#f59e0b"}.get(diag_status, "#f87171")
    diag_html = f"""
    <div style="background:{diag_color};border:1px solid {diag_border};border-radius:3px;
                padding:0.45rem 0.75rem;margin-bottom:0.5rem;
                font-family:'JetBrains Mono','Courier New',monospace;font-size:0.72rem;color:{diag_text};">
      <span style="font-weight:700;letter-spacing:0.08rem;">ALARM: {alarm_msg.upper()}</span>
      &nbsp;|&nbsp; {diag_msg}
    </div>
    """

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown(diag_html, unsafe_allow_html=True)
        if row["surface_ppa"] > 0.05 and row["bottomhole_ppa"] <= 0.05:
            st.markdown(
                f'<div style="background:#1e3a5f;border:1px solid #2563eb;border-radius:3px;'
                f'padding:0.35rem 0.75rem;font-family:Courier New,monospace;font-size:0.72rem;color:#93c5fd;">'
                f'⚡ Sand in transit — lag: {row["sand_lag_min"]:.2f} min</div>',
                unsafe_allow_html=True,
            )
    with right_col:
        pass  # diagnostic cards render below as a full-width block

    st.markdown(sand_html + equip_html + form_html, unsafe_allow_html=True)


# ── Full status render (called from main) ──────────────────────────────────────
def render_status(row: pd.Series) -> None:
    render_kpi_strip(row)
    render_diagnostic_panel(row)


# ── Training tab ───────────────────────────────────────────────────────────────
def find_event_onset_min(df: pd.DataFrame, scenario: str) -> float | None:
    if scenario == "Normal Job" or "event" not in df.columns:
        return None
    event_rows = df[df["event"].astype(str) != "Normal"]
    if event_rows.empty:
        return None
    return float(event_rows["time_min"].iloc[0])


def score_diagnosis_only(scenario: str, diagnosis: str) -> tuple[int, str]:
    info = SCENARIO_INFO.get(scenario)
    if info is None:
        return 0, "Scenario is not configured."
    if diagnosis == info.expected_diagnosis:
        return 10, "Diagnosis correct."
    return -5, f"Diagnosis mismatch. Expected: {info.expected_diagnosis}."


def render_training_tab(scenario: str, df: pd.DataFrame, current_time: float) -> None:
    st.markdown("### Diagnose the current situation")
    current_idx = int(np.argmin(np.abs(df["time_min"].to_numpy(float) - current_time)))
    current_row = df.iloc[current_idx]
    st.caption(
        f"Training clock: {float(current_row['time_min']):.2f} min  |  "
        f"Auto-diagnosis: {current_row.get('engineer_diagnosis', 'Unknown')}"
    )

    train_cols = st.columns(2)
    with train_cols[0]:
        user_diagnosis = st.selectbox("Your diagnosis", DIAGNOSIS_OPTIONS)
    with train_cols[1]:
        user_action = st.selectbox("Your action", action_options())

    evidence_cols = st.columns(5)
    evidence_cols[0].metric("Rate",        f"{current_row.get('slurry_rate_bpm', 0.0):.1f} bpm")
    evidence_cols[1].metric("BH PPA",      f"{current_row.get('bottomhole_ppa', 0.0):.2f}")
    evidence_cols[2].metric("Net Pressure",f"{current_row.get('net_pressure_psi', 0.0):,.0f} psi")
    evidence_cols[3].metric("Acceptance",  f"{100.0 * current_row.get('acceptance_index', 1.0):.0f}%")
    evidence_cols[4].metric("Risk",        f"{100.0 * current_row.get('screenout_risk', 0.0):.0f}%")

    if st.button("Submit diagnosis and apply action", type="primary"):
        diagnosis_score, diagnosis_feedback = score_diagnosis_only(scenario, user_diagnosis)
        try:
            action_result = process_player_action(
                df,
                time_min=float(current_row["time_min"]),
                scenario=scenario,
                diagnosis=user_diagnosis,
                selected_action=user_action,
                event_onset_min=find_event_onset_min(df, scenario),
            )
        except Exception as exc:
            st.error(f"Could not process action: {exc}")
            return

        total_score = diagnosis_score + action_result.score_delta
        st.session_state.score   += total_score
        st.session_state.attempts += 1

        score_cols = st.columns(4)
        score_cols[0].metric("Diagnosis",   f"{diagnosis_score:+d}")
        score_cols[1].metric("Action",      f"{action_result.score_delta:+d}")
        score_cols[2].metric("Total",       f"{total_score:+d}")
        score_cols[3].metric("Risk Change", f"{100.0 * action_result.risk_delta:+.1f} pp")

        (st.success if diagnosis_score > 0 else st.error)(diagnosis_feedback)
        quality_text = (
            f"Action quality: **{action_result.action_quality}** | "
            f"Recommended: **{action_result.recommended_action}**"
        )
        if action_result.score_delta > 0:
            st.success(quality_text)
        elif action_result.score_delta < 0:
            st.error(quality_text)
        else:
            st.info(quality_text)

        with st.expander("Action explanation and evidence", expanded=True):
            st.markdown(action_result.explanation.replace("\n", "\n\n"))
            st.dataframe(pd.DataFrame({"Evidence": action_result.evidence}),
                         width="stretch", hide_index=True)

        preview_start = max(float(df["time_min"].min()), float(current_row["time_min"]) - 3.0)
        preview_end   = min(float(df["time_min"].max()), float(current_row["time_min"]) + 14.0)
        preview_df    = action_result.modified_df[
            action_result.modified_df["time_min"].between(preview_start, preview_end)
        ].copy()
        preview_cols  = tuple(
            col for col in ("surface_pressure_psi", "net_pressure_psi", "bottomhole_ppa", "slurry_rate_bpm")
            if col in preview_df.columns
        )
        if preview_cols:
            st.plotly_chart(
                make_timeseries_plot(preview_df, preview_cols, "Action Response Preview",
                                     current_minute=float(current_row["time_min"]), height=380),
                width="stretch",
            )

    info = SCENARIO_INFO[scenario]
    st.markdown("### Scenario Teaching Note")
    st.info(
        f"Expected diagnosis: **{info.expected_diagnosis}**\n\n"
        f"Recommended action: **{info.recommended_action}**\n\n"
        f"{info.training_note}"
    )


# ── Excel Explorer ─────────────────────────────────────────────────────────────
def render_excel_explorer_tab() -> None:
    st.markdown("### Raw Data Explorer")
    raw_files = list_raw_files()

    if not raw_files:
        st.info(
            "No Excel or CSV files detected yet. Copy your files into data/raw/: "
            "MF.xls, HF2D.xls, or synthetic_frac_realtime_sample.xlsx."
        )
        st.code(
            "data/raw/MF.xls\n"
            "data/raw/HF2D.xls\n"
            "data/raw/synthetic_frac_realtime_sample.xlsx",
            language="text",
        )
        return

    file_options = {f"{file.name} ({file.size_kb} KB)": file for file in raw_files}
    selected_label = st.selectbox("Raw file", list(file_options.keys()))
    selected_file  = file_options[selected_label]

    try:
        metadata = describe_raw_file(selected_file.path)
        sheets   = get_sheet_names(selected_file.path)
    except Exception as exc:
        st.error(f"Could not inspect file: {exc}")
        return

    meta_cols = st.columns(4)
    meta_cols[0].metric("File",   str(metadata["name"]))
    meta_cols[1].metric("Type",   str(metadata["suffix"]))
    meta_cols[2].metric("Size",   f"{metadata['size_kb']} KB")
    meta_cols[3].metric("Sheets", str(metadata["sheet_count"]))

    controls = st.columns([2, 1, 1])
    with controls[0]: selected_sheet = st.selectbox("Sheet", sheets)
    with controls[1]: header_row     = st.number_input("Header row", min_value=0, max_value=50, value=0, step=1)
    with controls[2]: preview_rows   = st.selectbox("Preview rows", [10, 25, 50, 100], index=1)

    try:
        columns    = get_sheet_columns(selected_file.path, sheet_name=selected_sheet, header_row=int(header_row))
        preview_df = preview_table(selected_file.path, sheet_name=selected_sheet,
                                   header_row=int(header_row), rows=int(preview_rows))
    except Exception as exc:
        st.error(f"Could not read selected sheet with header row {header_row}: {exc}")
        st.caption("Try changing the header row. Some field files have title rows above the actual table.")
        return

    summary = summarize_dataframe(preview_df)
    st.markdown("### Columns")
    st.dataframe(pd.DataFrame({"column": columns}), width="stretch", hide_index=True)

    st.markdown("### Preview")
    preview_metrics = st.columns(4)
    preview_metrics[0].metric("Preview rows",  summary["rows"])
    preview_metrics[1].metric("Columns",       summary["columns"])
    preview_metrics[2].metric("Numeric cols",  len(summary["numeric_columns"]))
    preview_metrics[3].metric("Empty cols",    len(summary["empty_columns"]))
    st.dataframe(preview_df, width="stretch", hide_index=True)

    with st.expander("Column type hints"):
        st.write("Numeric columns")
        st.code("\n".join(summary["numeric_columns"]) or "None", language="text")
        st.write("Datetime columns")
        st.code("\n".join(summary["datetime_columns"]) or "None", language="text")
        st.write("Empty columns")
        st.code("\n".join(summary["empty_columns"]) or "None", language="text")

    st.markdown("### Column Mapper")
    st.caption(
        "Map raw Excel columns into the simulator's standard names. "
        "These mappings are only for inspection right now; calibration comes next."
    )
    suggested_mapping = suggest_column_mapping(columns)
    options            = mapping_options(columns)
    mapper_cols        = st.columns(2)
    selected_mapping: dict[str, str | None] = {}
    map_key_base = f"{selected_file.name}:{selected_sheet}:{int(header_row)}"

    for index, standard_column in enumerate(STANDARD_COLUMNS):
        default_source = suggested_mapping.get(standard_column.name)
        default_option = default_source if default_source in options else NO_MAPPING
        with mapper_cols[index % 2]:
            selected = st.selectbox(
                standard_column.label, options,
                index=options.index(default_option),
                key=f"map:{map_key_base}:{standard_column.name}",
            )
        selected_mapping[standard_column.name] = None if selected == NO_MAPPING else selected

    mapped_preview = clean_mapped_dataframe(apply_column_mapping(preview_df, selected_mapping))
    readiness      = mapping_readiness(mapped_preview)

    map_metrics = st.columns(4)
    map_metrics[0].metric("Mapped cols",  len(readiness["mapped_columns"]))
    map_metrics[1].metric("Usable rows",  readiness["usable_rows"])
    map_metrics[2].metric("Curve ready", "Yes" if readiness["can_plot"] else "No")
    map_metrics[3].metric("Curves",       len(readiness["available_curves"]))

    if not mapped_preview.empty:
        st.markdown("### Standardized Preview")
        st.dataframe(mapped_preview, width="stretch", hide_index=True)

        if readiness["can_plot"]:
            curve = st.selectbox("Preview mapped curve", readiness["available_curves"],
                                  key=f"curve:{map_key_base}")
            st.plotly_chart(
                make_timeseries_plot(mapped_preview, (curve,), f"Mapped Preview: {curve}"),
                width="stretch",
            )
    else:
        st.info("No standard columns are mapped yet.")

    if st.button("Load full selected sheet", type="primary"):
        try:
            full_df = load_table(selected_file.path, sheet_name=selected_sheet, header_row=int(header_row))
        except Exception as exc:
            st.error(f"Could not load full sheet: {exc}")
            return

        full_summary = summarize_dataframe(full_df)
        st.success(f"Loaded {full_summary['rows']} rows and {full_summary['columns']} columns.")
        st.dataframe(full_df.head(200), width="stretch", hide_index=True)

        full_mapped = clean_mapped_dataframe(apply_column_mapping(full_df, selected_mapping))
        if not full_mapped.empty:
            mapped_summary = summarize_dataframe(full_mapped)
            st.markdown("### Full Standardized Data")
            st.success(f"Mapped {mapped_summary['rows']} rows and {mapped_summary['columns']} standard columns.")
            st.dataframe(full_mapped.head(200), width="stretch", hide_index=True)
            csv = full_mapped.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download standardized mapped CSV", data=csv,
                file_name=f"mapped_{selected_file.path.stem}_{selected_sheet}.csv",
                mime="text/csv",
            )


# ── Calibration tab ────────────────────────────────────────────────────────────
CALIBRATION_VARIABLES = (
    "surface_pressure_psi", "bhp_psi", "net_pressure_psi",
    "slurry_rate_bpm", "clean_rate_bpm", "ppa",
    "surface_ppa", "bottomhole_ppa",
    "pipe_friction_psi", "perf_friction_psi", "screenout_risk",
)


def render_calibration_tab(sim_df: pd.DataFrame) -> None:
    st.markdown("### Calibration Against Field Data")
    raw_files = list_raw_files()

    if not raw_files:
        st.info("No Excel or CSV files detected in data/raw.")
        return

    file_options   = {f"{file.name} ({file.size_kb} KB)": file for file in raw_files}
    selected_label = st.selectbox("Calibration raw file", list(file_options.keys()), key="calibration:file")
    selected_file  = file_options[selected_label]

    try:
        sheets = get_sheet_names(selected_file.path)
    except Exception as exc:
        st.error(f"Could not inspect file: {exc}")
        return

    controls = st.columns([2, 1, 1, 1])
    with controls[0]: selected_sheet = st.selectbox("Calibration sheet", sheets, key="calibration:sheet")
    with controls[1]: header_row     = st.number_input("Calibration header row", min_value=0, max_value=50,
                                                        value=0, step=1, key="calibration:header")
    with controls[2]: dt_min         = st.select_slider("Calibration dt, min",
                                                          options=[0.10, 0.25, 0.50, 1.00], value=0.25,
                                                          key="calibration:dt")
    with controls[3]: max_lag_min    = st.slider("Max lag, min", 0.0, 12.0, 8.0, 0.5, key="calibration:max_lag")

    try:
        raw_df       = load_table(selected_file.path, sheet_name=selected_sheet, header_row=int(header_row))
        reference_df = standardize_dataframe(raw_df)
    except Exception as exc:
        st.error(f"Could not standardize reference data: {exc}")
        st.caption("Try another header row if the Excel has title blocks above the real column names.")
        return

    available_variables    = [col for col in CALIBRATION_VARIABLES if col in sim_df.columns and col in reference_df.columns]
    mapped_reference_cols  = [col for col in reference_df.columns if col in CALIBRATION_VARIABLES or col == "time_min"]

    status_cols = st.columns(4)
    status_cols[0].metric("Reference Rows", f"{len(reference_df):,}")
    status_cols[1].metric("Mapped Columns", len(mapped_reference_cols))
    status_cols[2].metric("Common Curves",  len(available_variables))
    status_cols[3].metric("Sim Rows",       f"{len(sim_df):,}")

    with st.expander("Standardized reference preview"):
        st.dataframe(
            reference_df[mapped_reference_cols].head(120) if mapped_reference_cols else reference_df.head(120),
            width="stretch", hide_index=True,
        )

    if not available_variables:
        st.warning(
            "No common calibration curves were found yet. Confirm that the Excel has time plus at least "
            "one curve like surface pressure, rate, PPA, BHP, or net pressure."
        )
        return

    default_variables = [
        col for col in ("surface_pressure_psi", "net_pressure_psi", "slurry_rate_bpm", "ppa")
        if col in available_variables
    ] or available_variables[: min(3, len(available_variables))]

    variables = st.multiselect("Curves to compare", available_variables,
                                default=default_variables, key="calibration:variables")

    tuning_cols = st.columns([1, 1, 2])
    with tuning_cols[0]: smoothing_window = st.select_slider("Smoothing window", options=[1,3,5,7,9], value=1,
                                                               key="calibration:smoothing")
    with tuning_cols[1]: apply_lag        = st.checkbox("Estimate lag", value=True, key="calibration:lag")
    with tuning_cols[2]: lag_variable     = st.selectbox("Lag reference curve",
                                                           variables if variables else available_variables,
                                                           index=0, key="calibration:lag_variable")

    if not variables:
        st.info("Select at least one curve to run calibration metrics.")
        return

    if st.button("Run calibration metrics", type="primary", key="calibration:run"):
        cfg = CalibrationConfig(
            dt_min=float(dt_min),
            max_lag_min=float(max_lag_min),
            apply_lag_correction=bool(apply_lag),
            smoothing_window=int(smoothing_window),
        )
        try:
            lag_min = (estimate_lag_by_correlation(sim_df, reference_df, variable=lag_variable, cfg=cfg)
                       if apply_lag else 0.0)
            aligned = align_simulation_to_reference(sim_df, reference_df, variables, cfg=cfg, lag_min=lag_min)
            metrics = calculate_metrics_table(aligned, variables, cfg=cfg)
            objective = calculate_objective(metrics)
        except Exception as exc:
            st.error(f"Calibration failed: {exc}")
            return

        result_cols = st.columns(4)
        result_cols[0].metric("Objective",      f"{objective:.4f}")
        result_cols[1].metric("Estimated Lag",  f"{lag_min:+.2f} min")
        result_cols[2].metric("Aligned Points", f"{len(aligned):,}")
        result_cols[3].metric("Curves Used",    len(variables))

        metric_cols = ["variable","rmse","nrmse","bias","bias_norm","max_abs","slope_nrmse","corr","r2","n_points"]
        st.dataframe(
            metrics[[col for col in metric_cols if col in metrics.columns]].round(4),
            width="stretch", hide_index=True,
        )

        plot_variable = variables[0]
        plot_df = aligned[["time_min", f"sim_{plot_variable}", f"ref_{plot_variable}"]].rename(
            columns={f"sim_{plot_variable}": f"sim {plot_variable}",
                     f"ref_{plot_variable}": f"ref {plot_variable}"}
        )
        st.plotly_chart(
            make_timeseries_plot(plot_df, (f"sim {plot_variable}", f"ref {plot_variable}"),
                                  f"Simulation vs Reference: {plot_variable}", height=420),
            width="stretch",
        )

        if "surface_pressure_psi" in reference_df.columns:
            with st.expander("Surface pressure component fit"):
                try:
                    fit = fit_surface_pressure_components(sim_df, reference_df, cfg=cfg)
                    st.dataframe(
                        pd.DataFrame([{"component": k, "coefficient": v} for k, v in fit.items()]),
                        width="stretch", hide_index=True,
                    )
                except Exception as exc:
                    st.caption(f"Component fit unavailable: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    init_session_state()
    controls = sidebar_controls()
    advance_replay(
        duration_min=float(controls["duration_min"]),
        dt_min=float(controls["dt_min"]),
        playing=bool(controls["playing"]),
        replay_speed=int(controls["replay_speed"]),
    )

    df = simulate_job(
        scenario=str(controls["scenario"]),
        duration_min=float(controls["duration_min"]),
        dt_min=float(controls["dt_min"]),
        target_rate_bpm=float(controls["target_rate_bpm"]),
        max_ppa=float(controls["max_ppa"]),
        ppa_nudge=float(controls["ppa_nudge"]),
        ppa_profile=str(controls["ppa_profile"]),
        severity=float(controls["severity"]),
        tvd_ft=float(controls["tvd_ft"]),
        measured_depth_ft=float(controls["measured_depth_ft"]),
        casing_id_in=float(controls["casing_id"]),
        wellbore_capacity_bbl_per_ft=float(controls["wellbore_capacity"]),
        mixing_efficiency=float(controls["mixing_efficiency"]),
        base_fluid_density_ppg=float(controls["fluid_density"]),
        closure_gradient_psi_ft=float(controls["closure_gradient"]),
        number_of_pumps=int(controls["number_of_pumps"]),
        pump_hhp=float(controls["pump_hhp"]),
        pump_efficiency=float(controls["pump_efficiency"]),
        max_treating_pressure_psi=float(controls["max_treating_pressure"]),
        rate_capacity_bpm=float(controls["rate_capacity"]),
        seed=int(controls["seed"]),
    )

    current_idx = int(np.argmin(np.abs(df["time_min"].to_numpy() - st.session_state.current_time)))
    live_df     = df.iloc[: current_idx + 1].copy()
    row         = df.iloc[current_idx]
    plan_event_log = df.attrs.get("planned_actual_event_log")
    if not isinstance(plan_event_log, pd.DataFrame):
        plan_event_log = pd.DataFrame(columns=["time_min", "event_type", "severity", "message", "evidence"])
    live_plan_events = plan_event_log[
        pd.to_numeric(plan_event_log.get("time_min", pd.Series(dtype=float)), errors="coerce")
        <= float(st.session_state.current_time)
    ].copy()

    def row_float(column: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(column, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    # ── Console header (logo, well metadata, alarm)
    render_console_header(row, controls)

    # ── Navigation controls (time slider + Reset / Advance)
    render_navigation(
        duration_min=float(controls["duration_min"]),
        dt_min=float(controls["dt_min"]),
        playing=bool(controls["playing"]),
    )

    # ── KPI strip + diagnostic cards
    render_status(row)

    # ── Main tabs
    tab_dashboard, tab_plan, tab_visual, tab_formation, tab_training, tab_excel, tab_calibration, tab_data = st.tabs(
        ["Dashboard", "Planned vs Actual", "Field Visual", "Formation Details",
         "Training Mode", "Excel Explorer", "Calibration", "Data"]
    )

    with tab_dashboard:
        st.plotly_chart(
            make_timeseries_plot(
                live_df,
                ("surface_pressure_psi", "bhp_psi", "net_pressure_psi"),
                "Primary Pressure Monitor",
                current_minute=st.session_state.current_time,
                show_event_labels=True,
                height=480,
            ),
            width="stretch",
        )
        st.plotly_chart(
            make_pressure_decomposition_plot(live_df, current_minute=st.session_state.current_time),
            width="stretch",
        )
        support_cols = st.columns(2)
        with support_cols[0]:
            st.plotly_chart(
                make_timeseries_plot(live_df, ("slurry_rate_bpm", "clean_rate_bpm"), "Rate",
                                     current_minute=st.session_state.current_time, height=390),
                width="stretch",
            )
        with support_cols[1]:
            st.plotly_chart(
                make_timeseries_plot(live_df, ("pipe_friction_psi", "perf_friction_psi", "net_pressure_psi"),
                                     "Friction and Net Pressure",
                                     current_minute=st.session_state.current_time, height=390),
                width="stretch",
            )
        st.plotly_chart(
            make_sand_transport_plot(live_df, current_minute=st.session_state.current_time),
            width="stretch",
        )

    with tab_plan:
        status = str(row.get("overall_execution_status", "ON PLAN"))
        metric_cols = st.columns(5)
        metric_cols[0].metric("Plan Score", f"{row_float('plan_compliance_score', 100.0):.0f}/100", status)
        metric_cols[1].metric("Rate Error", f"{row_float('rate_error_pct'):+.1f}%")
        metric_cols[2].metric("PPA Error", f"{row_float('ppa_error'):+.2f}")
        metric_cols[3].metric("Pressure Error", f"{row_float('surface_pressure_error_psi'):+,.0f} psi")
        metric_cols[4].metric("Cum Sand Error", f"{row_float('cum_sand_error_pct'):+.1f}%")

        interpretation = build_planned_actual_interpretation(
            df=df,
            event_log=plan_event_log,
            current_time_min=float(st.session_state.current_time),
        )
        interpretation_status = str(interpretation.get("status", "INFO")).upper()
        interpretation_text = (
            f"### {interpretation.get('title', 'Planned-vs-actual interpretation')}\n\n"
            f"{interpretation.get('summary', '')}"
        )
        if interpretation_status == "ON PLAN":
            st.success(interpretation_text)
        elif interpretation_status == "WATCH":
            st.warning(interpretation_text)
        elif interpretation_status in {"WARNING", "CRITICAL"}:
            st.error(interpretation_text)
        else:
            st.info(interpretation_text)

        st.markdown("**Evidence:**")
        for item in interpretation.get("evidence", []):
            st.write(f"- {item}")
        st.markdown(f"**Recommended action:** {interpretation.get('recommended_action', 'Continue monitoring')}")

        plot_cols = st.columns(2)
        with plot_cols[0]:
            st.plotly_chart(
                make_planned_actual_rate_plot(live_df, current_minute=st.session_state.current_time),
                width="stretch",
            )
            st.plotly_chart(
                make_planned_actual_sand_plot(live_df, current_minute=st.session_state.current_time),
                width="stretch",
            )
        with plot_cols[1]:
            st.plotly_chart(
                make_planned_actual_ppa_plot(live_df, current_minute=st.session_state.current_time),
                width="stretch",
            )
            st.plotly_chart(
                make_planned_actual_pressure_plot(live_df, current_minute=st.session_state.current_time),
                width="stretch",
            )

        st.subheader("Event Log")
        event_cols = ["time_min", "event_type", "severity", "message", "evidence"]
        if live_plan_events.empty:
            st.caption("No planned-vs-actual events have triggered up to the current replay time.")
        else:
            severity_colors = {
                "INFO": "background-color: #0f172a; color: #94a3b8;",
                "WATCH": "background-color: #422006; color: #fbbf24;",
                "WARNING": "background-color: #431407; color: #fb923c;",
                "CRITICAL": "background-color: #450a0a; color: #f87171;",
            }

            def color_severity(value: object) -> str:
                return severity_colors.get(str(value).upper(), "")

            event_table = live_plan_events[event_cols].tail(40).copy()
            st.dataframe(
                event_table.style.map(color_severity, subset=["severity"]),
                width="stretch",
                hide_index=True,
            )

    with tab_visual:
        render_field_visual_kpis(row)
        st.plotly_chart(make_field_visual(row), width="stretch")
        st.caption(
            "Fracture growth is a visual proxy based on cumulative slurry volume; "
            "it is not a calibrated geometry model."
        )

    with tab_formation:
        st.plotly_chart(
            make_formation_response_plot(live_df, current_minute=st.session_state.current_time),
            width="stretch",
        )
        details_cols = st.columns(4)
        details_cols[0].metric("Leakoff Rate",      f"{row.get('leakoff_rate_bpm', 0.0):.2f} bpm")
        details_cols[1].metric("Cumulative Leakoff",f"{row.get('cumulative_leakoff_bbl', 0.0):,.0f} bbl")
        details_cols[2].metric("Fluid Efficiency",  f"{100.0 * row.get('fluid_efficiency', 0.0):.0f}%")
        details_cols[3].metric("Frac Height",       f"{row.get('fracture_height_ft', 0.0):,.0f} ft")
        detail_cols = [
            "time_min","formation_state","bottomhole_ppa","net_pressure_psi",
            "pressure_slope_psi_min","fracture_width_in","fracture_half_length_ft",
            "fracture_height_ft","leakoff_rate_bpm","cumulative_leakoff_bbl",
            "fluid_efficiency","acceptance_index","formation_acceptance_pct",
            "screenout_risk","screenout_risk_pct",
        ]
        st.dataframe(live_df[[col for col in detail_cols if col in live_df.columns]].tail(120),
                     width="stretch", hide_index=True)

    with tab_training:
        render_training_tab(str(controls["scenario"]), df, float(st.session_state.current_time))

    with tab_excel:
        render_excel_explorer_tab()

    with tab_calibration:
        render_calibration_tab(df)

    with tab_data:
        display_cols = [
            "time_min","phase","slurry_rate_bpm","rate_bph","clean_rate_bpm",
            "measured_depth_ft","tvd_ft","casing_id_in",
            "surface_pressure_psi","treating_pressure_limit_psi","pressure_margin_psi",
            "surface_ppa","bottomhole_ppa","sand_lag_min","ppa_lag_delta",
            "sand_arrived_at_perfs","flush_started","flush_arrived_at_perfs",
            "sand_rate_lbm_min","sand_rate_bh_lbm_min","sand_in_wellbore_lb",
            "slurry_density_ppg","hydrostatic_psi","reynolds_proxy","friction_factor_proxy",
            "pipe_friction_psi","perf_friction_psi","bhp_psi","bhp_gradient_psi_ft",
            "net_pressure_psi","hhp_required","available_hhp","hhp_utilization",
            "rate_capacity_bpm","rate_margin_bpm","equipment_status",
            "formation_state","fracture_width_in","fracture_half_length_ft",
            "fracture_width_proxy_in","fracture_length_ft",
            "leakoff_bbl","cumulative_leakoff_bbl","fluid_efficiency",
            "acceptance_index","formation_acceptance_pct",
            "screenout_risk","screenout_risk_pct","alarm","engineer_diagnosis","event",
        ]
        st.dataframe(live_df[display_cols].tail(100), width="stretch", hide_index=True)
        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        file_scenario = str(controls["scenario"]).lower().replace(" ", "_")
        st.download_button(
            "Download full simulated job CSV", data=csv,
            file_name=f"simulated_frac_job_{file_scenario}.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "Engineering note: this simplified simulator trains real-time monitoring logic; "
        "it does not replace a calibrated hydraulic fracture model."
    )

    if controls["playing"] and st.session_state.current_time < float(controls["duration_min"]):
        time.sleep(0.45)
        st.rerun()


if __name__ == "__main__":
    main()
