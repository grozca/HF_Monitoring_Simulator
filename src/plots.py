from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ────────────────────────────────────────────────────────────────────
# Every color tuned for a near-black (#060809) background.
COLORS = {
    "surface":   "#22d3ee",   # cyan   – Surface Pressure (dominant channel)
    "bhp":       "#f87171",   # red    – BHP
    "net":       "#4ade80",   # green  – Net Pressure
    "rate":      "#38bdf8",   # sky    – Slurry / Clean Rate
    "ppa_surf":  "#4ade80",   # green  – Surface PPA
    "ppa_bh":    "#f59e0b",   # amber  – Bottomhole PPA
    "friction":  "#a78bfa",   # violet – Pipe / Perf Friction
    "hydro":     "#38bdf8",   # sky    – Hydrostatic
    "limit":     "#f87171",   # red    – Treating pressure limit
    "sand":      "#94a3b8",   # slate  – Sand in wellbore
    "width":     "#4ade80",   # green  – Fracture width
    "length":    "#38bdf8",   # sky    – Half-length
    "accept":    "#f59e0b",   # amber  – Acceptance
    "risk":      "#f87171",   # red    – Screenout risk
    "eff":       "#a78bfa",   # violet – Fluid efficiency
    "event":     "#f87171",   # red    – Scenario events
    "phase_div": "#1e293b",   # muted  – Phase transition lines
    "time_cur":  "#38bdf8",   # sky    – Current-time cursor
    "grid":      "#1a2332",   # dark blue-gray grid
    "bg_plot":   "#060809",   # near-black plot bg
    "bg_paper":  "#0a0c0f",   # page bg
}

# Bright overrides for event-annotation lines so they pop on dark background
EVENT_COLORS = {
    "Sand starts":   "#4ade80",
    "Sand at perfs": "#f59e0b",
    "Flush starts":  "#38bdf8",
    "Flush at perfs":"#a78bfa",
}

SERIES_LABELS = {
    "surface_pressure_psi":  "Surface Pressure",
    "bhp_psi":               "BHP",
    "net_pressure_psi":      "Net Pressure",
    "slurry_rate_bpm":       "Slurry Rate",
    "clean_rate_bpm":        "Clean Rate",
    "pipe_friction_psi":     "Pipe Friction",
    "perf_friction_psi":     "Perf Friction",
    "surface_ppa":           "Surface PPA",
    "bottomhole_ppa":        "BH PPA",
    "hydrostatic_psi":       "Hydrostatic",
    "fracture_width_in":     "Frac Width",
    "fracture_half_length_ft":"Half-Length",
    "acceptance_index":      "Acceptance",
    "screenout_risk":        "Screenout Risk",
    "fluid_efficiency":      "Fluid Efficiency",
}

# Map column → palette key for auto-coloring
_COL_COLOR: dict[str, str] = {
    "surface_pressure_psi":   COLORS["surface"],
    "bhp_psi":                COLORS["bhp"],
    "net_pressure_psi":       COLORS["net"],
    "slurry_rate_bpm":        COLORS["rate"],
    "clean_rate_bpm":         "#93c5fd",
    "pipe_friction_psi":      COLORS["friction"],
    "perf_friction_psi":      "#c4b5fd",
    "surface_ppa":            COLORS["ppa_surf"],
    "bottomhole_ppa":         COLORS["ppa_bh"],
    "hydrostatic_psi":        COLORS["hydro"],
    "fracture_width_in":      COLORS["width"],
    "fracture_half_length_ft":COLORS["length"],
    "acceptance_index":       COLORS["accept"],
    "screenout_risk":         COLORS["risk"],
    "fluid_efficiency":       COLORS["eff"],
    "treating_pressure_limit_psi": COLORS["limit"],
    "sand_in_wellbore_lb":    COLORS["sand"],
}


def _series_label(col: str) -> str:
    return SERIES_LABELS.get(
        col,
        col.replace("_", " ").replace("psi", "(psi)").replace("bpm", "(bpm)").title(),
    )


def _col_color(col: str, fallback: str = "#94a3b8") -> str:
    return _COL_COLOR.get(col, fallback)


# ── Base layout ────────────────────────────────────────────────────────────────

def _style(fig: go.Figure, title: str, *, height: int = 360) -> go.Figure:
    """Apply full dark SCADA theme to any figure."""
    fig.update_layout(
        title=dict(
            text=f"<span style='font-family:Courier New;font-size:13px;"
                 f"letter-spacing:2px;color:#94a3b8;'>{title.upper()}</span>",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        height=height,
        paper_bgcolor=COLORS["bg_paper"],
        plot_bgcolor=COLORS["bg_plot"],
        font=dict(
            family="Courier New, monospace",
            color="#94a3b8",
            size=11,
        ),
        margin=dict(l=60, r=28, t=44, b=72),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            font=dict(size=10, family="Courier New, monospace", color="#64748b"),
            bgcolor="rgba(0,0,0,0)",
            itemwidth=30,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            font_color="#e2e8f0",
            font_family="Courier New, monospace",
            font_size=11,
            bordercolor="#1e293b",
        ),
    )
    fig.update_xaxes(
        title_text="Time, min",
        title_font=dict(size=10, color="#475569"),
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=0.5,
        zeroline=False,
        tickfont=dict(color="#475569", size=10),
        linecolor="#1e293b",
        linewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=0.5,
        zeroline=False,
        tickfont=dict(color="#475569", size=10),
        title_font=dict(size=10, color="#475569"),
        linecolor="#1e293b",
        linewidth=1,
    )
    return fig


def _add_time_marker(fig: go.Figure, current_minute: int | float) -> None:
    fig.add_vline(
        x=current_minute,
        line_width=1.5,
        line_dash="dash",
        line_color=COLORS["time_cur"],
        opacity=0.6,
    )


def _add_stage_lines(fig: go.Figure, df: pd.DataFrame) -> None:
    if "phase" not in df.columns or "time_min" not in df.columns:
        return
    phase_changes = df.loc[df["phase"].ne(df["phase"].shift()), "time_min"].tolist()
    for x_value in phase_changes[1:]:
        fig.add_vline(
            x=x_value,
            line_width=0.8,
            line_dash="dot",
            line_color=COLORS["phase_div"],
            opacity=1.0,
        )


def _first_threshold_time(df: pd.DataFrame, col: str, threshold: float = 0.05) -> float | None:
    if col not in df.columns or "time_min" not in df.columns:
        return None
    mask = df[col] > threshold
    if not mask.any():
        return None
    return float(df.loc[mask, "time_min"].iloc[0])


def _first_true_time(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or "time_min" not in df.columns:
        return None
    mask = df[col].astype(bool)
    if not mask.any():
        return None
    return float(df.loc[mask, "time_min"].iloc[0])


def _add_event_markers(fig: go.Figure, df: pd.DataFrame, *, show_labels: bool = False) -> None:
    if "time_min" not in df.columns or df.empty:
        return

    events: list[tuple[float, str, str, str]] = []

    surface_time = _first_threshold_time(df, "surface_ppa")
    if surface_time is not None:
        events.append((surface_time, "Sand starts",   EVENT_COLORS["Sand starts"],   "dot"))

    bh_time = _first_threshold_time(df, "bottomhole_ppa")
    if bh_time is not None:
        events.append((bh_time,     "Sand at perfs",  EVENT_COLORS["Sand at perfs"], "dash"))

    flush_start = _first_true_time(df, "flush_started")
    if flush_start is not None:
        events.append((flush_start, "Flush starts",   EVENT_COLORS["Flush starts"],  "dot"))

    flush_arrival = _first_true_time(df, "flush_arrived_at_perfs")
    if flush_arrival is not None:
        events.append((flush_arrival,"Flush at perfs",EVENT_COLORS["Flush at perfs"],"dash"))

    if "event" in df.columns:
        mask = df["event"].astype(str).ne("Normal")
        if mask.any():
            event_time  = float(df.loc[mask, "time_min"].iloc[0])
            event_label = str(df.loc[mask, "event"].iloc[0])
            events.append((event_time, event_label, COLORS["event"], "solid"))

    seen: set[tuple[float, str]] = set()
    for index, (x_value, label, color, dash) in enumerate(events):
        marker_key = (round(x_value, 2), label)
        if marker_key in seen:
            continue
        seen.add(marker_key)
        kwargs: dict = {
            "x": x_value,
            "line_width": 1.2,
            "line_dash": dash,
            "line_color": color,
            "opacity": 0.85,
        }
        if show_labels:
            kwargs.update(
                annotation_text=f"<span style='font-family:Courier New;font-size:9px;"
                                f"letter-spacing:1px;'>{label.upper()}</span>",
                annotation_position="top",
                annotation_font_size=9,
                annotation_font_color=color,
                annotation_yshift=10 + 14 * (index % 2),
            )
        fig.add_vline(**kwargs)


# ── Public chart functions ─────────────────────────────────────────────────────

def make_timeseries_plot(
    df: pd.DataFrame,
    y_cols: tuple[str, ...],
    title: str,
    current_minute: int | float | None = None,
    *,
    show_event_labels: bool = False,
    height: int = 360,
) -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["time_min"],
                y=df[col],
                mode="lines",
                name=_series_label(col),
                line=dict(width=2.0, color=_col_color(col)),
            )
        )

    _add_stage_lines(fig, df)
    _add_event_markers(fig, df, show_labels=show_event_labels)
    if current_minute is not None:
        _add_time_marker(fig, current_minute)
    _style(fig, title, height=height)
    return fig


def pressure_plot(df: pd.DataFrame, current_minute: int | float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["surface_pressure_psi"],
                   name="Surface", mode="lines",
                   line=dict(color=COLORS["surface"], width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["bhp_psi"],
                   name="BHP", mode="lines",
                   line=dict(color=COLORS["bhp"], width=2.2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["net_pressure_psi"],
                   name="Net", mode="lines",
                   line=dict(color=COLORS["net"], width=2.0)),
        secondary_y=True,
    )
    if "treating_pressure_limit_psi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df["treating_pressure_limit_psi"],
                       name="Limit", mode="lines",
                       line=dict(color=COLORS["limit"], width=1.4, dash="dot")),
            secondary_y=False,
        )
    _add_time_marker(fig, current_minute)
    _add_event_markers(fig, df, show_labels=True)
    _style(fig, "Pressure Monitor", height=420)
    fig.update_yaxes(title_text="Surface / BHP  [psi]", secondary_y=False,
                     title_font=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="Net  [psi]", secondary_y=True,
                     title_font=dict(size=10, color="#475569"))
    return fig


def rate_sand_plot(df: pd.DataFrame, current_minute: int | float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["clean_rate_bpm"],
                   name="Clean Rate", mode="lines",
                   line=dict(color=COLORS["rate"], width=2.2)),
        secondary_y=False,
    )
    if "slurry_rate_bpm" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df["slurry_rate_bpm"],
                       name="Slurry Rate", mode="lines",
                       line=dict(color="#93c5fd", width=1.8, dash="dot")),
            secondary_y=False,
        )
    if "surface_ppa" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df["surface_ppa"],
                       name="Surface PPA", mode="lines",
                       line=dict(color=COLORS["ppa_surf"], width=2.0)),
            secondary_y=True,
        )
    _add_time_marker(fig, current_minute)
    _add_event_markers(fig, df)
    _style(fig, "Rate & PPA", height=360)
    fig.update_yaxes(title_text="Rate  [bpm]",   secondary_y=False,
                     title_font=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="PPA  [lb/gal]", secondary_y=True,
                     title_font=dict(size=10, color="#475569"))
    return fig


def make_sand_transport_plot(
    df: pd.DataFrame,
    current_minute: int | float | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["surface_ppa"],
                   name="Surface PPA", mode="lines",
                   line=dict(color=COLORS["ppa_surf"], width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["time_min"], y=df["bottomhole_ppa"],
                   name="BH PPA", mode="lines",
                   line=dict(color=COLORS["ppa_bh"], width=2.5)),
        secondary_y=False,
    )
    if "sand_in_wellbore_lb" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df["sand_in_wellbore_lb"],
                       name="Sand in Wellbore", mode="lines",
                       fill="tozeroy",
                       fillcolor="rgba(148,163,184,0.10)",
                       line=dict(color=COLORS["sand"], width=1.4)),
            secondary_y=True,
        )
    _add_stage_lines(fig, df)
    _add_event_markers(fig, df)
    if current_minute is not None:
        _add_time_marker(fig, current_minute)
    _style(fig, "Sand Transport", height=370)
    fig.update_yaxes(title_text="PPA  [lb/gal]",      secondary_y=False,
                     title_font=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="Sand in WB  [lb]",   secondary_y=True,
                     title_font=dict(size=10, color="#475569"))
    return fig


def make_pressure_decomposition_plot(
    df: pd.DataFrame,
    current_minute: int | float | None = None,
) -> go.Figure:
    fig = go.Figure()
    components = (
        ("surface_pressure_psi", COLORS["surface"],  "solid", 2.7, "Surface Pressure"),
        ("bhp_psi",              COLORS["bhp"],      "dash",  2.2, "BHP"),
        ("hydrostatic_psi",      COLORS["hydro"],    "solid", 1.8, "Hydrostatic"),
        ("pipe_friction_psi",    COLORS["friction"], "solid", 1.8, "Pipe Friction"),
        ("perf_friction_psi",    "#c4b5fd",          "solid", 1.8, "Perf Friction"),
        ("net_pressure_psi",     COLORS["net"],      "solid", 2.0, "Net Pressure"),
    )
    for col, color, dash, width, label in components:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df[col],
                       name=label, mode="lines",
                       line=dict(color=color, width=width, dash=dash))
        )
    _add_stage_lines(fig, df)
    _add_event_markers(fig, df)
    if current_minute is not None:
        _add_time_marker(fig, current_minute)
    _style(fig, "Pressure Decomposition", height=420)
    fig.update_yaxes(title_text="Pressure  [psi]",
                     title_font=dict(size=10, color="#475569"))
    return fig


def make_formation_response_plot(
    df: pd.DataFrame,
    current_minute: int | float | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    left_axis = (
        ("fracture_width_in",      COLORS["width"],  "Frac Width, in"),
        ("fracture_half_length_ft",COLORS["length"], "Half-Length, ft"),
    )
    right_axis = (
        ("acceptance_index", COLORS["accept"], "Acceptance",      "dash"),
        ("screenout_risk",   COLORS["risk"],   "Screenout Risk",  "solid"),
        ("fluid_efficiency", COLORS["eff"],    "Fluid Efficiency","dot"),
    )
    for col, color, label in left_axis:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df[col],
                       name=label, mode="lines",
                       line=dict(color=color, width=2.2)),
            secondary_y=False,
        )
    for col, color, label, dash in right_axis:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df["time_min"], y=df[col],
                       name=label, mode="lines",
                       line=dict(color=color, width=2.0, dash=dash)),
            secondary_y=True,
        )
    _add_stage_lines(fig, df)
    _add_event_markers(fig, df)
    if current_minute is not None:
        _add_time_marker(fig, current_minute)
    _style(fig, "Formation Response", height=430)
    fig.update_yaxes(title_text="Geometry proxy",
                     secondary_y=False,
                     title_font=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="Index  [fraction]",
                     range=[0.0, 1.05],
                     secondary_y=True,
                     title_font=dict(size=10, color="#475569"))
    return fig


def make_field_visual(row: pd.Series) -> go.Figure:
    """Subsurface frac schematic on a dark field-console canvas."""
    fig = go.Figure()

    phase_text = str(row.get("phase", ""))
    formation_state = str(row.get("formation_state", "Unknown"))
    event_text = str(row.get("event", "Normal"))
    surface_ppa = float(row.get("surface_ppa", row.get("ppa", 0.0)))
    bottomhole_ppa = float(row.get("bottomhole_ppa", surface_ppa))
    screenout_risk = float(row.get("screenout_risk", 0.0))
    cumulative_slurry = float(row.get("cum_slurry_bbl", row.get("cumulative_slurry_bbl", 0.0)))
    half_length_ft = float(row.get("fracture_half_length_ft", row.get("fracture_length_ft", 0.0)))
    frac_height_ft = float(row.get("fracture_height_ft", 0.0))
    sand_in_wellbore = float(row.get("sand_in_wellbore_lb", 0.0))
    sand_arrived = bool(row.get("sand_arrived_at_perfs", bottomhole_ppa > 0.05))
    pumps = int(row.get("number_of_pumps", 12)) if "number_of_pumps" in row else 12

    progress_from_volume = cumulative_slurry / 6500.0
    progress_from_length = half_length_ft / 2400.0 if half_length_ft > 0 else 0.0
    frac_progress = float(np.clip(max(progress_from_volume, progress_from_length, 0.04), 0.04, 1.0))

    surf_y = 66.0
    pay_y = 218.0
    pay_h = 44.0
    perf_y = pay_y + pay_h / 2.0
    wb_x = 430.0
    wb_w = 10.0
    wb_top = 30.0
    wb_bot = perf_y
    max_wing = 320.0
    wing_px = max_wing * frac_progress
    wing_h = float(np.clip(10.0 + frac_progress * 28.0 + frac_height_ft / 24.0, 14.0, 58.0))

    if screenout_risk >= 0.65 or "Screenout" in event_text:
        frac_fill = "rgba(248,113,113,0.42)"
        frac_line = "#f87171"
        status_color = "#f87171"
        status_text = "SCREENOUT RISK"
    elif bottomhole_ppa > 0.05:
        frac_fill = "rgba(245,158,11,0.36)"
        frac_line = "#f59e0b"
        status_color = "#f59e0b"
        status_text = "PROPPANT PLACEMENT"
    else:
        frac_fill = "rgba(34,211,238,0.24)"
        frac_line = "#22d3ee"
        status_color = "#22d3ee"
        status_text = "PAD / FLUSH"

    # Formation layers and target zone.
    layers = [
        (surf_y, surf_y + 28, "#0f1923"),
        (surf_y + 28, surf_y + 50, "#111c27"),
        (surf_y + 50, surf_y + 86, "#0d1520"),
        (surf_y + 86, surf_y + 112, "#10181f"),
        (surf_y + 112, surf_y + 160, "#0e1622"),
        (surf_y + 160, surf_y + 195, "#0d1520"),
        (surf_y + 195, 420, "#111c27"),
    ]
    for y0, y1, color in layers:
        fig.add_shape(type="rect", x0=0, x1=900, y0=y0, y1=y1,
                      fillcolor=color, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, x1=900, y0=pay_y, y1=pay_y + pay_h,
                  fillcolor="rgba(22,101,52,0.24)",
                  line=dict(color="#166534", width=1, dash="dot"),
                  layer="below")

    # Subtle rock grain.
    for idx, y in enumerate(np.linspace(surf_y + 16, 402, 22)):
        xs = np.linspace(12, 888, 40)
        ys = y + np.sin(xs * 0.035 + idx) * 2.0
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(100,116,139,0.16)", width=0.6),
            hoverinfo="skip", showlegend=False,
        ))

    # Ground line and target labels.
    fig.add_shape(type="line", x0=0, x1=900, y0=surf_y, y1=surf_y,
                  line=dict(color="#334155", width=1, dash="dash"))
    fig.add_annotation(x=8, y=surf_y - 8, text="GROUND LEVEL",
                       showarrow=False, xanchor="left",
                       font=dict(color="#475569", size=10, family="Courier New, monospace"))
    fig.add_annotation(x=8, y=pay_y + 13, text="TARGET FORMATION | HF ZONE",
                       showarrow=False, xanchor="left",
                       font=dict(color="#4ade80", size=10, family="Courier New, monospace"))

    # Surface equipment and treating iron.
    fig.add_shape(type="line", x0=58, x1=wb_x - wb_w - 3, y0=44, y1=44,
                  line=dict(color="#22d3ee", width=4))
    equipment = [
        ("DATA VAN", "MONITOR", 14, 16, 86, 34, "#1e3a5f", "#38bdf8"),
        ("BLENDER", "", 110, 22, 64, 28, "#1e3a5f", "#2563eb"),
        ("HHP", f"x{pumps} PUMPS", 184, 14, 76, 36, "#431407", "#f59e0b"),
        ("MISSILE", "", 270, 22, 64, 28, "#0d1117", "#64748b"),
    ]
    for label, sub, x, y, w, h, fill, border in equipment:
        fig.add_shape(type="rect", x0=x, x1=x + w, y0=y, y1=y + h,
                      fillcolor=fill, line=dict(color=border, width=1))
        fig.add_annotation(x=x + w / 2, y=y + h / 2 - (4 if sub else 0),
                           text=label, showarrow=False,
                           font=dict(color=border, size=9, family="Courier New, monospace"))
        if sub:
            fig.add_annotation(x=x + w / 2, y=y + h / 2 + 10,
                               text=sub, showarrow=False,
                               font=dict(color="#64748b", size=8, family="Courier New, monospace"))

    for x0, x1 in [(100, 110), (174, 184), (260, 270)]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=34, y1=34,
                      line=dict(color="#334155", width=2))
    fig.add_shape(type="line", x0=334, x1=378, y0=34, y1=44,
                  line=dict(color="#334155", width=2))
    fig.add_shape(type="line", x0=378, x1=410, y0=44, y1=44,
                  line=dict(color="#334155", width=2))

    # Wellhead and wellbore.
    fig.add_shape(type="rect", x0=wb_x - 18, x1=wb_x + 18, y0=30, y1=58,
                  fillcolor="#0d1117", line=dict(color="#22d3ee", width=1.5))
    fig.add_annotation(x=wb_x, y=44, text="WH", showarrow=False,
                       font=dict(color="#22d3ee", size=10, family="Courier New, monospace"))
    fig.add_shape(type="line", x0=wb_x, x1=wb_x, y0=58, y1=wb_top,
                  line=dict(color="#22d3ee", width=3))
    fig.add_shape(type="rect", x0=wb_x - wb_w - 4, x1=wb_x - wb_w + 2,
                  y0=wb_top, y1=wb_bot, fillcolor="#1e293b",
                  line=dict(color="#475569", width=1))
    fig.add_shape(type="rect", x0=wb_x + wb_w - 2, x1=wb_x + wb_w + 4,
                  y0=wb_top, y1=wb_bot, fillcolor="#1e293b",
                  line=dict(color="#475569", width=1))
    fluid_color = "rgba(245,158,11,0.42)" if surface_ppa > 0.05 else "rgba(34,211,238,0.28)"
    fig.add_shape(type="rect", x0=wb_x - wb_w + 3, x1=wb_x + wb_w - 3,
                  y0=wb_top, y1=wb_bot, fillcolor=fluid_color,
                  line=dict(width=0))
    fig.add_shape(type="rect", x0=wb_x - wb_w - 6, x1=wb_x + wb_w + 6,
                  y0=wb_bot - 5, y1=wb_bot + 2, fillcolor="#475569",
                  line=dict(width=0))

    # Perforation shots.
    for offset in [-18, -10, 10, 18]:
        fig.add_shape(type="line", x0=wb_x + offset * 0.45, x1=wb_x + offset,
                      y0=perf_y, y1=perf_y + 10,
                      line=dict(color="#f59e0b", width=1.2))
    fig.add_annotation(x=wb_x + 34, y=perf_y + 5, text="PERFS",
                       showarrow=False, xanchor="left",
                       font=dict(color="#f59e0b", size=10, family="Courier New, monospace"))

    # Fracture wings as filled curved polygons.
    t = np.linspace(0.0, 1.0, 34)
    left_top_x = wb_x - wb_w - wing_px * t
    left_top_y = perf_y - wing_h * (0.45 + 0.25 * np.sin(np.pi * t))
    left_bot_x = left_top_x[::-1]
    left_bot_y = (perf_y + wing_h * (0.45 + 0.25 * np.sin(np.pi * t)))[::-1]
    right_top_x = wb_x + wb_w + wing_px * t
    right_top_y = left_top_y
    right_bot_x = right_top_x[::-1]
    right_bot_y = left_bot_y

    for name, xs, ys in [
        ("Fracture wing L", np.r_[left_top_x, left_bot_x], np.r_[left_top_y, left_bot_y]),
        ("Fracture wing R", np.r_[right_top_x, right_bot_x], np.r_[right_top_y, right_bot_y]),
    ]:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=frac_fill,
            line=dict(color=frac_line, width=1.3),
            name=name,
            hovertemplate="Fracture proxy<br>Half-length: %{customdata:,.0f} ft<extra></extra>",
            customdata=np.full(len(xs), half_length_ft if half_length_ft > 0 else frac_progress * 2200),
            showlegend=False,
        ))

    # Fracture centerline and legend item.
    fig.add_trace(go.Scatter(
        x=[wb_x - wing_px, wb_x + wing_px],
        y=[perf_y, perf_y],
        mode="lines",
        line=dict(color="#4ade80", width=2, dash="dot"),
        name="Fracture wings",
        hoverinfo="skip",
    ))

    # Proppant in wellbore and fracture.
    if sand_in_wellbore > 1.0 or (surface_ppa > 0.05 and not sand_arrived):
        ys = np.linspace(wb_top + 24, max(wb_top + 28, perf_y - 18), 10)
        fig.add_trace(go.Scatter(
            x=np.full_like(ys, wb_x), y=ys, mode="markers",
            marker=dict(size=5, color="#f59e0b", line=dict(color="#92400e", width=0.5)),
            name="Proppant in WB",
            hovertemplate="Proppant in wellbore<br>%{y:.0f} visual depth<extra></extra>",
        ))
    if bottomhole_ppa > 0.05 and wing_px > 20:
        n_prop = int(np.clip(8 + bottomhole_ppa * 14 * frac_progress, 8, 58))
        idx = np.arange(n_prop)
        side = np.where(idx % 2 == 0, -1.0, 1.0)
        spread = (0.12 + 0.78 * ((np.sin(idx * 1.7) + 1.0) / 2.0)) * wing_px
        xs = wb_x + side * spread
        ys = perf_y + np.cos(idx * 2.1) * wing_h * 0.28
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=4 + min(bottomhole_ppa, 3.0) * 1.5,
                color="#fbbf24",
                opacity=0.72,
                line=dict(color="#92400e", width=0.4),
            ),
            name="Proppant / slurry",
            hovertemplate="BH PPA: %{customdata:.2f} lb/gal<extra></extra>",
            customdata=np.full(n_prop, bottomhole_ppa),
        ))

    # Screenout bridge visual.
    if screenout_risk >= 0.65 or "Screenout" in event_text:
        theta = np.linspace(0, 2 * np.pi, 50)
        radius = 16.0
        fig.add_trace(go.Scatter(
            x=wb_x + np.cos(theta) * radius,
            y=perf_y + np.sin(theta) * radius,
            mode="lines", fill="toself",
            fillcolor="rgba(248,113,113,0.16)",
            line=dict(color="#f87171", width=1.4),
            name="Bridge / restriction",
            hovertemplate="Screenout risk: %{customdata:.0%}<extra></extra>",
            customdata=np.full(len(theta), screenout_risk),
        ))
        fig.add_annotation(x=wb_x, y=perf_y - 24, text="BRIDGE<br>PLUG",
                           showarrow=False,
                           font=dict(color="#f87171", size=10, family="Courier New, monospace"))

    # Depth callouts and status panel.
    depth_labels = [
        (surf_y, "0 ft", "#475569"),
        (surf_y + 58, "2,500 ft", "#64748b"),
        (pay_y, f"{float(row.get('tvd_ft', 8500.0)):,.0f} ft TVD", "#4ade80"),
        (pay_y + pay_h, f"{float(row.get('tvd_ft', 8500.0)) + 200:,.0f} ft", "#64748b"),
        (404, f"{float(row.get('measured_depth_ft', 10000.0)):,.0f} ft MD", "#475569"),
    ]
    fig.add_shape(type="line", x0=744, x1=744, y0=surf_y, y1=404,
                  line=dict(color="#334155", width=0.8))
    for y, label, color in depth_labels:
        fig.add_shape(type="line", x0=724, x1=756, y0=y, y1=y,
                      line=dict(color=color, width=0.8, dash="dot"))
        fig.add_annotation(x=764, y=y, text=label, showarrow=False,
                           xanchor="left",
                           font=dict(color=color, size=9, family="Courier New, monospace"))

    status_x = 764
    status_y = pay_y + pay_h + 38
    fig.add_annotation(x=status_x, y=status_y, text=status_text,
                       showarrow=False, xanchor="left",
                       font=dict(color=status_color, size=11, family="Courier New, monospace"))
    fig.add_annotation(x=status_x, y=status_y + 16, text="FRACTURE STATE",
                       showarrow=False, xanchor="left",
                       font=dict(color="#475569", size=9, family="Courier New, monospace"))
    fig.add_annotation(x=status_x, y=status_y + 31,
                       text=f"{formation_state} | Growth {frac_progress * 100:.0f}%",
                       showarrow=False, xanchor="left",
                       font=dict(color="#94a3b8", size=9, family="Courier New, monospace"))

    # Legend helper traces.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="#22d3ee", line=dict(color="#22d3ee", width=1)),
                             name="Surface iron / WH"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="#1e3a5f", line=dict(color="#2563eb", width=1)),
                             name="Pumping equipment"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=8, color="#64748b", line=dict(color="#475569", width=1)),
                             name="Formation rock"))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-family:Courier New;font-size:13px;"
                 f"letter-spacing:2px;color:#94a3b8;'>FIELD VISUAL  |  PHASE: {phase_text.upper()}</span>",
            x=0.0, xanchor="left",
        ),
        paper_bgcolor=COLORS["bg_paper"],
        plot_bgcolor=COLORS["bg_plot"],
        xaxis=dict(visible=False, range=[0, 900], fixedrange=True),
        yaxis=dict(visible=False, range=[420, 0], fixedrange=True),
        height=540,
        margin=dict(l=8, r=8, t=46, b=14),
        showlegend=True,
        legend=dict(
            orientation="h",
            font=dict(size=10, family="Courier New, monospace", color="#64748b"),
            bgcolor="rgba(0,0,0,0)",
            yanchor="top",
            y=-0.03,
            xanchor="left",
            x=0.0,
        ),
        font=dict(family="Courier New, monospace", color="#94a3b8"),
        hoverlabel=dict(
            bgcolor="#111827",
            font_color="#e2e8f0",
            font_family="Courier New, monospace",
            font_size=11,
            bordercolor="#1e293b",
        ),
    )
    return fig
