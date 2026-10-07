from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ParkSmart Night-Ops palette (dark bg): green = free/healthy, red = taken/critical.
GREEN = "#34d399"
GREEN_SOFT = "rgba(52,211,153,0.12)"
CHARCOAL = "#f87171"
CHARCOAL_SOFT = "rgba(248,113,113,0.10)"
AMBER = "#fbbf24"
RED = "#f87171"
BLUE = "#60a5fa"
SLATE = "#8ea0b8"
GRID = "#22354f"

_FIT_CACHE: dict = {}


def _base(height: int, **extra):
    layout = dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=11, color="#c6d2e2"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#13233c", font_size=11, font_color="#e8eef6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#8ea0b8"), linecolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, tickfont=dict(size=10, color="#8ea0b8"), linecolor=GRID, rangemode="tozero"),
    )
    layout.update(extra)
    return layout


def _show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _tick_format(hours: int) -> str:
    if hours <= 1:
        return "%H:%M"
    if hours <= 48:
        return "%H:%M\n%d %b"
    return "%d %b"


def occupancy_chart(trends: pd.DataFrame):
    """How busy is the parking facility over time?"""
    hours = 24
    if len(trends) > 1:
        span = (trends["timestamp"].max() - trends["timestamp"].min()).total_seconds() / 3600
        hours = max(1, int(span))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trends["timestamp"], y=trends["occupied"], mode="lines", name="Occupied",
        line=dict(width=2.2, color=CHARCOAL, shape="spline", smoothing=0.5),
        fill="tozeroy", fillcolor=CHARCOAL_SOFT,
        hovertemplate="<b>%{y:.0f}</b> occupied<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trends["timestamp"], y=trends["available"], mode="lines", name="Available",
        line=dict(width=2.2, color=GREEN, shape="spline", smoothing=0.5),
        fill="tozeroy", fillcolor=GREEN_SOFT,
        hovertemplate="<b>%{y:.0f}</b> available<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.update_layout(**_base(340, yaxis_title="Spaces"))
    fig.update_xaxes(tickformat=_tick_format(hours), nticks=8)
    _show(fig)


def entry_exit_chart(trends: pd.DataFrame):
    """Are vehicles entering faster than leaving?"""
    hours = 24
    if len(trends) > 1:
        span = (trends["timestamp"].max() - trends["timestamp"].min()).total_seconds() / 3600
        hours = max(1, int(span))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trends["timestamp"], y=trends["entries"], name="Entries",
        marker=dict(color=CHARCOAL, line=dict(width=1, color="white")),
        hovertemplate="<b>%{y:.0f}</b> entries<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=trends["timestamp"], y=trends["exits"], name="Exits",
        marker=dict(color=GREEN, line=dict(width=1, color="white")),
        hovertemplate="<b>%{y:.0f}</b> exits<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.update_layout(**_base(300, yaxis_title="Vehicles", barmode="group", bargap=0.32, bargroupgap=0.15))
    fig.update_xaxes(tickformat=_tick_format(hours), nticks=8)
    _show(fig)


def main_occupancy_forecast_chart(
    trends: pd.DataFrame,
    current_available: int,
    predicted_available: float,
):
    """Largest Overview chart: solid history anchored at NOW, dashed 30-min forecast."""
    last_time = trends["timestamp"].max()
    future_time = last_time + pd.Timedelta(minutes=30)
    upper = predicted_available * 1.08
    lower = max(0, predicted_available * 0.92)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trends["timestamp"], y=trends["available"], mode="lines", name="Available (history)",
        line=dict(width=2.4, color=GREEN, shape="spline", smoothing=0.5),
        fill="tozeroy", fillcolor=GREEN_SOFT,
        hovertemplate="<b>%{y:.0f}</b> free<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trends["timestamp"], y=trends["occupied"], mode="lines", name="Occupied (history)",
        line=dict(width=1.6, color=CHARCOAL, shape="spline", smoothing=0.5),
        hovertemplate="<b>%{y:.0f}</b> occupied<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[last_time, future_time, future_time, last_time],
        y=[float(current_available), lower, upper, float(current_available)],
        fill="toself", fillcolor="rgba(14,122,95,0.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[last_time, future_time], y=[current_available, predicted_available],
        mode="lines+markers", name="Forecast +30 min",
        line=dict(width=2.6, dash="dash", color=GREEN),
        marker=dict(size=8, color=GREEN, line=dict(width=2, color="white")),
        hovertemplate="<b>%{y:.0f}</b> spaces<br>%{x|%H:%M}<extra>forecast</extra>",
    ))
    # NOW marker (datetime axis needs a datetime, not millis)
    fig.add_vline(x=last_time, line_dash="dot", line_color="#9aa5a1", line_width=1)
    fig.update_layout(**_base(380, yaxis_title="Spaces"))
    hours = 24
    if len(trends) > 1:
        span = (trends["timestamp"].max() - trends["timestamp"].min()).total_seconds() / 3600
        hours = max(1, int(span))
    fig.update_xaxes(tickformat=_tick_format(hours), nticks=9)
    fig.add_annotation(x=future_time, y=predicted_available, text=f"<b>{predicted_available:.0f}</b> expected",
                       showarrow=True, arrowhead=2, ax=0, ay=-30,
                       bgcolor="#13233c", bordercolor=GRID, font=dict(size=11, color="#e8eef6"))
    _show(fig)


def forecast_chart(historical: pd.DataFrame, current_available: int, predicted_available: float):
    """Prediction page chart: NOW → +30 min path."""
    last_time = historical["timestamp"].max()
    future_time = last_time + pd.Timedelta(minutes=30)
    upper = predicted_available * 1.08
    lower = max(0, predicted_available * 0.92)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical["timestamp"], y=historical["available"], mode="lines", name="History",
        line=dict(width=2.2, color=SLATE, shape="spline", smoothing=0.5),
        hovertemplate="<b>%{y:.0f}</b> free<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[last_time, future_time, future_time, last_time],
        y=[float(current_available), lower, upper, float(current_available)],
        fill="toself", fillcolor="rgba(14,122,95,0.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[last_time, future_time], y=[current_available, predicted_available],
        mode="lines+markers", name="Forecast +30 min",
        line=dict(width=2.6, dash="dash", color=GREEN),
        marker=dict(size=8, color=GREEN, line=dict(width=2, color="white")),
        hovertemplate="<b>%{y:.0f}</b> spaces<br>%{x|%H:%M}<extra>forecast</extra>",
    ))
    fig.update_layout(**_base(340, yaxis_title="Free spaces"))
    fig.add_annotation(x=future_time, y=predicted_available, text=f"<b>{predicted_available:.0f}</b>",
                       showarrow=True, arrowhead=2, ax=0, ay=-30,
                       bgcolor="#13233c", bordercolor=GRID, font=dict(size=11, color="#e8eef6"))
    _show(fig)


def live_tick_occupancy_chart(stream: pd.DataFrame):
    """Live Parking Occupancy: occupied vs available per simulation tick."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stream["tick"], y=stream["occupied"], mode="lines+markers", name="Occupied spaces",
        line=dict(width=2.2, color=CHARCOAL, shape="spline", smoothing=0.4),
        marker=dict(size=5),
        hovertemplate="Tick %{x}: <b>%{y:.0f}</b> occupied<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stream["tick"], y=stream["available"], mode="lines+markers", name="Available spaces",
        line=dict(width=2.2, color=GREEN, shape="spline", smoothing=0.4),
        marker=dict(size=5),
        hovertemplate="Tick %{x}: <b>%{y:.0f}</b> available<extra></extra>",
    ))
    fig.update_layout(**_base(300, xaxis_title="Live simulation tick", yaxis_title="Number of spaces"))
    _show(fig)


def live_tick_prediction_chart(stream: pd.DataFrame):
    """Live AI Availability Prediction: actual vs predicted per tick."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stream["tick"], y=stream["available"], mode="lines+markers", name="Actual available",
        line=dict(width=2.2, color=BLUE, shape="spline", smoothing=0.4),
        marker=dict(size=5),
        hovertemplate="Tick %{x}: <b>%{y:.0f}</b> actual<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stream["tick"], y=stream["predicted"], mode="lines+markers", name="AI predicted",
        line=dict(width=2.2, color=AMBER, shape="spline", smoothing=0.4),
        marker=dict(size=5),
        hovertemplate="Tick %{x}: <b>%{y:.0f}</b> predicted<extra></extra>",
    ))
    fig.update_layout(**_base(300, xaxis_title="Live simulation tick", yaxis_title="Available spaces"))
    _show(fig)


def historical_fit_chart(df: pd.DataFrame, hours: int = 24):
    """Historical Actual vs Predicted: model replayed over past hours."""
    from backend.data_service import zone_trends_map
    from ml.predictor import predict_facility_availability

    key = (id(df), hours)
    cached = _FIT_CACHE.get(key)
    if cached is not None:
        times, actual, predicted = cached
    else:
        stamps = sorted(df["timestamp"].unique())
        if len(stamps) > hours * 4:
            stamps = stamps[-(hours * 4):]
        # hourly replay points
        picks = stamps[::4] or stamps
        actual, predicted, times = [], [], []
        for ts in picks:
            snap = df[df["timestamp"] == ts]
            if snap.empty:
                continue
            occ = int(snap["occupancy"].sum())
            avl = 100 - occ
            try:
                from simulation.live_engine import get_zone_state
                z = get_zone_state(snap)
                rain = int(snap["rain_intensity"].max())
                p = predict_facility_availability(z, rain, zone_trends_map(df, z))
            except Exception:
                continue
            times.append(ts)
            actual.append(avl)
            predicted.append(round(p, 1))
        if len(_FIT_CACHE) > 4:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = (times, actual, predicted)
    if not times:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=actual, mode="lines", name="Actual",
        line=dict(width=2.2, color=BLUE, shape="spline", smoothing=0.5),
        hovertemplate="<b>%{y:.0f}</b> actual<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=predicted, mode="lines", name="Predicted",
        line=dict(width=2.2, color=AMBER, shape="spline", smoothing=0.5),
        hovertemplate="<b>%{y:.0f}</b> predicted<br>%{x|%d %b, %H:%M}<extra></extra>",
    ))
    fig.update_layout(**_base(340, yaxis_title="Available spaces"))
    fig.update_xaxes(tickformat="%H:%M\n%d %b", nticks=8)
    _show(fig)


def severity_donut(counts: dict):
    """Anomaly composition at a glance: how much is critical?"""
    labels = [k for k in ["Critical", "Warning", "Informational"] if counts.get(k, 0) > 0]
    values = [counts[k] for k in labels]
    if not labels:
        return
    colors = {"Critical": RED, "Warning": AMBER, "Informational": BLUE}
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=[colors[k] for k in labels], line=dict(color="#0e1b31", width=3)),
        textinfo="label+percent", textfont=dict(size=11, color="#e8eef6"),
        hovertemplate="<b>%{value}</b> %{label}<extra></extra>",
        sort=False,
    ))
    total = sum(values)
    fig.update_layout(
        **_base(240),
        showlegend=False,
        annotations=[dict(text=f"<b style='font-size:20px'>{total}</b><br><span style='font-size:11px;color:#8ea0b8'>cases</span>",
                          x=0.5, y=0.5, showarrow=False, align="center")],
    )
    _show(fig)


def zone_forecast_chart(names: list, free_now: list, free_pred: list):
    """Which zone will still have space in 30 minutes?"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=free_now, name="Free now",
        marker=dict(color=GREEN, line=dict(width=1, color="#0e1b31")),
        hovertemplate="<b>%{x}</b>: %{y:.0f} free now<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=names, y=free_pred, name="Predicted +30 min",
        marker=dict(color=AMBER, line=dict(width=1, color="#0e1b31")),
        hovertemplate="<b>%{x}</b>: %{y:.0f} predicted<extra></extra>",
    ))
    fig.update_layout(**_base(300, yaxis_title="Free bays", barmode="group", bargap=0.3, bargroupgap=0.15))
    _show(fig)


def occupancy_donut(available: int, occupied: int, labels: tuple[str, str] = ("Free", "Occupied"), center_suffix: str = "free"):
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=list(labels), values=[available, occupied], hole=0.68,
        marker=dict(colors=[GREEN, CHARCOAL], line=dict(color="white", width=3)),
        textinfo="none", hovertemplate="<b>%{value}</b> %{label}<br>%{percent}<extra></extra>",
        sort=False, direction="clockwise",
    ))
    total = available + occupied
    pct = (available / total * 100) if total else 0
    base = _base(250)
    base.pop("legend", None)
    fig.update_layout(
        **base,
        showlegend=False,
        annotations=[dict(text=f"<b style='font-size:22px'>{available}</b><br><span style='font-size:11px;color:#6b7672'>{center_suffix} • {pct:.0f}%</span>",
                          x=0.5, y=0.5, showarrow=False, align="center")],
    )
    _show(fig)


def zone_bar_h(zones: pd.DataFrame):
    """Which zone needs attention? Highest load first."""
    df = zones.sort_values("occupancy_rate", ascending=True).copy()
    color = df["demand_status"].map({"High": RED, "Moderate": AMBER, "Low": GREEN}).fillna(SLATE)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=(df["occupancy_rate"] * 100).round(1), y=df["zone"], orientation="h",
        marker=dict(color=color.tolist(), line=dict(width=1, color="white")),
        text=(df["occupancy_rate"] * 100).round(0).astype(int).astype(str) + "%",
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b>: %{x:.1f}% occupied<extra></extra>",
    ))
    fig.update_layout(**_base(240, xaxis_title="% occupied", yaxis_title=None, showlegend=False))
    fig.update_xaxes(range=[0, max(100, (df["occupancy_rate"].max() * 100) + 18)])
    _show(fig)


def zone_hourly_chart(df: pd.DataFrame):
    """Demand by hour: when does each zone peak?"""
    per_zone = df.groupby(["timestamp", "zone"]).agg(
        occupied=("occupancy", "sum"),
    ).reset_index()
    per_zone["hour"] = per_zone["timestamp"].dt.hour
    hourly = per_zone.groupby(["hour", "zone"])["occupied"].mean().reset_index()
    colors = {"Zone A": RED, "Zone B": AMBER, "Zone C": GREEN}
    fig = go.Figure()
    for zone in ["Zone A", "Zone B", "Zone C"]:
        z = hourly[hourly["zone"] == zone]
        fig.add_trace(go.Scatter(
            x=z["hour"], y=z["occupied"], mode="lines+markers", name=zone,
            line=dict(width=2.2, color=colors[zone], shape="spline", smoothing=0.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{zone}</b>: " + "%{y:.1f} avg occupied<br>hour %{x}:00<extra></extra>",
        ))
    fig.update_layout(**_base(300, xaxis_title="Hour of day", yaxis_title="Avg occupied bays"))
    fig.update_xaxes(dtick=2)
    _show(fig)


def rain_demand_chart(df: pd.DataFrame):
    """Does rain change demand? Avg arrivals per rain level."""
    labels = {0: "No rain", 1: "Light", 2: "Moderate", 3: "Heavy"}
    per_zone = df.groupby(["timestamp", "zone"]).agg(
        entries=("entries", "first"),
        rain=("rain_intensity", "max"),
    ).reset_index()
    g = per_zone.groupby("rain")["entries"].mean().reindex([0, 1, 2, 3]).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[labels[i] for i in g.index], y=g.values,
        marker=dict(color=[GREEN, BLUE, AMBER, RED], line=dict(width=1, color="white")),
        hovertemplate="<b>%{x}</b>: %{y:.1f} avg occupied<extra></extra>",
    ))
    fig.update_layout(**_base(280, xaxis_title=None, yaxis_title="Avg arrivals / interval", showlegend=False))
    _show(fig)


def anomaly_timeline(anomalies: pd.DataFrame):
    """When did abnormal behaviour occur?"""
    if anomalies.empty:
        return
    work = anomalies.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    daily = work.groupby([pd.Grouper(key="timestamp", freq="D"), "severity"]).size().reset_index(name="count")
    colors = {"Critical": RED, "Warning": AMBER, "Informational": BLUE}
    fig = go.Figure()
    for sev in ["Critical", "Warning", "Informational"]:
        s = daily[daily["severity"] == sev]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s["timestamp"], y=s["count"], name=sev,
            marker=dict(color=colors[sev], line=dict(width=1, color="white")),
            hovertemplate=f"<b>{sev}</b>: " + "%{y} cases<br>%{x|%d %b}<extra></extra>",
        ))
    fig.update_layout(**_base(260, yaxis_title="Cases", barmode="stack", bargap=0.4))
    fig.update_xaxes(tickformat="%d %b")
    _show(fig)


def prediction_error_chart(df: pd.DataFrame):
    """Prediction error distribution: how far off are forecasts?"""
    zone_pts = (
        df.dropna(subset=["future_availability"])
        .groupby(["timestamp", "zone"])
        .agg(now=("availability", "first"), later=("future_availability", "first"))
        .reset_index()
    )
    if len(zone_pts) > 6000:
        zone_pts = zone_pts.sample(6000, random_state=42)
    zone_pts["error"] = zone_pts["later"] - zone_pts["now"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=zone_pts["error"], nbinsx=40, name="Error",
        marker=dict(color=GREEN, line=dict(width=1, color="white")),
        hovertemplate="Error %{x:.0f}: %{y} cases<extra></extra>",
    ))
    fig.update_layout(**_base(280, xaxis_title="Error (actual future − now, bays)", yaxis_title="Cases", showlegend=False))
    _show(fig)


def feature_importance_chart(names: list, values: list):
    import numpy as np

    vals = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(vals))
    colors = [RED if v < 0 else GREEN for v in vals[order]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.abs(vals[order]), y=[names[i] for i in order], orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="white")),
        text=[f"{v:+.2f}" for v in vals[order]], textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
    ))
    fig.update_layout(**_base(300, xaxis_title="Effect strength |coef|", showlegend=False))
    _show(fig)


ZONES = ["Zone A", "Zone B", "Zone C"]
COLS = 10
COLORS = {"Available": "#34d399", "Occupied": "#f87171", "Offline": "#64748b"}
FILLS = {"Available": "rgba(52,211,153,0.16)", "Occupied": "rgba(248,113,113,0.16)", "Offline": "rgba(100,116,139,0.18)"}
def _zone_grid(zone_spaces: pd.DataFrame):
    xs, ys, texts, statuses = [], [], [], []
    ordered = zone_spaces.sort_values("parking_space_id")
    for k, (_, row) in enumerate(ordered.iterrows()):
        col = k % COLS
        r = k // COLS
        xs.append(col)
        ys.append(-r)
        health = int(row["occupancy_sensor_health"])
        occ = int(row["occupancy"])
        status = "Offline" if not health else ("Occupied" if occ else "Available")
        statuses.append(status)
        updated = pd.to_datetime(row["timestamp"]).strftime("%H:%M:%S")
        sensor = "Healthy" if health else "Offline"
        texts.append(
            f"<b>Space: {row['parking_space_id']}</b><br>"
            f"Status: {status}<br>"
            f"Zone: {row['zone']}<br>"
            f"Last update: {updated}<br>"
            f"Sensor: {sensor}"
        )
    return xs, ys, texts, statuses
def render_parking_map(
    spaces: pd.DataFrame,
    zone_filter: str = "All",
    search: str = "",
    anomalies: pd.DataFrame | None = None,
):
    st.subheader("Live parking slot map", anchor=False)
    st.caption("Green = available • red = occupied • grey = sensor offline. Hover any bay.")
    picked = st.segmented_control("Zone", ["All", "Zone A", "Zone B", "Zone C"],
                                  default=zone_filter if zone_filter in ("All", "Zone A", "Zone B", "Zone C") else "All")
    show_zones = ZONES if picked in (None, "All") else [picked]
    # Counts follow the selected zone(s) — All = facility, Zone A = only A.
    view = spaces[spaces["zone"].isin(show_zones)]
    total = len(view)
    free = int(((view["occupancy"] == 0) & (view["occupancy_sensor_health"] == 1)).sum())
    occ = int(((view["occupancy"] == 1) & (view["occupancy_sensor_health"] == 1)).sum())
    off = int((view["occupancy_sensor_health"] == 0).sum())
    anomaly_zones: set[str] = set()
    if anomalies is not None and not anomalies.empty and "zone" in anomalies.columns:
        anomaly_zones = set(anomalies["zone"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total bays", total)
    with c2:
        st.metric("Free", free, f"{free/total*100:.0f}%" if total else None)
    with c3:
        st.metric("Occupied", occ, f"{occ/total*100:.0f}%" if total else None, delta_color="inverse")
    with c4:
        st.metric("Offline", off)
    if anomaly_zones:
        st.caption(f"⚠️ Active anomaly signals in: {', '.join(sorted(anomaly_zones))}")
    n = len(show_zones)
    titles = [f"<b style='color:#e8eef6'>{z}</b>" + (" ⚠️" if z in anomaly_zones else "") for z in show_zones]
    fig = make_subplots(rows=n, cols=1, subplot_titles=titles, vertical_spacing=0.14)
    for idx, zone in enumerate(show_zones, start=1):
        zs = spaces[spaces["zone"] == zone].sort_values("parking_space_id")
        if zs.empty:
            continue
        xs, ys, texts, statuses = _zone_grid(zs)
        ids = zs["parking_space_id"].tolist()
        for status in ["Available", "Occupied", "Offline"]:
            sel = [(x, y, t, i) for x, y, t, s, i in zip(xs, ys, texts, statuses, ids) if s == status]
            if not sel:
                continue
            sx, sy, stx, six = zip(*sel)
            fig.add_trace(go.Scatter(
                x=list(sx), y=list(sy), mode="markers+text", name=status if idx == 1 else None,
                showlegend=(idx == 1), legendgroup=status, text=list(six), hovertext=list(stx), hoverinfo="text",
                textposition="middle center", textfont=dict(size=9, color="#e8eef6"),
                marker=dict(symbol="square", size=34, color=FILLS[status],
                            line=dict(width=1.5, color=COLORS[status])),
            ), row=idx, col=1)
        nrows = (len(zs) + COLS - 1) // COLS
        fig.update_xaxes(range=[-0.8, COLS - 0.2], showgrid=False, zeroline=False,
                         showticklabels=False, row=idx, col=1)
        fig.update_yaxes(range=[-nrows + 0.2, 0.8], showgrid=False, zeroline=False,
                         showticklabels=False, scaleanchor="x", row=idx, col=1)
    fig.update_layout(
        template="plotly_dark", height=250 * n + 70,
        margin=dict(l=8, r=8, t=46, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=11, color="#c6d2e2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    with st.expander("Bay list", expanded=False):
        view = spaces[spaces["zone"].isin(show_zones)].copy()
        if search:
            s = search.lower()
            view = view[view["parking_space_id"].str.lower().str.contains(s, na=False)]
        view["Status"] = view.apply(
            lambda r: "Offline" if int(r["occupancy_sensor_health"]) == 0
            else ("Occupied" if int(r["occupancy"]) == 1 else "Free"), axis=1)
        view["Sensor"] = view["occupancy_sensor_health"].map({1: "Healthy", 0: "Offline"})
        view["Load"] = view["occupancy"].astype(float)
        st.dataframe(
            view[["parking_space_id", "zone", "Status", "Sensor", "Load", "timestamp"]].rename(
                columns={"parking_space_id": "Bay", "zone": "Zone", "timestamp": "Last update"}),
            width="stretch", hide_index=True,
            column_config={
                "Load": st.column_config.ProgressColumn("Load", min_value=0, max_value=1),
                "Last update": st.column_config.DatetimeColumn("Last update", format="HH:mm:ss"),
            },
        )
