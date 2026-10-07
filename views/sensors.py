from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.data_service import get_trends
from components.charts import entry_exit_chart, occupancy_donut
from components.ui import render_status_badge
from views.anomalies import load_anomalies


def render_sensors_page(df, live=None):
    st.caption("ParkSmart / Sensors")
    st.title("Sensors")
    st.caption("Live bay health plus 24-hour gate activity.")

    # Live snapshot drives health; history drives trends.
    live_latest = live if live is not None and len(live) else df[df["timestamp"] == df["timestamp"].max()]
    latest = live_latest
    total = len(latest)
    healthy = int(latest["occupancy_sensor_health"].sum())
    score = (healthy / total * 100) if total else 0
    level = "Healthy" if score >= 95 else ("Warning" if score >= 85 else "Critical")

    # True-scale flow: sensor values repeat per bay, take zone values.
    zmeans = latest.groupby("zone")[["entries", "exits", "estimated_vehicle_count"]].first().sum()
    true_entries, true_exits = int(zmeans["entries"]), int(zmeans["exits"])
    true_vc = int(zmeans["estimated_vehicle_count"])

    left, right = st.columns([1, 1.4], gap="medium")
    with left:
        with st.container(border=True):
            st.subheader("Network health", anchor=False)
            occupancy_donut(healthy, total - healthy, labels=("Online", "Offline"), center_suffix="online")
            st.caption(f"{healthy}/{total} bay sensors online")
    with right:
        with st.container(border=True):
            st.subheader("Status", anchor=False)
            render_status_badge(level, level)
            st.metric("Health score", f"{score:.0f}%", f"{healthy}/{total} online")
            st.progress(healthy / total if total else 0)
            st.caption(f"Last ping: {latest['timestamp'].max()}")
            if score >= 95:
                st.success("All sensors reporting normally.", icon="✅")
            elif score >= 85:
                st.warning("Some sensors degraded — see action list.", icon="⚠️")
            else:
                st.error("Sensor network needs attention.", icon="🚨")

    groups = [
        ("🚗 Bay occupancy", f"{healthy}/{total} units", f"{latest['occupancy'].sum():.0f} detections",
         ("Distance", "Occupancy")),
        ("🔁 Gate counters", "Entry + exit gates", f"{true_entries} in / {true_exits} out",
         ("Entry", "Exit")),
        ("🚙 Vehicle estimator", "Fused estimate", f"{true_vc} vehicles",
         ("Vehicle",)),
        ("🌧️ Rain gauge", "Weather", f"Level {int(latest['rain_intensity'].max())}",
         ()),
    ]
    anomalies = load_anomalies()
    if not anomalies.empty:
        _cutoff = pd.to_datetime(anomalies["timestamp"]).max() - pd.Timedelta(hours=24)
        anomalies = anomalies[pd.to_datetime(anomalies["timestamp"]) >= _cutoff].copy()
    _rank = {"Critical": 3, "Warning": 2, "Informational": 1}
    for name, s1, s2, keywords in groups:
        count = 0
        worst = ""
        if not anomalies.empty and keywords:
            blob = (anomalies["sensor"].fillna("") + " " + anomalies["metric"].fillna(""))
            hit = anomalies[blob.str.lower().apply(lambda t: any(k.lower() in t for k in keywords))]
            count = len(hit)
            if count:
                worst = max(hit["severity"], key=lambda s: _rank.get(s, 0))
        badge = worst if worst else ("Healthy" if score >= 95 else ("Warning" if score >= 85 else "Critical"))
        with st.container(border=True):
            a, b, c, d = st.columns([2, 1, 1, 1], vertical_alignment="center")
            with a:
                st.write(f"**{name}**")
                st.caption(s1 + " • last 24h")
            with b:
                st.metric("Reading", s2)
            with c:
                st.metric("Anomalies", count)
            with d:
                render_status_badge(badge, badge)

    with st.container(border=True):
        st.subheader("Gate flow — last 24 hours", anchor=False)
        st.caption("Entries vs exits: are counters alive and balanced?")
        entry_exit_chart(get_trends(df, hours=24))

    with st.container(border=True):
        st.subheader("Action list — offline bays", anchor=False)
        off = latest[latest["occupancy_sensor_health"] == 0][["parking_space_id", "zone", "timestamp"]]
        if off.empty:
            st.success("Nothing to fix — zero offline sensors.", icon="✅")
        else:
            st.caption(f"{len(off)} bays need a visit")
            st.dataframe(
                off.rename(columns={"parking_space_id": "Bay", "zone": "Zone", "timestamp": "Last seen"}),
                width="stretch", hide_index=True,
                column_config={"Last seen": st.column_config.DatetimeColumn("Last seen", format="D MMM, HH:mm")},
            )
