from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from components.charts import anomaly_timeline, severity_donut
from components.ui import render_status_badge


def load_anomalies():
    return _load_anomalies_cached()


@st.cache_data(ttl=300, show_spinner=False)
def _load_anomalies_cached():
    path = "data/anomalies.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def recent_anomalies(anomalies: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Only recent cases drive sidebar badges and attention (§33: no alert noise)."""

    if anomalies.empty:
        return anomalies

    cutoff = pd.to_datetime(anomalies["timestamp"]).max() - pd.Timedelta(hours=hours)

    return anomalies[pd.to_datetime(anomalies["timestamp"]) >= cutoff].copy()


def render_anomalies_page():
    st.caption("ParkSmart / Anomaly Center")
    st.title("Anomalies")
    st.caption("Live detection running • cases update automatically")

    anomalies = load_anomalies()
    if anomalies.empty:
        st.success("No anomalies — all sensors within expected ranges.", icon="✅")
        return

    live_anoms = st.session_state.get("ps_live_anomalies", [])
    if live_anoms:
        with st.container(border=True):
            st.subheader("Live detections — this session", anchor=False)
            st.caption("Caught by the running simulation just now")
            import pandas as _pd
            _ldf = _pd.DataFrame(live_anoms).drop(columns=["tick"], errors="ignore")
            st.dataframe(_ldf, width="stretch", hide_index=True)

    critical = len(anomalies[anomalies["severity"] == "Critical"])
    warning = len(anomalies[anomalies["severity"] == "Warning"])
    info = len(anomalies[anomalies["severity"] == "Informational"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Open cases", len(anomalies))
    with m2:
        st.metric("Critical", critical)
    with m3:
        st.metric("Warning", warning)
    with m4:
        st.metric("Info", info)

    with st.container(border=True):
        st.subheader("Composition", anchor=False)
        st.caption("Share of each severity right now")
        severity_donut({"Critical": critical, "Warning": warning, "Informational": info})

    with st.container(border=True):
        st.subheader("When did abnormal behaviour occur?", anchor=False)
        st.caption("Cases per day, stacked by severity")
        anomaly_timeline(anomalies)

    filtered = anomalies.sort_values("timestamp", ascending=False).copy()

    with st.container(border=True):
        st.subheader(f"Case list ({len(filtered)})", anchor=False)
        st.caption("Newest first • z-score = deviation strength")
        st.dataframe(
            filtered[["timestamp", "zone", "sensor", "metric", "observed_value",
                      "expected_range", "z_score", "severity", "status"]],
            width="stretch", hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("When", format="D MMM, HH:mm"),
                "zone": st.column_config.TextColumn("Zone"),
                "sensor": st.column_config.TextColumn("Sensor"),
                "metric": st.column_config.TextColumn("Metric"),
                "observed_value": st.column_config.NumberColumn("Observed", format="%.1f"),
                "expected_range": st.column_config.TextColumn("Expected"),
                "z_score": st.column_config.NumberColumn("Z", format="%.2f"),
                "severity": st.column_config.TextColumn("Severity"),
                "status": st.column_config.TextColumn("Status"),
            },
        )
        st.download_button("⬇ Export filtered", filtered.to_csv(index=False),
                           file_name="anomalies_filtered.csv", mime="text/csv")

    st.subheader("Case details", anchor=False)
    sel_idx = st.selectbox("Select case", filtered.index,
                           format_func=lambda x: f"{filtered.loc[x, 'severity']} • {filtered.loc[x, 'zone']} • {filtered.loc[x, 'metric']}")
    sel = filtered.loc[sel_idx]
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            st.write(f"**{sel['sensor']}** — {sel['metric']}")
            st.caption(f"{sel['timestamp']} • {sel['zone']} • {sel['status']}")
        with c2:
            render_status_badge(str(sel["severity"]), str(sel["severity"]))
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Observed", f"{sel['observed_value']}")
        with d2:
            st.metric("Expected", f"{sel['expected_range']}")
        with d3:
            st.metric("Z-score", f"{sel['z_score']}")
        st.info("Inspect the sensor, verify gate counts, confirm if this is real demand or a fault.", icon="🛠️")
