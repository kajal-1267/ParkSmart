from __future__ import annotations

import streamlit as st

from backend.data_service import (
    get_previous_hour_summary,
    get_trends,
    get_zone_trends,
    load_dataset,
    zone_trends_map,
)
from components.charts import (
    entry_exit_chart,
    live_tick_occupancy_chart,
    live_tick_prediction_chart,
    main_occupancy_forecast_chart,
    render_parking_map,
)
from components.ui import apply_styles, render_metric_card, render_status_badge
from ml.predictor import predict_facility_availability, predict_zone_availability
from views.anomalies import load_anomalies, recent_anomalies, render_anomalies_page
from views.ml_performance import render_ml_page
from views.predictions import render_prediction_page
from views.sensors import render_sensors_page
from views.zones import render_zones_page
from simulation.live_engine import (
    advance_live_state,
    aggregate_current_state,
    get_current_space_map,
    get_zone_state,
)

st.set_page_config(
    page_title="ParkSmart — Smart Parking Operations",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()


@st.cache_data(ttl=600, show_spinner="Updating parking intelligence...")
def load_data():
    return load_dataset()


def _tick() -> int:
    return int(st.session_state.get("ps_tick", 0))


def intelligent_status(summary, zones, change: float):
    occupancy = summary["occupancy_rate"]
    if occupancy >= 0.85:
        status, tone = "Near capacity", "critical"
    elif occupancy >= 0.70:
        status, tone = "Busy", "warning"
    elif occupancy >= 0.50:
        status, tone = "Moderately busy", "informational"
    else:
        status, tone = "Comfortable", "healthy"
    top = zones.sort_values("occupancy_rate", ascending=False).iloc[0]
    if top["occupancy_rate"] >= 0.75:
        zone_msg = f"{top['zone']} is nearing high occupancy ({top['occupancy_rate']*100:.0f}%). Guide drivers to other zones."
    else:
        zone_msg = f"{top['zone']} has the highest demand right now ({top['occupancy_rate']*100:.0f}% occupied)."
    if change > 3:
        trend_msg = "Availability is expected to improve over the next 30 minutes."
    elif change < -3:
        trend_msg = "Availability is expected to decline slightly over the next 30 minutes."
    else:
        trend_msg = "Availability is expected to stay stable over the next 30 minutes."
    return status, tone, zone_msg, trend_msg


def render_sidebar(anomaly_count: int):
    with st.sidebar:
        st.title("🅿️ ParkSmart")
        st.caption("SMART PARKING OPERATIONS")
        st.caption("NAVIGATION")
        page = st.radio(
            "Navigate",
            ["📊 Overview", "🔮 Predictions", "🗂 Zones", "📡 Sensors", "⚠️ Anomalies", "🤖 ML Performance"],
            label_visibility="collapsed",
        )
        page = page.split(" ", 1)[1] if " " in page else page
        st.divider()
        st.caption("SYSTEM STATUS")
        if anomaly_count > 0:
            st.warning(f"● {anomaly_count} anomalies need review", icon="⚠️")
        else:
            st.success("● System operating normally", icon="✅")
        interval = 5
        default_range = "24H"
        return page, interval, default_range


def overview_page(df, live_summary, zones, spaces, anomalies, settings):
    interval, default_range = settings
    now_stamp = live_summary["timestamp"]
    loaded_at = st.session_state.get("ps_loaded_at", now_stamp)
    try:
        elapsed = max(0, int((now_stamp - loaded_at).total_seconds()))
    except Exception:
        elapsed = 0

    _otmap = zone_trends_map(df, zones)
    _os = st.session_state.get("ps_stream", [])
    if len(_os) >= 2:
        _otail = _os[-4:]
        _okeys = (("za_in", "za_out"), ("zb_in", "zb_out"), ("zc_in", "zc_out"))
        for _oi, _on in enumerate(["Zone A", "Zone B", "Zone C"]):
            _ki, _ko = _okeys[_oi]
            _ins = [_r.get(_ki) for _r in _otail if _r.get(_ki) is not None]
            _outs = [_r.get(_ko) for _r in _otail if _r.get(_ko) is not None]
            if _ins:
                _otmap[_on]["entries_trend"] = sum(_ins) / len(_ins)
            if _outs:
                _otmap[_on]["exits_trend"] = sum(_outs) / len(_outs)
    prediction = predict_facility_availability(
        zones, live_summary["rain_intensity"], _otmap
    )
    change = prediction - live_summary["available"]
    pred_int = max(0, int(prediction + 0.5))
    pred_delta = f"{pred_int - live_summary['available']:+d} spaces"
    tick = _tick()

    # ---- Hero banner (ParkSmart identity, reference-style)
    st.markdown(
        """
        <div style="text-align:center; padding: 14px 0 4px; overflow: visible;">
            <div style="font-family:'Space Grotesk','Segoe UI',system-ui,sans-serif; font-weight:700;
                        font-size: clamp(2rem, 5vw, 3.2rem); color:#f2f6fb; line-height:1.25;
                        letter-spacing: 0.01em;">🅿️ ParkSmart</div>
            <div style="color:#9db1cc; font-size: clamp(0.85rem, 2vw, 1.05rem); margin-top: 6px;">
                AI-Based Smart Parking Availability Prediction &amp; Anomaly Detection</div>
            <div style="display:inline-block; margin-top: 10px; padding: 6px 18px; border-radius: 999px;
                        border: 1px solid #22354f; background: rgba(19,35,60,0.8); color: #c6d2e2; font-size: 0.85rem;">
                🅿️ 100 Bays &nbsp;•&nbsp; 🤖 Machine Learning &nbsp;•&nbsp; 📡 Real-Time Analytics
                &nbsp;•&nbsp; Zones A / B / C</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Command strip under the hero
    render_status_badge("● LIVE SIMULATION ACTIVE", "healthy")
    st.caption(f"● Live • updated {elapsed}s ago • tick {tick} • refreshing every {interval}s")

    # ---- 1. KPI row: TOTAL / OCCUPIED / AVAILABLE / PREDICTED (§25)
    prev = get_previous_hour_summary(df)
    if prev and prev["occupied"]:
        occ_delta = f"{(live_summary['occupied']-prev['occupied'])/prev['occupied']*100:+.1f}% vs prev hour"
        avl_delta = f"{live_summary['available']-prev['available']:+d} vs prev hour"
    else:
        occ_delta, avl_delta = "live state", "live state"
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Total spaces", "100", "A40 • B30 • C30")
    with k2:
        render_metric_card("Occupied spaces", str(live_summary["occupied"]), delta=occ_delta, delta_color="inverse", help_text="Currently occupied")
    with k3:
        render_metric_card("Available spaces", str(live_summary["available"]), delta=avl_delta, help_text="Currently available")
    with k4:
        render_metric_card("Predicted in 30 min", str(pred_int), delta=pred_delta, help_text="Expected free bays")

    # ---- Sim state in one quiet line (no confusing strip)
    st.caption(f"● Live • {now_stamp.strftime('%H:%M:%S')} • tick {tick} • refreshing every {interval}s")

    # ---- 2. Availability bar + status summary
    status, tone, zone_msg, trend_msg = intelligent_status(live_summary, zones, change)
    s1, s2 = st.columns([1.2, 1], gap="medium")
    with s1:
        with st.container(border=True):
            st.subheader("Current parking availability", anchor=False)
            st.write(f"**{live_summary['occupied']} / 100 spaces occupied ({live_summary['occupancy_rate']*100:.1f}%)**")
            st.progress(float(live_summary["occupancy_rate"]))
            direction = "Improving" if change > 3 else ("Declining" if change < -3 else "Stable")
            _rmse = 2.1
            try:
                from ml.predictor import model_metrics as _mm
                _rmse = float(_mm()["rmse"])
            except Exception:
                pass
            _lo, _hi = max(0, int(prediction - _rmse + 0.5)), max(0, int(prediction + _rmse + 0.5))
            st.write(f"**Forecast:** {pred_int} free in 30 min (likely {_lo}–{_hi}) — **{direction}**.")
    with s2:
        with st.container(border=True):
            st.caption("PARKING STATUS")
            render_status_badge(status, tone)
            st.write(f"{live_summary['available']} bays free. {zone_msg}")
            st.write(trend_msg)

    # ---- Live IoT sensor readings (reference-style row)
    st.subheader("📡 Live IoT sensor readings", anchor=False)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_metric_card("Occupancy sensor", str(live_summary["occupied"]), "bay detections")
    with r2:
        render_metric_card("Vehicle entry sensor", str(live_summary["entries"]), "this interval")
    with r3:
        render_metric_card("Vehicle exit sensor", str(live_summary["exits"]), "this interval")
    with r4:
        render_metric_card("Available spaces", str(live_summary["available"]), "right now")

    # ---- Live AI analysis (prediction + trend only; anomalies live in attention)
    st.subheader("🤖 Live AI analysis", anchor=False)
    a1, a2 = st.columns(2)
    with a1:
        render_metric_card("AI predicted future availability", str(pred_int), "30 min ahead")
    with a2:
        with st.container(border=True):
            arrow = "↗ Rising" if change > 3 else ("↘ Falling" if change < -3 else "— Stable")
            st.metric("Availability trend", arrow)
            st.caption(f"{pred_int - live_summary['available']:+d} bays expected")

    # ---- 5. Slot map (hero visual)
    with st.container(border=True):
        render_parking_map(spaces, anomalies=anomalies)

    # ---- 3. Main occupancy chart + forecast (§28–29) with ranges
    with st.container(border=True):
        st.subheader("📈 Live parking occupancy", anchor=False)
        h1, h2 = st.columns([3, 1])
        with h1:
            st.caption("Solid = history up to NOW • dashed = 30-min forecast")
        with h2:
            rng = st.segmented_control("Range", ["1H", "6H", "24H", "7D", "30D"], default=default_range)
        hours = {"1H": 1, "6H": 6, "24H": 24, "7D": 168, "30D": 720}.get(rng or default_range, 24)
        main_occupancy_forecast_chart(get_trends(df, hours=hours), live_summary["available"], prediction)

    # ---- 4. Zone intelligence (§30)
    st.subheader("Which zone needs attention?", anchor=False)
    tmap = zone_trends_map(df, zones)
    zcols = st.columns(3)
    for col, (_, z) in zip(zcols, zones.sort_values("occupancy_rate", ascending=False).iterrows()):
        with col:
            with st.container(border=True):
                h, b = st.columns([2, 1], vertical_alignment="center")
                with h:
                    st.write(f"**{z['zone']}**")
                    st.caption(f"{int(z['capacity'])} spaces")
                with b:
                    render_status_badge(f"{z['demand_status']}", z["demand_status"])
                st.metric("Available", int(z["available"]), f"{z['occupancy_rate']*100:.0f}% occupied", delta_color="inverse")
                st.progress(float(z["occupancy_rate"]))
                zcurr = {"occupied": int(z["occupied"]), "available": int(z["available"]),
                         "entries": int(z["entries"]), "exits": int(z["exits"]),
                         "vehicle_count": int(z["vehicle_count"]), "rain_intensity": live_summary["rain_intensity"],
                         "cumulative_entries": int(z.get("cumulative_entries", 0)),
                         "cumulative_exits": int(z.get("cumulative_exits", 0))}
                zcurr.update(tmap.get(z["zone"], {}))
                zpred = predict_zone_availability(zcurr, live_summary, int(z["capacity"]))
                zt = get_zone_trends(df, z["zone"], hours=1)
                if len(zt) >= 2 and zt["occupied"].iloc[-1] > zt["occupied"].iloc[0]:
                    trend = "↗ rising"
                elif len(zt) >= 2 and zt["occupied"].iloc[-1] < zt["occupied"].iloc[0]:
                    trend = "↘ easing"
                else:
                    trend = "→ steady"
                st.caption(f"Predicted in 30 min: **{max(0, int(zpred + 0.5))}** free • trend {trend}")

    # ---- 6. Vehicle flow (§32)
    with st.container(border=True):
        st.subheader("Are vehicles entering faster than leaving?", anchor=False)
        net = live_summary["entries"] - live_summary["exits"]
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("Entries", live_summary["entries"], "this interval")
        with f2:
            st.metric("Exits", live_summary["exits"], "this interval")
        with f3:
            st.metric("Net flow", f"{net:+d}", "in − out")
        with f4:
            st.metric("Vehicles (est.)", live_summary["vehicle_count"], "on site")
        entry_exit_chart(get_trends(df, hours=6))

    # ---- Live tick charts (reference-style: per-tick occupancy + prediction)
    stream = st.session_state.get("ps_stream", [])
    if len(stream) >= 2:
        import pandas as pd
        _sdf = pd.DataFrame(stream)
        lt1, lt2 = st.columns(2, gap="medium")
        with lt1:
            with st.container(border=True):
                st.subheader("📈 Live parking occupancy", anchor=False)
                st.caption("Occupied vs available per simulation tick")
                live_tick_occupancy_chart(_sdf)
        with lt2:
            with st.container(border=True):
                st.subheader("🤖 Live AI availability prediction", anchor=False)
                st.caption("Actual vs predicted per simulation tick")
                live_tick_prediction_chart(_sdf)
    else:
        st.caption("Live tick charts appear after a few refresh cycles.")

    # ---- Live sensor data stream (reference-style tick table)
    with st.container(border=True):
        st.subheader("📡 Live sensor data stream", anchor=False)
        st.caption("Every tick appended — newest first")
        if stream:
            import pandas as pd
            sview = pd.DataFrame(stream).tail(12).iloc[::-1]
            st.dataframe(
                sview[["tick", "time", "occupied", "available", "entries", "exits", "predicted", "anomaly"]].rename(
                    columns={"tick": "Tick", "time": "Time", "occupied": "Occupied", "available": "Available",
                             "entries": "Entry", "exits": "Exit", "predicted": "Predicted", "anomaly": "Anomaly"}),
                width="stretch", hide_index=True,
            )
        else:
            st.caption("Stream starts filling as ticks advance.")

    # ---- 7. System attention (§33)
    with st.container(border=True):
        st.subheader("Needs attention", anchor=False)
        sensing = spaces["occupancy_sensor_health"].mean() * 100 if len(spaces) else 100
        notes = []
        if len(anomalies):
            crit = len(anomalies[anomalies["severity"] == "Critical"])
            notes.append(f"🔴 {len(anomalies)} active anomalies ({crit} critical) — review on the Anomalies page.")
        top = zones.sort_values("occupancy_rate", ascending=False).iloc[0]
        if top["occupancy_rate"] >= 0.75:
            notes.append(f"🟠 {top['zone']} approaching high occupancy ({top['occupancy_rate']*100:.0f}%).")
        if sensing < 95:
            notes.append(f"🟠 Sensor health at {sensing:.0f}% — check the Sensors page.")
        if not notes:
            st.success("System operating normally. No active issues.", icon="✅")
        for note in notes:
            st.write(note)


@st.fragment(run_every=5)
def live_overview_fragment(active, interval, default_range):
    """Overview live region: advances every 5s without blocking navigation."""

    from datetime import datetime as _dt

    df = load_data()

    st.session_state["ps_tick"] = _tick() + 1
    st.session_state["ps_loaded_at"] = _dt.now()

    live = advance_live_state(df, _tick())
    live_summary = aggregate_current_state(live)
    zones = get_zone_state(live)
    from simulation.live_engine import get_current_space_map
    spaces = get_current_space_map(live)

    # Append this tick to the live sensor data stream (with spike check).
    _tmap = zone_trends_map(df, zones)
    _s = st.session_state.get("ps_stream", [])
    if len(_s) >= 2:
        # Live momentum overrides stale history: sustained inflow pushes
        # the forecast down, outflow pushes it up (§11, §14).
        _tail = _s[-4:]
        _keys = (("za_in", "za_out"), ("zb_in", "zb_out"), ("zc_in", "zc_out"))
        for _zi, _zn in enumerate(["Zone A", "Zone B", "Zone C"]):
            _ki, _ko = _keys[_zi]
            _ins = [_r.get(_ki) for _r in _tail if _r.get(_ki) is not None]
            _outs = [_r.get(_ko) for _r in _tail if _r.get(_ko) is not None]
            if _ins:
                _tmap[_zn]["entries_trend"] = sum(_ins) / len(_ins)
            if _outs:
                _tmap[_zn]["exits_trend"] = sum(_outs) / len(_outs)
    _stream_pred = predict_facility_availability(
        zones, live_summary["rain_intensity"], _tmap
    )
    _hist = get_trends(df, hours=24)["entries"]
    _mean = float(_hist.mean()) if len(_hist) else 0.0
    _std = float(_hist.std() or 0.0)
    _z = (live_summary["entries"] - _mean) / _std if _std else 0.0
    _spike = bool(len(_hist) and _z >= 2.0)
    if "ps_live_anomalies" not in st.session_state:
        st.session_state["ps_live_anomalies"] = []
    _live_anoms = st.session_state["ps_live_anomalies"]
    if _spike and (not _live_anoms or _live_anoms[-1]["tick"] != _tick()):
        _sev = "Critical" if _z >= 5 else ("Warning" if _z >= 4 else "Informational")
        _top_zone = zones.sort_values("entries", ascending=False).iloc[0]["zone"]
        _live_anoms.append({
            "timestamp": live_summary["timestamp"],
            "zone": _top_zone,
            "sensor": "Gate counters",
            "metric": "Entry flow",
            "observed_value": live_summary["entries"],
            "expected_range": f"≤ {_mean + 2 * _std:.1f} / interval",
            "z_score": round(_z, 2),
            "severity": _sev,
            "status": "Active",
            "tick": _tick(),
        })
        st.session_state["ps_live_anomalies"] = _live_anoms[-50:]
    _stream = st.session_state["ps_stream"]
    if not _stream or _stream[-1]["tick"] != _tick():
        _zrow = {z["zone"]: z for _, z in zones.iterrows()}

        def _zv(zone, field):
            row = _zrow.get(zone)
            try:
                return int(row.get(field, 0))
            except Exception:
                return 0

        _stream.append({
            "tick": _tick(),
            "time": live_summary["timestamp"].strftime("%H:%M:%S"),
            "occupied": live_summary["occupied"],
            "available": live_summary["available"],
            "entries": live_summary["entries"],
            "exits": live_summary["exits"],
            "predicted": max(0, int(_stream_pred + 0.5)),
            "anomaly": "Spike" if _spike else "Normal",
            "za_in": _zv("Zone A", "entries"),
            "za_out": _zv("Zone A", "exits"),
            "zb_in": _zv("Zone B", "entries"),
            "zb_out": _zv("Zone B", "exits"),
            "zc_in": _zv("Zone C", "entries"),
            "zc_out": _zv("Zone C", "exits"),
        })
        st.session_state["ps_stream"] = _stream[-120:]

    import pandas as _pd
    _combined = active
    _sess_anoms = st.session_state.get("ps_live_anomalies", [])
    if _sess_anoms:
        _ldf = _pd.DataFrame(_sess_anoms).drop(columns=["tick"], errors="ignore")
        _combined = _pd.concat([active, _ldf], ignore_index=True) if len(active) else _ldf
    overview_page(df, live_summary, zones, spaces, _combined, (interval, default_range))


def main():
    try:
        df = load_data()
    except FileNotFoundError as exc:
        st.title("ParkSmart")
        st.error("Parking data unavailable", icon="🚨")
        st.write(str(exc))
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    anomalies = load_anomalies()
    active = recent_anomalies(anomalies)
    live_anoms = st.session_state.get("ps_live_anomalies", [])
    page, interval, default_range = render_sidebar(len(active) + len(live_anoms))

    if "ps_tick" not in st.session_state:
        st.session_state["ps_tick"] = 0
    if "ps_stream" not in st.session_state:
        st.session_state["ps_stream"] = []
    if "ps_live_anomalies" not in st.session_state:
        st.session_state["ps_live_anomalies"] = []
    if "ps_loaded_at" not in st.session_state:
        from datetime import datetime as _dt
        st.session_state["ps_loaded_at"] = _dt.now()

    # Live state for non-Overview pages (static snapshot, no loop).
    live = advance_live_state(df, _tick())
    live_summary = aggregate_current_state(live)
    zones = get_zone_state(live)
    from simulation.live_engine import get_current_space_map
    spaces = get_current_space_map(live)

    if page == "Overview":
        live_overview_fragment(active, interval, default_range)
    elif page == "Predictions":
        render_prediction_page(df, live_summary, get_trends(df, hours=6), zones)
    elif page == "Zones":
        render_zones_page(df, zones)
    elif page == "Sensors":
        render_sensors_page(df, live)
    elif page == "Anomalies":
        render_anomalies_page()
    elif page == "ML Performance":
        render_ml_page(df)


if __name__ == "__main__":
    main()
