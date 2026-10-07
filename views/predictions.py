from __future__ import annotations

import streamlit as st

from backend.data_service import zone_trends_map
from components.charts import forecast_chart, zone_forecast_chart
from components.ui import render_metric_card, render_status_badge
from ml.predictor import model_metrics, predict_facility_availability, predict_zone_availability
from simulation.live_engine import get_zone_state

CAPACITY = {"All Zones": 100, "Zone A": 40, "Zone B": 30, "Zone C": 30}


def _zone_current(df, zone: str, rain: int) -> dict:
    """True-scale zone state: bay sums + zone sensor values."""

    latest = df[df["timestamp"] == df["timestamp"].max()]
    latest = latest[latest["zone"] == zone]

    return {
        "occupied": int(latest["occupancy"].sum()),
        "available": int(latest["availability"].sum()),
        "entries": int(latest["entries"].iloc[0]),
        "exits": int(latest["exits"].iloc[0]),
        "vehicle_count": int(latest["estimated_vehicle_count"].iloc[0]),
        "rain_intensity": rain,
        "cumulative_entries": int(latest["cumulative_entries"].iloc[0]),
        "cumulative_exits": int(latest["cumulative_exits"].iloc[0]),
    }


def render_prediction_page(df, summary, trends, zones_live=None):
    st.caption("ParkSmart / Availability Forecast")
    st.title("Forecast")
    st.caption("Free bays expected in the next 30 minutes, per zone.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            zone = st.selectbox("Zone", ["All Zones", "Zone A", "Zone B", "Zone C"])
        with c2:
            st.selectbox("Horizon", ["30 minutes"], index=0, disabled=True)

    rain_labels = {0: "No rain", 1: "Light rain", 2: "Moderate rain", 3: "Heavy rain"}
    rain = int(summary["rain_intensity"])
    st.caption(f"Model inputs: live occupancy, gate flow, vehicle estimate • Rain now: {rain_labels.get(rain, '—')}")

    if zone == "All Zones":
        current = {
            "occupied": summary["occupied"],
            "available": summary["available"],
            "entries": summary["entries"],
            "exits": summary["exits"],
            "vehicle_count": summary["vehicle_count"],
            "rain_intensity": rain,
        }
        zones_live = zones_live if zones_live is not None and len(zones_live) else get_zone_state(df)
        tmap = zone_trends_map(df, zones_live)
        prediction = predict_facility_availability(zones_live, rain, tmap)
    else:
        current = _zone_current(df, zone, rain)
        _zl = zones_live if zones_live is not None and len(zones_live) else get_zone_state(df)
        current.update(zone_trends_map(df, _zl).get(zone, {}))
        prediction = predict_zone_availability(
            current, summary, CAPACITY[zone]
        )
    change = prediction - current["available"]
    pred_int = max(0, int(prediction + 0.5))
    rmse = model_metrics()["rmse"]
    lo, hi = max(0, int(prediction - rmse + 0.5)), max(0, int(prediction + rmse + 0.5))
    cap = CAPACITY[zone]
    occ_pred = max(0, min(100, 100 - (prediction / cap * 100)))
    demand = "High" if occ_pred >= 75 else ("Moderate" if occ_pred >= 50 else "Low")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Predicted free", str(pred_int), f"likely {lo}–{hi}", delta=f"{pred_int - current['available']:+d}")
    with m2:
        render_metric_card("Free now", str(current["available"]), zone)
    with m3:
        render_metric_card("Load then", f"{occ_pred:.0f}%", "expected")
    with m4:
        with st.container(border=True):
            st.metric("Demand then", demand)
            render_status_badge(demand + " demand", demand)

    with st.container(border=True):
        v1, v2 = st.columns([1, 2], vertical_alignment="center")
        with v1:
            if change < -5:
                render_status_badge("↘ declining", "warning")
            elif change > 5:
                render_status_badge("↗ improving", "healthy")
            else:
                render_status_badge("→ stable", "informational")
        with v2:
            if change < -5:
                st.write(f"**≈{abs(change):.0f} bays will fill** — divert drivers to emptier zones now.")
            elif change > 5:
                st.write(f"**≈{change:.0f} bays will free up** — pressure easing, no action needed.")
            else:
                st.write(f"**Load holds steady** ({current['available']} → {pred_int}) — routine monitoring.")
        st.caption("Verdict recomputed live from occupancy, gate flow, rain and time of day.")

    with st.container(border=True):
        st.subheader("30-minute path", anchor=False)
        st.caption("Solid = observed free bays • dashed = model path with ±8% band")
        forecast_chart(trends, current["available"], prediction)

    with st.container(border=True):
        st.subheader("All zones in 30 minutes", anchor=False)
        st.caption("Where will space still be? Green = now, amber = forecast")
        _zl = get_zone_state(df)
        _tmap = zone_trends_map(df, _zl)
        _names, _now, _pred = [], [], []
        for _, _z in _zl.iterrows():
            _zc = {"occupied": int(_z["occupied"]), "available": int(_z["available"]),
                   "entries": int(_z["entries"]), "exits": int(_z["exits"]),
                   "vehicle_count": int(_z["vehicle_count"]), "rain_intensity": rain,
                   "cumulative_entries": int(_z.get("cumulative_entries", 0)),
                   "cumulative_exits": int(_z.get("cumulative_exits", 0))}
            _zc.update(_tmap.get(_z["zone"], {}))
            _names.append(_z["zone"])
            _now.append(int(_z["available"]))
            _pred.append(max(0, int(predict_zone_availability(
                _zc, {"entries": int(_z["entries"]), "exits": int(_z["exits"])}, int(_z["capacity"])) + 0.5)))
        zone_forecast_chart(_names, _now, _pred)

    st.subheader("Why this number?", anchor=False)
    load_rate = current["occupied"] / cap
    factors = [
        ("Occupancy load", "High impact" if load_rate > 0.70 else "Moderate impact",
         f"{current['occupied']} bays taken in {zone}."),
        ("Gate flow", "Negative impact" if current["entries"] > current["exits"] else "Positive impact",
         f"{current['entries']} in vs {current['exits']} out last interval."),
        ("Rain", "Positive impact" if current["rain_intensity"] >= 2 else "Low impact",
         f"{rain_labels.get(current['rain_intensity'], '—')} — rain keeps cars parked longer."),
        ("Zone habit", "High impact" if zone == "Zone A" else "Moderate impact",
         "Zone A fills first on most days."),
    ]
    cols = st.columns(4)
    for col, (name, impact, desc) in zip(cols, factors):
        with col:
            with st.container(border=True):
                st.write(f"**{name}**")
                render_status_badge(impact, impact)
                st.caption(desc)
