from __future__ import annotations

import streamlit as st

from backend.data_service import get_zone_trends, zone_trends_map
from components.charts import occupancy_chart, rain_demand_chart, zone_bar_h, zone_hourly_chart
from components.ui import render_status_badge
from ml.predictor import predict_zone_availability


def render_zones_page(df, zones):
    st.caption("ParkSmart / Zones")
    st.title("Zones")
    st.caption("Which zone needs attention — and what happens next there?")

    filt = st.segmented_control("View", ["All Zones", "Zone A", "Zone B", "Zone C"], default="All Zones")
    view = zones.copy() if filt in (None, "All Zones") else zones[zones["zone"] == filt]
    tmap = zone_trends_map(df, zones)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Zones", len(view))
    with m2:
        st.metric("Total bays", int(view["capacity"].sum()))
    with m3:
        st.metric("Free", int(view["available"].sum()))
    with m4:
        avg = (view["occupied"].sum() / view["capacity"].sum() * 100) if view["capacity"].sum() else 0
        st.metric("Avg load", f"{avg:.0f}%")

    with st.container(border=True):
        st.subheader("Load comparison", anchor=False)
        st.caption("Highest first — divert drivers toward greener zones")
        zone_bar_h(view)

    with st.container(border=True):
        st.subheader("Zone list", anchor=False)
        st.caption("Sorted by load, highest first")
        table = view.sort_values("occupancy_rate", ascending=False).copy()
        table["Load"] = table["occupancy_rate"].astype(float)
        table["Load %"] = (table["occupancy_rate"] * 100).round(1)
        st.dataframe(
            table[["zone", "capacity", "occupied", "available", "Load %", "Load", "demand_status"]].rename(
                columns={"zone": "Zone", "capacity": "Capacity", "occupied": "Used",
                         "available": "Free", "demand_status": "Demand"}),
            width="stretch", hide_index=True,
            column_config={
                "Load": st.column_config.ProgressColumn("Bar", min_value=0, max_value=1),
                "Load %": st.column_config.NumberColumn("Load %", format="%.1f"),
            },
        )

    for _, z in view.iterrows():
        with st.container(border=True):
            h1, h2 = st.columns([3, 1], vertical_alignment="center")
            with h1:
                st.subheader(z["zone"], anchor=False)
                st.caption(f"{int(z['available'])} free of {int(z['capacity'])} • "
                           f"{int(z['occupied'])} occupied")
            with h2:
                render_status_badge(f"{z['demand_status']} demand", z["demand_status"])
            st.progress(float(z["occupancy_rate"]))

            zt = get_zone_trends(df, z["zone"], hours=24)
            peak = float(zt["occupied"].max()) if len(zt) else float(z["occupied"])
            avgz = float(zt["occupied"].mean()) if len(zt) else float(z["occupied"])
            rain = int(df[df["timestamp"] == df["timestamp"].max()]["rain_intensity"].max())
            zcurr = {"occupied": int(z["occupied"]), "available": int(z["available"]),
                     "entries": int(z["entries"]), "exits": int(z["exits"]),
                     "vehicle_count": int(z["vehicle_count"]), "rain_intensity": rain,
                     "cumulative_entries": int(z.get("cumulative_entries", 0)),
                     "cumulative_exits": int(z.get("cumulative_exits", 0))}
            zcurr.update(tmap.get(z["zone"], {}))
            zsummary = {"entries": int(z["entries"]), "exits": int(z["exits"])}
            zpred = predict_zone_availability(zcurr, zsummary, int(z["capacity"]))
            zpred_int = max(0, int(zpred + 0.5))

            # Occupied → Available → Predicted (§30 order)
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("Occupied", int(z["occupied"]), f"{z['occupancy_rate']*100:.0f}% full")
            with o2:
                st.metric("Available", int(z["available"]), "free now")
            with o3:
                st.metric("Predicted free", zpred_int, "+30 min")

            with st.expander(f"Flow + trend — {z['zone']}"):
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Peak (24h)", f"{peak:.0f}")
                with s2:
                    st.metric("Average (24h)", f"{avgz:.0f}")
                with s3:
                    st.metric("Entries", int(z["entries"]))
                with s4:
                    st.metric("Exits", int(z["exits"]))
                st.caption("Occupied vs free bays over the last 24 hours")
                occupancy_chart(zt)

    with st.container(border=True):
        st.subheader("When does each zone peak?", anchor=False)
        st.caption("Average occupied bays by hour of day")
        zone_hourly_chart(df)

    with st.container(border=True):
        st.subheader("Does rain change demand?", anchor=False)
        st.caption("Fewer arrivals as rain gets heavier — plan staffing accordingly")
        rain_demand_chart(df)
