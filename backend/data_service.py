from __future__ import annotations

import os

import pandas as pd

from simulation.live_engine import (
    TOTAL_SPACES,
    aggregate_current_state,
    get_current_space_map,
    get_zone_state,
)


DATA_PATH = (
    "data/parking_dataset.csv"
)

# Aggregation rules keep the dashboard fast (§47):
# short ranges stay granular, long ranges aggregate.
AGG_RULES = [
    (1, "5min"),
    (6, "15min"),
    (24, "h"),
    (168, "h"),
    (720, "D"),
    (float("inf"), "D"),
]

# In-memory cache: the historical dataset never changes at runtime,
# so trends are computed once per (dataset, hours) — no 288k-row
# regroup on every refresh.
_TRENDS_CACHE: dict = {}
_ZONE_TRENDS_CACHE: dict = {}


def _cache_get(cache: dict, key, limit: int = 16):
    if key in cache:
        return cache[key].copy()
    return None


def _cache_put(cache: dict, key, value, limit: int = 16):
    if len(cache) >= limit:
        cache.clear()
    cache[key] = value


def load_dataset() -> pd.DataFrame:

    if not os.path.exists(
        DATA_PATH
    ):
        raise FileNotFoundError(
            "Parking data unavailable. "
            "We couldn't retrieve the latest parking data. "
            "Run simulation/simulator.py first, then retry."
        )

    return pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "timestamp"
        ],
    )


def get_dashboard_summary(
    df: pd.DataFrame,
) -> dict:

    return aggregate_current_state(
        df
    )


def get_zones(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return get_zone_state(
        df
    )


def get_space_map(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return get_current_space_map(
        df
    )


def _resample_rule(hours: int) -> str:
    for limit, rule in AGG_RULES:
        if hours <= limit:
            return rule
    return "D"


def get_trends(
    df: pd.DataFrame,
    hours: int = 24,
) -> pd.DataFrame:
    """Aggregated facility trends ending at NOW (latest timestamp)."""

    cached = _cache_get(_TRENDS_CACHE, (id(df), hours))
    if cached is not None:
        return cached

    latest_timestamp = df[
        "timestamp"
    ].max()

    start_time = (
        latest_timestamp
        - pd.Timedelta(
            hours=hours
        )
    )

    filtered = df[
        df["timestamp"] >= start_time
    ].copy()

    # Zone truth first (flow sensors repeat per bay), then facility sums.
    zoned = (
        filtered
        .groupby(["timestamp", "zone"])
        .agg(
            occupied=("occupancy", "sum"),
            available=("availability", "sum"),
            entries=("entries", "first"),
            exits=("exits", "first"),
            vehicle_count=("estimated_vehicle_count", "first"),
            rain=("rain_intensity", "max"),
        )
        .reset_index()
    )

    trends = (
        zoned
        .groupby("timestamp")
        .agg(
            occupied=(
                "occupied",
                "sum",
            ),
            available=(
                "available",
                "sum",
            ),
            entries=(
                "entries",
                "sum",
            ),
            exits=(
                "exits",
                "sum",
            ),
            vehicle_count=(
                "vehicle_count",
                "sum",
            ),
            rain=(
                "rain",
                "max",
            ),
        )
        .reset_index()
    )

    # Aggregate long ranges so charts stay fast.
    rule = _resample_rule(hours)
    if rule in ("h", "D") and len(trends) > 60:
        trends = (
            trends.set_index("timestamp")
            .resample(rule)
            .agg(
                {
                    "occupied": "mean",
                    "available": "mean",
                    "entries": "sum",
                    "exits": "sum",
                    "vehicle_count": "mean",
                    "rain": "max",
                }
            )
            .round(1)
            .reset_index()
            .sort_values("timestamp")
        )

    trends["occupancy_rate"] = (
        trends["occupied"]
        / TOTAL_SPACES
    )

    _cache_put(_TRENDS_CACHE, (id(df), hours), trends)

    return trends


def get_previous_hour_summary(
    df: pd.DataFrame,
) -> dict | None:
    """State ~60 min before NOW for KPI deltas. None when unavailable."""

    latest = df["timestamp"].max()
    target = latest - pd.Timedelta(hours=1)
    past = df[df["timestamp"] <= target]

    if past.empty:
        return None

    snap_time = past["timestamp"].max()
    snap = df[df["timestamp"] == snap_time]

    occupied = int(snap["occupancy"].sum())

    return {
        "timestamp": snap_time,
        "occupied": occupied,
        "available": TOTAL_SPACES - occupied,
    }


def get_zone_trends(
    df: pd.DataFrame,
    zone: str,
    hours: int = 24,
) -> pd.DataFrame:
    """Per-zone occupancy trend for the Zones workspace."""

    cached = _cache_get(_ZONE_TRENDS_CACHE, (id(df), zone, hours))
    if cached is not None:
        return cached

    latest = df["timestamp"].max()
    start = latest - pd.Timedelta(hours=hours)

    filtered = df[
        (df["timestamp"] >= start)
        & (df["zone"] == zone)
    ].copy()

    trends = (
        filtered
        .groupby("timestamp")
        .agg(
            occupied=("occupancy", "sum"),
            available=("availability", "sum"),
            entries=("entries", "first"),
            exits=("exits", "first"),
        )
        .reset_index()
    )

    rule = _resample_rule(hours)
    if rule in ("h", "D") and len(trends) > 60:
        trends = (
            trends.set_index("timestamp")
            .resample(rule)
            .agg({"occupied": "mean", "available": "mean", "entries": "sum", "exits": "sum"})
            .round(1)
            .reset_index()
            .sort_values("timestamp")
        )

    _cache_put(_ZONE_TRENDS_CACHE, (id(df), zone, hours), trends)

    return trends


def zone_trends_map(
    df: pd.DataFrame,
    zones_df: pd.DataFrame,
    hours: int = 2,
) -> dict:
    """Recent-trend features per zone for the model (rolling means)."""

    out = {}

    for zone in zones_df["zone"]:
        zt = get_zone_trends(df, zone, hours=hours)

        out[zone] = {
            "occ_trend": float(zt["occupied"].tail(4).mean()) if len(zt) else 0.0,
            "entries_trend": float(zt["entries"].tail(4).mean()) if len(zt) else 0.0,
            "exits_trend": float(zt["exits"].tail(4).mean()) if len(zt) else 0.0,
        }

    return out


def get_recent_period(
    df: pd.DataFrame,
    hours: int,
) -> pd.DataFrame:

    latest = df[
        "timestamp"
    ].max()

    start = (
        latest
        - pd.Timedelta(
            hours=hours
        )
    )

    return df[
        df["timestamp"] >= start
    ].copy()
