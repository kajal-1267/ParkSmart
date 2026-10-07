from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


TOTAL_SPACES = 100

ZONE_CAPACITY = {
    "Zone A": 40,
    "Zone B": 30,
    "Zone C": 30,
}

# Centralized configuration (§55).
ZONE_A_CAPACITY = 40
ZONE_B_CAPACITY = 30
ZONE_C_CAPACITY = 30
PREDICTION_HORIZON_MINUTES = 30
DATA_INTERVAL_MINUTES = 15
RANDOM_SEED = 42

# Max bays that flip state per live tick: keeps motion gradual (§7).
LIVE_MAX_FLIPS = 3


def get_current_time() -> datetime:

    return datetime.now()


def get_latest_dataset_state(
    df: pd.DataFrame,
) -> pd.DataFrame:

    latest_timestamp = df[
        "timestamp"
    ].max()

    latest = df[
        df["timestamp"] == latest_timestamp
    ].copy()

    return latest


def _zone_flow_frame(
    latest: pd.DataFrame,
) -> pd.DataFrame:
    """Zone-level truth from the expanded space rows.

    occupancy/availability sum over bays (true totals), while flow
    sensors (entries/exits/vehicle/rain) are identical across a zone's
    bays — so the zone truth is their first value, NOT the sum.
    """

    grouped = (
        latest
        .groupby("zone")
        .agg(
            occupied=(
                "occupancy",
                "sum",
            ),
            available=(
                "availability",
                "sum",
            ),
            entries=(
                "entries",
                "first",
            ),
            exits=(
                "exits",
                "first",
            ),
            vehicle_count=(
                "estimated_vehicle_count",
                "first",
            ),
            rain=(
                "rain_intensity",
                "max",
            ),
            cumulative_entries=(
                "cumulative_entries",
                "first",
            ),
            cumulative_exits=(
                "cumulative_exits",
                "first",
            ),
        )
        .reset_index()
    )

    return grouped


def aggregate_current_state(
    df: pd.DataFrame,
) -> dict:

    latest = get_latest_dataset_state(
        df
    )

    occupied = int(
        latest["occupancy"].sum()
    )

    available = (
        TOTAL_SPACES
        - occupied
    )

    flow = _zone_flow_frame(latest)

    entries = int(
        flow["entries"].sum()
    )

    exits = int(
        flow["exits"].sum()
    )

    vehicle_count = int(
        flow["vehicle_count"].sum()
    )

    rain = int(
        flow["rain"].max()
    ) if len(flow) else 0

    return {
        "timestamp": latest[
            "timestamp"
        ].iloc[0],
        "occupied": occupied,
        "available": available,
        "occupancy_rate": (
            occupied / TOTAL_SPACES
        ),
        "entries": entries,
        "exits": exits,
        "vehicle_count": vehicle_count,
        "rain_intensity": rain,
    }


def get_zone_state(
    df: pd.DataFrame,
) -> pd.DataFrame:

    latest = get_latest_dataset_state(
        df
    )

    grouped = _zone_flow_frame(latest)

    grouped["capacity"] = (
        grouped["zone"]
        .map(ZONE_CAPACITY)
    )

    grouped["occupancy_rate"] = (
        grouped["occupied"]
        / grouped["capacity"]
    )

    grouped["availability_rate"] = (
        grouped["available"]
        / grouped["capacity"]
    )

    grouped["demand_status"] = (
        grouped["occupancy_rate"]
        .apply(
            lambda x:
            "High"
            if x >= 0.75
            else (
                "Moderate"
                if x >= 0.50
                else "Low"
            )
        )
    )

    return grouped


def get_current_space_map(
    df: pd.DataFrame,
) -> pd.DataFrame:

    latest = get_latest_dataset_state(
        df
    )

    return latest[
        [
            "parking_space_id",
            "zone",
            "occupancy",
            "availability",
            "occupancy_sensor_health",
            "timestamp",
        ]
    ].copy()


def generate_live_snapshot(
    historical_df: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:

    """
    Creates a live operational snapshot.

    The historical dataset remains untouched.

    Current dashboard values are generated from
    the latest state with controlled variation.
    """

    rng = np.random.default_rng(
        seed
    )

    latest = get_latest_dataset_state(
        historical_df
    ).copy()

    # Controlled live variation.
    for index in latest.index:

        current = int(
            latest.at[
                index,
                "occupancy"
            ]
        )

        capacity = ZONE_CAPACITY[
            latest.at[
                index,
                "zone"
            ]
        ]

        change = int(
            rng.choice(
                [-1, 0, 0, 0, 1]
            )
        )

        latest.at[
            index,
            "occupancy"
        ] = int(
            np.clip(
                current + change,
                0,
                1,
            )
        )

    latest["availability"] = (
        1 - latest["occupancy"]
    )

    latest["timestamp"] = (
        get_current_time()
    )

    return latest


def _arrival_probability(moment) -> float:
    """Time-of-day arrival bias: mornings fill, evenings empty (§6)."""

    hour = moment.hour + moment.minute / 60

    if 7 <= hour < 10:
        return 0.72
    if 10 <= hour < 14:
        return 0.58
    if 14 <= hour < 17:
        return 0.52
    if 17 <= hour < 21:
        return 0.34

    return 0.45


def advance_live_state(
    historical_df: pd.DataFrame,
    tick: int,
) -> pd.DataFrame:
    """Live tick keyed to the current time (§9–§10).

    Each refresh replays a few arrival/departure events with a
    time-of-day bias, so bays keep moving gradually and the forecast
    visibly leads or lags the current state (§11).
    """

    now = get_current_time()

    rng = np.random.default_rng(
        (RANDOM_SEED + tick * 7919) % (2 ** 32)
    )

    latest = get_latest_dataset_state(
        historical_df
    ).copy()

    arrival_p = _arrival_probability(now)

    # Demand waves: a few ticks of inflow surge, then outflow, then
    # calm — so availability visibly trends instead of hovering.
    phase = (tick // 6) % 3
    if phase == 1:
        arrival_p = 0.85
        emin, emax = 2, 5
    elif phase == 2:
        arrival_p = 0.15
        emin, emax = 2, 5
    else:
        emin, emax = 1, 4

    for zone in latest["zone"].unique():
        mask = latest["zone"] == zone
        idx = latest.index[mask].to_numpy()

        occupied_idx = idx[latest.loc[idx, "occupancy"].astype(int) == 1]
        free_idx = idx[latest.loc[idx, "occupancy"].astype(int) == 0]

        events = int(rng.integers(emin, emax))

        zone_rate = (
            int(latest.loc[idx, "occupancy"].sum()) / len(idx)
            if len(idx) else 0
        )

        arrivals = 0
        departures = 0

        for _ in range(events):
            # Turnover pressure: a packed zone only empties,
            # an empty zone only fills — keeps the map alive (§7).
            if zone_rate >= 0.95:
                do_arrival = False
            elif zone_rate <= 0.05:
                do_arrival = True
            else:
                do_arrival = rng.random() < arrival_p

            if do_arrival and len(free_idx):
                pick = rng.choice(free_idx)
                latest.at[pick, "occupancy"] = 1
                free_idx = free_idx[free_idx != pick]
                occupied_idx = np.append(occupied_idx, pick)
                arrivals += 1
            elif len(occupied_idx):
                pick = rng.choice(occupied_idx)
                latest.at[pick, "occupancy"] = 0
                occupied_idx = occupied_idx[occupied_idx != pick]
                free_idx = np.append(free_idx, pick)
                departures += 1

        latest.loc[mask, "entries"] = arrivals
        latest.loc[mask, "exits"] = departures

    # Sensor health breathes: rare dropouts and recoveries so the
    # Sensors page and health indicators move with the live tick.
    health = latest["occupancy_sensor_health"].astype(int).to_numpy()
    drop = rng.random(len(health)) < 0.004
    recover = (rng.random(len(health)) < 0.05) & (health == 0)
    health[drop] = 0
    health[recover] = 1
    latest["occupancy_sensor_health"] = health

    latest["availability"] = 1 - latest["occupancy"]

    latest["timestamp"] = now

    return latest