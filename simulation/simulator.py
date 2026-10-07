from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


TOTAL_SPACES = 100

ZONE_CAPACITY = {
    "Zone A": 40,
    "Zone B": 30,
    "Zone C": 30,
}

ZONE_BASE_DEMAND = {
    "Zone A": 0.78,
    "Zone B": 0.56,
    "Zone C": 0.38,
}


def rain_intensity(hour: int, rng: np.random.Generator) -> int:
    """
    Generate realistic rain intensity.

    0 = No Rain
    1 = Light
    2 = Moderate
    3 = Heavy
    """

    # Mumbai-like seasonal simplification.
    # Rain is more likely during afternoon/evening.
    if 11 <= hour <= 20:
        probability = rng.random()

        if probability < 0.72:
            return 0
        elif probability < 0.88:
            return 1
        elif probability < 0.96:
            return 2
        else:
            return 3

    probability = rng.random()

    if probability < 0.86:
        return 0
    elif probability < 0.95:
        return 1
    elif probability < 0.99:
        return 2
    else:
        return 3


def demand_multiplier(
    timestamp: datetime,
    zone: str,
    rain: int,
) -> float:
    """
    Calculate demand based on:
    - time of day
    - weekday/weekend
    - zone
    - rain
    """

    hour = timestamp.hour + timestamp.minute / 60

    # Base demand
    multiplier = ZONE_BASE_DEMAND[zone]

    # Morning increase
    if 7 <= hour < 10:
        multiplier += 0.16

    # Office-hour peak
    if 10 <= hour < 14:
        multiplier += 0.12

    # Afternoon
    if 14 <= hour < 17:
        multiplier += 0.08

    # Evening demand
    if 17 <= hour < 21:
        multiplier += 0.18

    # Overnight reduction
    if hour >= 22 or hour < 6:
        multiplier -= 0.25

    # Weekend behaviour
    if timestamp.weekday() >= 5:
        multiplier -= 0.08

        # Evening weekend activity
        if 17 <= hour < 22:
            multiplier += 0.08

    # Rain generally reduces incoming demand
    rain_effect = {
        0: 0.00,
        1: -0.04,
        2: -0.09,
        3: -0.16,
    }

    multiplier += rain_effect[rain]

    return float(np.clip(multiplier, 0.05, 1.20))


def generate_dataset(
    days: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    # Fixed starting point makes the dataset reproducible.
    start = datetime.now().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=days)

    timestamps = pd.date_range(
        start=start,
        periods=days * 96,
        freq="15min",
    )

    rows = []

    # Current occupancy for each zone.
    zone_occupancy = {
        zone: int(capacity * ZONE_BASE_DEMAND[zone])
        for zone, capacity in ZONE_CAPACITY.items()
    }

    cumulative_entries = {
        zone: 0 for zone in ZONE_CAPACITY
    }

    cumulative_exits = {
        zone: 0 for zone in ZONE_CAPACITY
    }

    for timestamp in timestamps:

        timestamp_dt = timestamp.to_pydatetime()

        rain = rain_intensity(
            timestamp_dt.hour,
            rng,
        )

        for zone, capacity in ZONE_CAPACITY.items():

            current_occupancy = zone_occupancy[zone]

            demand = demand_multiplier(
                timestamp_dt,
                zone,
                rain,
            )

            # Entry probability increases with demand.
            entry_lambda = max(
                0.2,
                demand * 2.8,
            )

            exit_lambda = max(
                0.2,
                (1.15 - demand) * 2.2,
            )

            entries = int(
                rng.poisson(entry_lambda)
            )

            exits = int(
                rng.poisson(exit_lambda)
            )

            # Avoid impossible flows.
            available_before = capacity - current_occupancy

            entries = min(
                entries,
                max(0, available_before),
            )

            exits = min(
                exits,
                current_occupancy,
            )

            # Small stochastic variation.
            if rng.random() < 0.08:
                entries += int(
                    rng.integers(0, 3)
                )

            if rng.random() < 0.08:
                exits += int(
                    rng.integers(0, 3)
                )

            entries = min(
                entries,
                max(0, capacity - current_occupancy),
            )

            exits = min(
                exits,
                current_occupancy,
            )

            new_occupancy = (
                current_occupancy
                + entries
                - exits
            )

            # Gradual transition constraint.
            maximum_change = 5

            change = new_occupancy - current_occupancy

            if change > maximum_change:
                new_occupancy = (
                    current_occupancy
                    + maximum_change
                )

            if change < -maximum_change:
                new_occupancy = (
                    current_occupancy
                    - maximum_change
                )

            new_occupancy = int(
                np.clip(
                    new_occupancy,
                    0,
                    capacity,
                )
            )

            actual_entries = max(
                0,
                new_occupancy
                - current_occupancy
                + exits,
            )

            actual_exits = max(
                0,
                current_occupancy
                + actual_entries
                - new_occupancy,
            )

            cumulative_entries[zone] += actual_entries
            cumulative_exits[zone] += actual_exits

            zone_occupancy[zone] = new_occupancy

            available = capacity - new_occupancy

            vehicle_count = int(
                np.clip(
                    new_occupancy
                    + rng.normal(0, 1.5),
                    0,
                    capacity + 5,
                )
            )

            # Relevant sensor values.
            occupancy_sensor_health = int(
                rng.random() > 0.015
            )

            distance_reading = float(
                np.clip(
                    100
                    - (new_occupancy / capacity) * 80
                    + rng.normal(0, 3),
                    10,
                    100,
                )
            )

            entry_sensor_value = actual_entries
            exit_sensor_value = actual_exits

            rows.append(
                {
                    "timestamp": timestamp_dt,
                    "parking_space_id": None,
                    "zone": zone,
                    "occupancy": new_occupancy,
                    "availability": available,
                    "entries": actual_entries,
                    "exits": actual_exits,
                    "cumulative_entries": cumulative_entries[zone],
                    "cumulative_exits": cumulative_exits[zone],
                    "estimated_vehicle_count": vehicle_count,
                    "rain_intensity": rain,
                    "distance_reading": distance_reading,
                    "entry_sensor_value": entry_sensor_value,
                    "exit_sensor_value": exit_sensor_value,
                    "occupancy_sensor_health": occupancy_sensor_health,
                }
            )

    zone_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Create future availability target.
    # 30 minutes = 2 x 15-minute intervals.
    # ------------------------------------------------------------------

    zone_df["future_availability"] = (
        zone_df
        .groupby("zone")["availability"]
        .shift(-2)
    )

    # Remove rows where future target does not exist.
    zone_df = zone_df.dropna(
        subset=["future_availability"]
    ).reset_index(drop=True)

    zone_df["future_availability"] = (
        zone_df["future_availability"]
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Expand zone-level information to individual parking spaces.
    # ------------------------------------------------------------------

    expanded_rows = []

    for _, row in zone_df.iterrows():

        zone = row["zone"]
        capacity = ZONE_CAPACITY[zone]

        occupied_count = int(row["occupancy"])

        occupied_spaces = set(
            rng.choice(
                np.arange(1, capacity + 1),
                size=occupied_count,
                replace=False,
            )
        )

        for space_number in range(1, capacity + 1):

            space_id = (
                f"{zone[-1]}-{space_number:02d}"
            )

            occupied = int(
                space_number in occupied_spaces
            )

            space_row = row.to_dict()

            space_row["parking_space_id"] = space_id
            space_row["occupancy"] = occupied
            space_row["availability"] = 1 - occupied

            expanded_rows.append(
                space_row
            )

    dataset = pd.DataFrame(expanded_rows)

    dataset = dataset.sort_values(
        [
            "timestamp",
            "zone",
            "parking_space_id",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Artificial anomalies
    # ------------------------------------------------------------------

    anomaly_count = max(
        20,
        int(len(dataset) * 0.0001),
    )

    anomaly_indices = rng.choice(
        dataset.index,
        size=anomaly_count,
        replace=False,
    )

    anomaly_records = []

    for index in anomaly_indices:

        row = dataset.loc[index]

        anomaly_type = rng.choice(
            [
                "entry",
                "exit",
                "vehicle",
                "distance",
            ]
        )

        if anomaly_type == "entry":

            observed = int(
                row["entries"] + rng.integers(20, 80)
            )

            dataset.loc[
                index,
                "entries"
            ] = observed

            metric = "Entry Sensor"

        elif anomaly_type == "exit":

            observed = int(
                row["exits"] + rng.integers(20, 80)
            )

            dataset.loc[
                index,
                "exits"
            ] = observed

            metric = "Exit Sensor"

        elif anomaly_type == "vehicle":

            observed = int(
                row["estimated_vehicle_count"]
                + rng.integers(40, 120)
            )

            dataset.loc[
                index,
                "estimated_vehicle_count"
            ] = observed

            metric = "Vehicle Count"

        else:

            observed = float(
                row["distance_reading"]
                + rng.choice(
                    [
                        -60,
                        -45,
                        45,
                        60,
                    ]
                )
            )

            dataset.loc[
                index,
                "distance_reading"
            ] = observed

            metric = "Distance Sensor"

        anomaly_records.append(
            {
                "timestamp": row["timestamp"],
                "zone": row["zone"],
                "parking_space_id": row[
                    "parking_space_id"
                ],
                "metric": metric,
                "observed_value": observed,
                "status": "Unresolved",
            }
        )

    anomalies = pd.DataFrame(
        anomaly_records
    )

    return dataset, anomalies


def save_dataset(
    dataset: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:

    os.makedirs(
        "data",
        exist_ok=True,
    )

    dataset.to_csv(
        "data/parking_dataset.csv",
        index=False,
    )

    anomalies.to_csv(
        "data/anomalies.csv",
        index=False,
    )


if __name__ == "__main__":

    print(
        "Generating ParkSmart historical dataset..."
    )

    dataset, anomalies = generate_dataset(
        days=30,
        seed=42,
    )

    save_dataset(
        dataset,
        anomalies,
    )

    print(
        f"Dataset generated: {len(dataset):,} records"
    )

    print(
        f"Anomalies injected: {len(anomalies):,}"
    )

    print(
        "Saved to data/parking_dataset.csv"
    )

    print(
        "Saved to data/anomalies.csv"
    )