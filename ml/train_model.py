from __future__ import annotations

import os

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/parking_dataset.csv"
MODEL_PATH = "models/parking_model.pkl"


FEATURES = [
    "occupancy",
    "availability",
    "entries",
    "exits",
    "estimated_vehicle_count",
    "rain_intensity",
    "hour",
    "day_of_week",
    "is_weekend",
    "occ_trend",
    "entries_trend",
    "exits_trend",
]


TARGET = "future_availability"


def train():

    print("Loading ParkSmart dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    # Zone-level truth: occupancy sums over bays, flow sensors take the
    # zone value (identical across a zone's bays), target = zone future.
    zone = (
        df
        .groupby(["timestamp", "zone"])
        .agg(
            occupancy=("occupancy", "sum"),
            availability=("availability", "sum"),
            entries=("entries", "first"),
            exits=("exits", "first"),
            estimated_vehicle_count=("estimated_vehicle_count", "first"),
            rain_intensity=("rain_intensity", "max"),
            cumulative_entries=("cumulative_entries", "first"),
            cumulative_exits=("cumulative_exits", "first"),
            future_availability=("future_availability", "first"),
        )
        .reset_index()
        .sort_values(["zone", "timestamp"])
    )

    # Time-of-week context engineered from the timestamp (§14).
    timestamps = pd.to_datetime(zone["timestamp"])
    zone["hour"] = timestamps.dt.hour + timestamps.dt.minute / 60
    zone["day_of_week"] = timestamps.dt.weekday
    zone["is_weekend"] = (timestamps.dt.weekday >= 5).astype(int)

    # Recent-trend context: rolling 4-interval means per zone (§14).
    zone["occ_trend"] = (
        zone.groupby("zone")["occupancy"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    zone["entries_trend"] = (
        zone.groupby("zone")["entries"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    zone["exits_trend"] = (
        zone.groupby("zone")["exits"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )

    zone = zone.dropna(
        subset=FEATURES + [TARGET]
    )

    X = zone[FEATURES]
    y = zone[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    model_data = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    os.makedirs(
        "models",
        exist_ok=True,
    )

    joblib.dump(
        model_data,
        MODEL_PATH,
    )

    print("\nModel training completed.")

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"\nModel saved to {MODEL_PATH}"
    )


if __name__ == "__main__":
    train()