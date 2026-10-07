from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import os

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/parking_model.pkl"


@lru_cache(maxsize=1)
def load_model():
    """Model file is static at runtime — load once, not per prediction."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Parking model not found. "
            "Run: python ml/train_model.py"
        )

    return joblib.load(
        MODEL_PATH
    )


def _time_features(moment: datetime | None = None) -> dict:
    """Time-of-week context for the model (§14). Defaults to now."""

    moment = moment or datetime.now()

    return {
        "hour": moment.hour + moment.minute / 60,
        "day_of_week": moment.weekday(),
        "is_weekend": int(moment.weekday() >= 5),
    }


def predict_availability(
    occupancy: int,
    availability: int,
    entries: int,
    exits: int,
    vehicle_count: int,
    rain_intensity: int,
    cumulative_entries: int,
    cumulative_exits: int,
    moment: datetime | None = None,
    occ_trend: float = 0.0,
    entries_trend: float = 0.0,
    exits_trend: float = 0.0,
) -> float:
    """Recalculate the 30-min forecast from the live state (§11).

    Builds the full feature row (flow + time context) and uses
    whichever features the trained model expects, so the predictor
    stays compatible across retrains.
    """

    model_data = load_model()

    model = model_data["model"]

    features = model_data["features"]

    row = {
        "occupancy": occupancy,
        "availability": availability,
        "entries": entries,
        "exits": exits,
        "estimated_vehicle_count": vehicle_count,
        "rain_intensity": rain_intensity,
        "cumulative_entries": cumulative_entries,
        "cumulative_exits": cumulative_exits,
    }
    row.update(_time_features(moment))
    row.update(
        {
            "occ_trend": occ_trend,
            "entries_trend": entries_trend,
            "exits_trend": exits_trend,
        }
    )

    values = pd.DataFrame([row])

    # Backward/forward compatible: use the intersection, fill gaps.
    for missing in [f for f in features if f not in values.columns]:
        values[missing] = 0

    values = values[features]

    prediction = model.predict(
        values
    )[0]

    return float(
        np.clip(
            prediction,
            0,
            100,
        )
    )


def predict_zone_availability(
    zone_current: dict,
    summary: dict,
    zone_capacity: int,
) -> float:
    """Zone-level 30-min forecast, clipped to the zone capacity."""

    prediction = predict_availability(
        occupancy=zone_current["occupied"],
        availability=zone_current["available"],
        entries=zone_current["entries"],
        exits=zone_current["exits"],
        vehicle_count=zone_current["vehicle_count"],
        rain_intensity=zone_current["rain_intensity"],
        cumulative_entries=zone_current.get(
            "cumulative_entries", summary["entries"]
        ),
        cumulative_exits=zone_current.get(
            "cumulative_exits", summary["exits"]
        ),
        occ_trend=zone_current.get("occ_trend", 0.0),
        entries_trend=zone_current.get("entries_trend", 0.0),
        exits_trend=zone_current.get("exits_trend", 0.0),
    )

    return float(
        np.clip(prediction, 0, zone_capacity)
    )


def predict_facility_availability(
    zones: pd.DataFrame,
    rain_intensity: int,
    trends_map: dict | None = None,
) -> float:
    """Facility forecast = sum of zone forecasts (model is zone-scale)."""

    total = 0.0
    trends_map = trends_map or {}

    for _, z in zones.iterrows():
        zt = trends_map.get(z["zone"], {})
        total += predict_zone_availability(
            {
                "occupied": int(z["occupied"]),
                "available": int(z["available"]),
                "entries": int(z["entries"]),
                "exits": int(z["exits"]),
                "vehicle_count": int(z["vehicle_count"]),
                "rain_intensity": int(rain_intensity),
                "cumulative_entries": int(z.get("cumulative_entries", 0)),
                "cumulative_exits": int(z.get("cumulative_exits", 0)),
                "occ_trend": float(zt.get("occ_trend", 0.0)),
                "entries_trend": float(zt.get("entries_trend", 0.0)),
                "exits_trend": float(zt.get("exits_trend", 0.0)),
            },
            {"entries": 0, "exits": 0},
            int(z["capacity"]),
        )

    return float(total)


def model_metrics():

    model_data = load_model()

    return {
        "r2": model_data["r2"],
        "mae": model_data["mae"],
        "rmse": model_data["rmse"],
        "features": model_data["features"],
        "target": model_data["target"],
    }
