from __future__ import annotations

import numpy as np
import pandas as pd


MONITORED_METRICS = {
    "entries": "Entry Sensor",
    "exits": "Exit Sensor",
    "estimated_vehicle_count": "Vehicle Count",
    "distance_reading": "Distance Sensor",
}


def calculate_z_scores(
    series: pd.Series,
) -> pd.Series:

    mean = series.mean()
    std = series.std()

    if std == 0 or pd.isna(std):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        (series - mean) / std
    )


def detect_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for column, metric_name in MONITORED_METRICS.items():

        if column not in df.columns:
            continue

        working = df[
            [
                "timestamp",
                "zone",
                "parking_space_id",
                column,
            ]
        ].copy()

        working["z_score"] = (
            calculate_z_scores(
                working[column]
            )
        )

        # One record per (timestamp, zone, metric): bays repeat the same
        # zone reading, so keep the strongest signal only (§33: no noise).
        abnormal = working[
            working["z_score"].abs() >= 3.5
        ].copy()

        if abnormal.empty:
            continue

        abnormal["abs_z"] = abnormal["z_score"].abs()

        abnormal = (
            abnormal
            .sort_values("abs_z", ascending=False)
            .drop_duplicates(
                subset=["timestamp", "zone"],
                keep="first",
            )
            .drop(columns="abs_z")
        )

        for _, row in abnormal.iterrows():

            z = float(
                abs(row["z_score"])
            )

            if z >= 5:
                severity = "Critical"
            elif z >= 4.2:
                severity = "Warning"
            else:
                severity = "Informational"

            observed = float(
                row[column]
            )

            mean = float(
                working[column].mean()
            )

            std = float(
                working[column].std()
            )

            expected_low = max(
                0.0,
                mean - 2 * std,
            )

            expected_high = (
                mean + 2 * std
            )

            records.append(
                {
                    "timestamp": row["timestamp"],
                    "zone": row["zone"],
                    "sensor": metric_name,
                    "metric": metric_name,
                    "observed_value": observed,
                    "expected_range": (
                        f"{expected_low:.1f}"
                        f" – "
                        f"{expected_high:.1f}"
                    ),
                    "z_score": round(
                        z,
                        2,
                    ),
                    "severity": severity,
                    "status": "Unresolved",
                }
            )

    if not records:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "zone",
                "sensor",
                "metric",
                "observed_value",
                "expected_range",
                "z_score",
                "severity",
                "status",
            ]
        )

    result = pd.DataFrame(
        records
    )

    return result.sort_values(
        "timestamp",
        ascending=False,
    ).reset_index(drop=True)


if __name__ == "__main__":

    df = pd.read_csv(
        "data/parking_dataset.csv",
        parse_dates=["timestamp"],
    )

    anomalies = detect_anomalies(
        df
    )

    anomalies.to_csv(
        "data/anomalies.csv",
        index=False,
    )

    print(
        f"Detected {len(anomalies)} anomalies."
    )