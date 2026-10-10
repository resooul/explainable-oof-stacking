"""Synthetic data with the schema of the Kaggle NYC taxi training file."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_nyc_like(n: int = 30_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2016-01-01")
    pickup = start + pd.to_timedelta(np.sort(rng.uniform(0, 181 * 86400, n)), unit="s")
    plat, plon = rng.normal(40.75, 0.03, n), rng.normal(-73.98, 0.03, n)
    dlat, dlon = plat + rng.normal(0, 0.02, n), plon + rng.normal(0, 0.02, n)
    km = np.hypot((dlat - plat) * 111, (dlon - plon) * 84)
    hour = pickup.hour.to_numpy()
    rush = np.isin(hour, [8, 9, 17, 18]).astype(float)
    vendor = rng.integers(1, 3, n)
    seconds = 120 + km * (150 + 90 * rush) + 30 * (vendor == 2) + rng.gamma(2.0, 60, n)
    seconds = np.clip(seconds, 61, 17_000).round()
    return pd.DataFrame({
        "id": [f"id{i:07d}" for i in range(n)], "vendor_id": vendor,
        "pickup_datetime": pickup.strftime("%Y-%m-%d %H:%M:%S"),
        "dropoff_datetime": (pickup + pd.to_timedelta(seconds, unit="s")).strftime("%Y-%m-%d %H:%M:%S"),
        "passenger_count": rng.integers(1, 7, n), "pickup_longitude": plon, "pickup_latitude": plat,
        "dropoff_longitude": dlon, "dropoff_latitude": dlat,
        "store_and_fwd_flag": rng.choice(["N", "Y"], n, p=[0.99, 0.01]), "trip_duration": seconds.astype(int)})
