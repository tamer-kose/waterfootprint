from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def fill_temperature_by_station(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tarih" in df.columns:
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.sort_values(["station_no", "tarih"]).reset_index(drop=True)

    for col in ["tminc", "tmaxc"]:
        if col in df.columns:
            df[col] = df.groupby("station_no")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
    return df


def fill_rain_knn_by_blg(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    df = df.copy()
    if "tarih" in df.columns:
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.sort_values(["station_no", "tarih"]).reset_index(drop=True)

    if "rain_mm" not in df.columns or "blg" not in df.columns:
        raise ValueError("rain_mm and blg are required columns for rain completion.")

    stations = df.drop_duplicates(subset="station_no")[["station_no", "enlem", "boylam"]].set_index("station_no")
    missing_rows = df[df["rain_mm"].isna()].copy()

    for idx, row in missing_rows.iterrows():
        target_station = row["station_no"]
        target_blg = row["blg"]
        target_date = row["tarih"]

        candidates = df[(df["tarih"] == target_date) & (df["rain_mm"].notna())].copy()
        if target_blg is not None and not pd.isna(target_blg):
            candidates = candidates[candidates["blg"] == target_blg]

        if candidates.empty:
            candidates = df[(df["tarih"] == target_date) & (df["rain_mm"].notna())].copy()
        if candidates.empty:
            continue

        target_coords = stations.loc[target_station].values.reshape(1, -1)
        candidate_coords = stations.loc[candidates["station_no"]].values
        distances = cdist(target_coords, candidate_coords)[0]

        k_actual = min(k, len(candidates))
        nearest = np.argsort(distances)[:k_actual]
        df.at[idx, "rain_mm"] = np.median(candidates.iloc[nearest]["rain_mm"])

    return df
