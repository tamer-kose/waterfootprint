from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "station_no" not in df.columns:
        raise ValueError("station_no column is required.")

    df["station_no"] = pd.to_numeric(df["station_no"], errors="coerce")

    for col in ["enlem", "boylam", "rakim"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["station_no"].isna().any():
        raise ValueError("station_no contains missing values.")

    if df.duplicated(subset=["station_no"]).any():
        logger.warning("Duplicate station_no values detected.")

    return df
