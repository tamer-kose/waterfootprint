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


def clean_station_metadata(reference_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    ref = validate_station_metadata(reference_df)
    selected = validate_station_metadata(selected_df)

    ref_key = ref[["station_no", "enlem", "boylam", "rakim"]].drop_duplicates(subset="station_no")
    selected_key = selected[["station_no", "enlem", "boylam", "rakim"]].drop_duplicates(subset="station_no")

    comparison = selected_key.merge(
        ref_key,
        on="station_no",
        how="left",
        suffixes=("_selected", "_ref"),
        validate="one_to_one",
    )

    comparison["enlem_match"] = comparison["enlem_selected"].round(6).eq(comparison["enlem_ref"].round(6))
    comparison["boylam_match"] = comparison["boylam_selected"].round(6).eq(comparison["boylam_ref"].round(6))

    mismatches = comparison[~(comparison["enlem_match"] & comparison["boylam_match"])]
    logger.info("Coordinate mismatches: %s", len(mismatches))

    station_meta = selected.copy()
    station_meta = station_meta.merge(
        ref[["station_no", "enlem", "boylam", "rakim"]],
        on="station_no",
        how="left",
        suffixes=("_selected", "_ref"),
    )

    for col in ["enlem", "boylam", "rakim"]:
        selected_col = station_meta.get(f"{col}_selected", pd.Series(pd.NA, index=station_meta.index))
        ref_col = station_meta.get(f"{col}_ref", pd.Series(pd.NA, index=station_meta.index))
        station_meta[col] = selected_col.combine_first(ref_col)

    station_meta = station_meta[
        [c for c in station_meta.columns if not c.endswith("_selected") and not c.endswith("_ref")]
    ].drop_duplicates(subset="station_no").reset_index(drop=True)

    return station_meta
