from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_ra(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "enlem" not in df.columns or "doy" not in df.columns:
        raise ValueError("Need 'enlem' and 'doy' for Ra calculation.")

    df["enlem_rad"] = np.pi / 180 * df["enlem"]
    df["dr"] = 1 + 0.033 * np.cos(2 * np.pi * df["doy"] / 365)
    df["delta"] = 0.409 * np.sin((2 * np.pi * df["doy"] / 365) - 1.39)

    tan_term = -np.tan(df["enlem_rad"]) * np.tan(df["delta"])
    tan_term = np.clip(tan_term, -1.0, 1.0)
    df["omega_s"] = np.arccos(tan_term)

    gsc = 0.0820
    df["ra_mj_m2_gun"] = (24 * 60 / np.pi) * gsc * df["dr"] * (
        df["omega_s"] * np.sin(df["enlem_rad"]) * np.sin(df["delta"])
        + np.cos(df["enlem_rad"]) * np.cos(df["delta"]) * np.sin(df["omega_s"])
    )

    df["ra_mj_m2_gun"] = df["ra_mj_m2_gun"].round(2)
    return df
