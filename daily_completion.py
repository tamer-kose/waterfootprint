from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config import CLIMATE_FILE, OUTPUT_DIR
from utils import ensure_dir, load_excel

logger = logging.getLogger(__name__)


def build_calendar(start: str = '2023-01-01', end: str = '2023-12-31') -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq='D')
    calendar = pd.DataFrame({'tarih': dates})
    calendar['yyyy'] = calendar['tarih'].dt.year
    calendar['mm'] = calendar['tarih'].dt.month
    calendar['dd'] = calendar['tarih'].dt.day
    calendar['doy'] = calendar['tarih'].dt.dayofyear
    return calendar


def fill_temperature_by_station(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'tarih' in df.columns:
        df['tarih'] = pd.to_datetime(df['tarih'])
    df = df.sort_values(['station_no', 'tarih']).reset_index(drop=True)

    for col in ['tminc', 'tmaxc']:
        if col in df.columns:
            df[col] = df.groupby('station_no')[col].transform(
                lambda s: s.interpolate(method='linear', limit_direction='both')
            )
    return df


def fill_rain_knn_by_blg(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    df = df.copy()
    if 'tarih' in df.columns:
        df['tarih'] = pd.to_datetime(df['tarih'])
    df = df.sort_values(['station_no', 'tarih']).reset_index(drop=True)

    if 'rain_mm' not in df.columns or 'blg' not in df.columns:
        raise ValueError('rain_mm and blg are required columns for rain completion.')

    stations = df.drop_duplicates(subset='station_no')[['station_no', 'enlem', 'boylam']].set_index('station_no')
    missing_rows = df[df['rain_mm'].isna()].copy()

    for idx, row in missing_rows.iterrows():
        target_station = row['station_no']
        target_blg = row['blg']
        date = row['tarih']

        candidates = df[(df['tarih'] == date) & (df['rain_mm'].notna())].copy()
        if target_blg is not None and not pd.isna(target_blg):
            candidates = candidates[candidates['blg'] == target_blg]
        if candidates.empty:
            candidates = df[(df['tarih'] == date) & (df['rain_mm'].notna())].copy()
        if candidates.empty:
            continue

        target_coords = stations.loc[target_station].values.reshape(1, -1)
        candidate_coords = stations.loc[candidates['station_no']].values
        distances = cdist(target_coords, candidate_coords)[0]

        k_actual = min(k, len(candidates))
        nearest_idx = np.argsort(distances)[:k_actual]
        df.at[idx, 'rain_mm'] = np.median(candidates.iloc[nearest_idx]['rain_mm'])

    return df


def prepare_daily_climate(input_path: str | Path = CLIMATE_FILE,
                         output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    df = load_excel(input_path)
    df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')
    df = fill_temperature_by_station(df)
    df = fill_rain_knn_by_blg(df, k=3)

    output_file = output_dir / 'Climate_160blg_completed.xlsx'
    df.to_excel(output_file, index=False)
    logger.info('Saved completed climate dataset: %s', output_file)
    return df


if __name__ == '__main__':
    prepare_daily_climate()
