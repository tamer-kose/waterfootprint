from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_COLUMNS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        for col in df.columns
    ]

    for canonical, aliases in DEFAULT_COLUMNS.items():
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: canonical}, inplace=True)

    return df


def load_excel(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_excel(p)
    return normalize_columns(df)


def save_excel(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(p, index=False)
