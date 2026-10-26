from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


class DataStore:
    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}
        self._loaders: Dict[str, Callable[[], pd.DataFrame]] = {}

    def register(self, key: str, loader: Callable[[], pd.DataFrame]) -> None:
        self._loaders[key] = loader

    def get(self, key: str, *, force_reload: bool = False) -> pd.DataFrame:
        if force_reload:
            self._cache.pop(key, None)

        if key not in self._cache:
            if key not in self._loaders:
                raise KeyError(f"No loader registered for key: {key}")
            self._cache[key] = self._loaders[key]()

        return self._cache[key]

    def get_view(self, key: str, *, deep: bool = False) -> pd.DataFrame:
        return self.get(key).copy(deep=deep)

    def clear(self) -> None:
        self._cache.clear()
