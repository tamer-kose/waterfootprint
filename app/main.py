from __future__ import annotations

import logging
from pathlib import Path

from app.config import MODEL_OUTPUT_DIR, RA_FILE, ET0_FILE
from app.data_store import DataStore
from app.io import load_excel, save_excel
from app.processing.climate import fill_rain_knn_by_blg, fill_temperature_by_station
from app.processing.et0 import calculate_ra
from app.processing.station import clean_station_metadata
from app.processing.modeling import train_et0_models

logger = logging.getLogger(__name__)


def build_store() -> DataStore:
    store = DataStore()
    store.register("reference", lambda: load_excel(Path("/content/drive/MyDrive/waterfootprint/raw_data/istasyon.xlsx") if Path("/content/drive/MyDrive/waterfootprint/raw_data/istasyon.xlsx").exists() else "data/raw/istasyon.xlsx"))
    store.register("selected", lambda: load_excel(Path("/content/drive/MyDrive/waterfootprint/raw_data/Secili_Istasyon_Verileri.xlsx") if Path("/content/drive/MyDrive/waterfootprint/raw_data/Secili_Istasyon_Verileri.xlsx").exists() else "data/raw/Secili_Istasyon_Verileri.xlsx"))
    store.register("climate", lambda: load_excel(Path("/content/drive/MyDrive/waterfootprint/raw_data/Climate_160blg.xlsx") if Path("/content/drive/MyDrive/waterfootprint/raw_data/Climate_160blg.xlsx").exists() else "data/raw/Climate_160blg.xlsx"))
    store.register("et0", lambda: load_excel(Path("/content/drive/MyDrive/waterfootprint/raw_data/ET0mm_F_160.xlsx") if Path("/content/drive/MyDrive/waterfootprint/raw_data/ET0mm_F_160.xlsx").exists() else "data/raw/ET0mm_F_160.xlsx"))
    return store


def main() -> None:
    logger.info("Starting waterfootprint pipeline")
    store = build_store()

    reference_df = store.get("reference")
    selected_df = store.get("selected")
    cleaned = clean_station_metadata(reference_df, selected_df)

    climate_df = store.get("climate")
    climate_df = fill_temperature_by_station(climate_df)
    climate_df = fill_rain_knn_by_blg(climate_df, k=3)
    save_excel(climate_df, MODEL_OUTPUT_DIR / "Climate_160blg_completed.xlsx")

    et0_df = store.get("et0")
    et0_df = calculate_ra(et0_df)
    save_excel(et0_df, MODEL_OUTPUT_DIR / "ET0mm_F_160_Ra_added.xlsx")

    if RA_FILE.exists():
        input_path = RA_FILE
    else:
        input_path = MODEL_OUTPUT_DIR / "ET0mm_F_160_Ra_added.xlsx"

    train_et0_models(input_path=input_path, output_dir=MODEL_OUTPUT_DIR)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
