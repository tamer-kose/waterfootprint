from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GOOGLE_DRIVE_ROOT = Path("/content/drive/MyDrive/waterfootprint")

if GOOGLE_DRIVE_ROOT.exists():
    RAW_DATA_DIR = GOOGLE_DRIVE_ROOT / "raw_data"
    PROCESSED_DATA_DIR = GOOGLE_DRIVE_ROOT / "processed"
    MODEL_OUTPUT_DIR = GOOGLE_DRIVE_ROOT / "models"
else:
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

REFERENCE_FILE = RAW_DATA_DIR / "istasyon.xlsx"
SELECTED_FILE = RAW_DATA_DIR / "Secili_Istasyon_Verileri.xlsx"
CLIMATE_FILE = RAW_DATA_DIR / "Climate_160blg.xlsx"
ET0_FILE = RAW_DATA_DIR / "ET0mm_F_160.xlsx"
RA_FILE = RAW_DATA_DIR / "Climate_58400_Ra.xlsx"

DEFAULT_COLUMNS = {
    "station_no": ["istasyon_no", "station_no", "station_id"],
    "enlem": ["enlem", "latitude"],
    "boylam": ["boylam", "longitude"],
    "rakim": ["rakim", "elevation", "altitude"],
    "blg": ["blg"],
    "tarih": ["tarih", "date"],
    "tmaxc": ["tmaxc", "tmax"],
    "tminc": ["tminc", "tmin"],
    "rain_mm": ["rain_mm", "rain"],
    "et0mm": ["et0mm", "et0"],
    "doy": ["doy", "day_of_year"],
}
