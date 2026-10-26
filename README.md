# Waterfootprint pipeline

This repository contains a production-friendly, modular data pipeline for the water footprint / ET0 processing workflow.

## Structure

- `app/config.py` for environment-specific paths
- `app/io.py` for read/write helpers
- `app/data_store.py` for cached dataset access
- `app/processing/` for pipeline stages
- `app/main.py` for orchestration

## Storage strategy

- GitHub repo: code and scripts
- Google Drive: large raw data, processed files, and trained models
- Local fallback: `data/` and `output/` when Drive is not mounted

## Run

```bash
python main.py
```

## Notes

- If you are using Colab, place the raw Excel files inside:
  `/content/drive/MyDrive/waterfootprint/raw_data`
- If Drive is not mounted, the code falls back to local `data/raw` and `output` folders.
