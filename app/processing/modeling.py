from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_et0_models(input_path: str | Path, output_dir: str | Path) -> dict:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        import joblib
        import xgboost as xgb
        from tensorflow.keras.layers import Dense, LSTM
        from tensorflow.keras.models import Sequential
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Model dependencies are missing: {exc}") from exc

    df = pd.read_excel(input_path)
    df = df.dropna(subset=["et0mm"]).copy()

    feature_candidates = ["tmaxc", "tminc", "rain_mm", "ra_mj_m2_gun", "enlem", "boylam", "rakim", "doy"]
    feature_cols = [col for col in feature_candidates if col in df.columns]

    if not feature_cols:
        raise ValueError("No valid feature columns found for training.")

    X = df[feature_cols]
    y = df["et0mm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)
    ann_pred = ann_model.predict(X_test_scaled)

    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    lstm_model = Sequential([
        LSTM(64, activation="relu", input_shape=(1, X_train_scaled.shape[1])),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, verbose=0)
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    metrics = {
        "Random Forest": {"rmse": np.sqrt(mean_squared_error(y_test, rf_pred)), "r2": r2_score(y_test, rf_pred)},
        "XGBoost": {"rmse": np.sqrt(mean_squared_error(y_test, xgb_pred)), "r2": r2_score(y_test, xgb_pred)},
        "ANN": {"rmse": np.sqrt(mean_squared_error(y_test, ann_pred)), "r2": r2_score(y_test, ann_pred)},
        "LSTM": {"rmse": np.sqrt(mean_squared_error(y_test, lstm_pred)), "r2": r2_score(y_test, lstm_pred)},
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(rf_model, output_dir / "rf_model.pkl")
    joblib.dump(xgb_model, output_dir / "xgb_model.pkl")
    joblib.dump(ann_model, output_dir / "ann_model.pkl")
    lstm_model.save(output_dir / "lstm_model.keras")

    for name, values in metrics.items():
        logger.info("%s: RMSE=%.4f, R²=%.4f", name, values["rmse"], values["r2"])

    return metrics
