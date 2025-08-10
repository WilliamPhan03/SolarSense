import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_1step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_1step.pkl"
MODEL_PATH= MODEL_DIR / "flux_hgbr_1step.pkl"

WINDOW = 720  # 12h

# This script retrains the model using the last 7 days of data. Useful for periodic updates.

def load_last_7d(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp").last("7D").asfreq("1min").interpolate("time").ffill().bfill().reset_index()
    for c in ("long_flux", "short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def make_supervised_1step(df, window):
    vals = df[["long_flux", "short_flux"]].to_numpy(float)
    log_vals = np.log10(np.maximum(vals, 1e-12))
    y = np.log10(np.maximum(df["long_flux"].shift(-1).to_numpy(float), 1e-12))

    X, Y = [], []
    for i in range(window, len(df) - 1):
        X.append(log_vals[i - window:i, :].reshape(-1))
        Y.append(y[i])
    return np.asarray(X), np.asarray(Y)

def main(csv):
    df = load_last_7d(csv)
    if len(df) <= WINDOW + 1:
        raise RuntimeError("Not enough data in the last 7 days to retrain.")

    X, y = make_supervised_1step(df, WINDOW)

    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y).ravel()

    model = HistGradientBoostingRegressor(
        max_depth=6, max_iter=500, learning_rate=0.05, random_state=42
    )
    model.fit(Xs, ys)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(x_scaler, XSC_PATH)
    joblib.dump(y_scaler, YS_PATH)
    print("Retrained and saved:", MODEL_PATH)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/goes_90d_clean.csv")
    args = ap.parse_args()
    main(args.csv)
