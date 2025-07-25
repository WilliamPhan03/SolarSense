# scripts/model/predict_next_24h.py
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

WINDOW  = 360
HORIZON = 1440

MODEL_DIR  = Path("models/regressors")
MODEL_PATH = MODEL_DIR / "flux_mlp.pkl"
XS_PATH    = MODEL_DIR / "x_scaler.pkl"

CLASS_THRESH = [
    (1e-4, "X"),
    (1e-5, "M"),
    (1e-6, "C"),
    (1e-7, "B"),
    (0.0,  "A"),
]

def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"

def main(csv_path: str, out_csv: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    if len(df) < WINDOW:
        raise RuntimeError(f"Need at least {WINDOW} rows from {csv_path}")

    model  = joblib.load(MODEL_PATH)
    x_scaler = joblib.load(XS_PATH)

    X_last = df[["long_flux", "short_flux"]].iloc[-WINDOW:].to_numpy(dtype=float)
    X_last_log = np.log10(np.maximum(X_last, 1e-12))
    X_last_flat = X_last_log.reshape(1, -1)
    X_last_s = x_scaler.transform(X_last_flat)

    y_log_pred = model.predict(X_last_s)[0]        # shape (HORIZON,)
    y_pred = np.power(10.0, y_log_pred)            # back to linear

    start_ts = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
    idx = pd.date_range(start=start_ts, periods=HORIZON, freq="T")

    out = pd.DataFrame({
        "timestamp": idx,
        "long_flux_pred": y_pred,
        "goes_class_pred": [flux_to_class(x) for x in y_pred],
    })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved forecast -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/goes_24h_clean.csv")
    ap.add_argument("--out", default="data/processed/forecast_24h.csv")
    args = ap.parse_args()
    main(args.csv, args.out)
