import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_1step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_1step.pkl"
MODEL_PATH= MODEL_DIR / "flux_hgbr_1step.pkl"

WINDOW  = 720
HORIZON = 1440  # 1440 mins = 24 hours

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

def load_clean(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    for c in ("long_flux", "short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = (df.set_index("timestamp")
            .asfreq("1min")
            .interpolate(method="time")
            .ffill()
            .bfill()
            .reset_index())
    return df

def main(csv_path, out_dir):
    df = load_clean(csv_path)
    if len(df) < WINDOW:
        raise RuntimeError(f"Need at least {WINDOW} minutes of data")

    model = joblib.load(MODEL_PATH)
    x_scaler = joblib.load(XSC_PATH)
    y_scaler = joblib.load(YS_PATH)

    buf = df[["long_flux", "short_flux"]].iloc[-WINDOW:].to_numpy(dtype=float)
    buf_log = np.log10(np.maximum(buf, 1e-12))

    preds = []
    ts_start = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    for _ in range(HORIZON):
        X = buf_log.reshape(1, -1)
        Xs = x_scaler.transform(X)
        y_log_scaled = model.predict(Xs).reshape(-1, 1)
        y_log = y_scaler.inverse_transform(y_log_scaled).ravel()[0]
        y_lin = 10**y_log

        preds.append(y_lin)

        next_row = np.array([[y_lin, buf_log[-1, 0]]])
        buf_log = np.vstack([buf_log[1:], np.log10(np.maximum(next_row, 1e-12))])

    idx = pd.date_range(start=ts_start, periods=HORIZON, freq="1min")
    forecast_minute = pd.DataFrame({
        "timestamp": idx,
        "long_flux_pred": preds
    })

    # Resample by the hour
    forecast_minute.set_index("timestamp", inplace=True)
    forecast_hourly = forecast_minute.resample("1H").mean().dropna().reset_index()

    # Assign GOES class to each hourly average
    forecast_hourly["goes_class_pred"] = forecast_hourly["long_flux_pred"].apply(flux_to_class)

    # Output
    start_date = forecast_hourly["timestamp"].iloc[0].strftime("%Y_%m_%d")
    end_date = forecast_hourly["timestamp"].iloc[-1].strftime("%Y_%m_%d")
    out_csv = Path(out_dir) / f"forecast_hourly_{start_date}-{end_date}.csv"

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    forecast_hourly.to_csv(out_csv, index=False)
    print(f"✅ Saved hourly forecast -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/goes_90d_clean.csv")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    main(args.csv, args.out_dir)
