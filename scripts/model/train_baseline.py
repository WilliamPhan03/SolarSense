# scripts/model/train_baseline.py
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------- small helpers (inline) ---------------------------

def load_flux_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def make_supervised(df: pd.DataFrame,
                    window: int,
                    horizon: int,
                    in_cols=("long_flux", "short_flux")):
    """
    Build (X, y) with:
      X: past `window` minutes (log10 of inputs)
      y: next `horizon` minutes of log10(long_flux)
    """
    vals = df[list(in_cols)].to_numpy(dtype=float)
    target = df["long_flux"].to_numpy(dtype=float)

    log_vals = np.log10(np.maximum(vals, 1e-12))
    log_target = np.log10(np.maximum(target, 1e-12))

    n = len(df)
    Xs, ys = [], []
    last_start = n - window - horizon
    for start in range(last_start):
        end_hist = start + window
        end_all = end_hist + horizon
        Xs.append(log_vals[start:end_hist, :].reshape(-1))
        ys.append(log_target[end_hist:end_all])
    return np.asarray(Xs), np.asarray(ys)

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  
    rmse = np.sqrt(mse)
    return {"MAE": mae, "RMSE": rmse}

# ------------------------------ config -------------------------------------

WINDOW  = 360     # minutes of history (6h)
HORIZON = 1440    # 24h forecast
IN_COLS = ("long_flux", "short_flux")

MODEL_DIR  = Path("models/regressors")
MODEL_PATH = MODEL_DIR / "flux_mlp.pkl"
XS_PATH    = MODEL_DIR / "x_scaler.pkl"

# ---------------------------------------------------------------------------

def main(csv_path: str):
    df = load_flux_csv(csv_path)

    X, y = make_supervised(df, WINDOW, HORIZON, IN_COLS)
    if len(X) < 2:
        raise RuntimeError("Not enough samples to train. Reduce WINDOW/HORIZON or use more data.")

    n = len(X)
    n_val = max(int(0.1 * n), 1)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)

    model = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train_s, y_train)

    y_val_pred = model.predict(X_val_s)
    m = regression_metrics(y_val, y_val_pred)
    print("Validation (log10 space):", m)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(x_scaler, XS_PATH)
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {XS_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/goes_7day_clean.csv")
    args = ap.parse_args()
    main(args.csv)
