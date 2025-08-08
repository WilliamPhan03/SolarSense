import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from astropy.time import Time
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from parfive import Downloader

# Save path
SAVE_PATH = "data/processed/goes_1day_clean.csv"

# Retry config
MAX_ATTEMPTS = 5
SLEEP_BETWEEN_ATTEMPTS = 20
MAX_CONN = 1

# Classification thresholds
CLASS_THRESH = [
    (1e-4, "X"),
    (1e-5, "M"),
    (1e-6, "C"),
    (1e-7, "B"),
    (0.0,  "A"),
]

# Exact time window
START_TIME = datetime(2025, 7, 29, 4, 6, 0, tzinfo=timezone.utc)
END_TIME   = datetime(2025, 7, 30, 4, 5, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"

def fetch_with_retries(start_dt, end_dt) -> list[str]:
    print(f"Searching SunPy: {start_dt} → {end_dt}")
    res = Fido.search(a.Time(start_dt, end_dt), a.Instrument("XRS"))

    if len(res) == 0:
        print("  No results found.")
        return []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            dl = Downloader(max_conn=MAX_CONN, progress=False)
            files = Fido.fetch(res, downloader=dl, progress=False)
            return list(files)
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(SLEEP_BETWEEN_ATTEMPTS)
    return []

def to_clean_df(file_paths: list[str], start, end) -> pd.DataFrame:
    dfs = []
    for fp in file_paths:
        try:
            ts = TimeSeries(fp)
            df = ts.to_dataframe().reset_index()
            df = df.rename(columns={"index": "timestamp", "xrsb": "long_flux", "xrsa": "short_flux"})
            dfs.append(df[["timestamp", "long_flux", "short_flux"]])
        except Exception as e:
            print(f"  Failed to parse {fp}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("min")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    df = (
        df.groupby("timestamp", as_index=False)
          .mean(numeric_only=True)
          .sort_values("timestamp")
    )

    return df

def finalize(df: pd.DataFrame, out_path: str):
    if df.empty:
        print("No data to save.")
        return

    df["goes_class"] = df["long_flux"].apply(flux_to_class)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} ({len(df)} rows)")
    print(f"First = {df['timestamp'].iloc[0]}, Last = {df['timestamp'].iloc[-1]}")

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print(f"Fetching GOES XRS data from: {START_TIME} → {END_TIME}")
    files = fetch_with_retries(START_TIME, END_TIME)

    if not files:
        print("No files downloaded.")
        return

    df = to_clean_df(files, START_TIME, END_TIME)
    finalize(df, SAVE_PATH)

if __name__ == "__main__":
    main()
