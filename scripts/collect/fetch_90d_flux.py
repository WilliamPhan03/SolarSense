import os
import time
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from parfive import Downloader

OUT = "data/processed/goes_90d_clean.csv"

# polite settings for the historical (SunPy) part
MAX_ATTEMPTS = 5
SLEEP_BETWEEN_ATTEMPTS = 30  # seconds
CHUNK_DAYS = 7
MAX_CONN = 1  # sequential

SWPC_7DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fetch_chunk(t0_iso: str, t1_iso: str) -> list[str]:
    """Fetch one [t0, t1) chunk with manual retries. Returns list of paths (can be empty)."""
    res = Fido.search(a.Time(t0_iso, t1_iso), a.Instrument("XRS"))
    if len(res) == 0:
        return []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            dl = Downloader(max_conn=MAX_CONN, progress=False)
            files = Fido.fetch(res, downloader=dl, progress=False)
            return list(files)
        except Exception as e:
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed for {t0_iso} → {t1_iso}: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(SLEEP_BETWEEN_ATTEMPTS)
    return []

def to_clean_df_from_timeseries(file_paths: list[str]) -> pd.DataFrame:
    """Read SunPy TimeSeries files and return a tidy 1-min df."""
    dfs = []
    for fp in file_paths:
        try:
            ts = TimeSeries(fp)
            df = ts.to_dataframe().reset_index()
            df = df.rename(columns={"index": "timestamp", "xrsa": "short_flux", "xrsb": "long_flux"})
            dfs.append(df[["timestamp", "long_flux", "short_flux"]])
        except Exception as e:
            print(f"  failed to parse {fp}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("min")
    # average overlapping minutes (e.g., multiple satellites)
    df = (
        df.groupby("timestamp", as_index=False)
          .mean(numeric_only=True)
          .sort_values("timestamp")
    )
    return df

def fetch_7day_swpc_json() -> pd.DataFrame:
    """Fetch the fresh 7-day JSON and return tidy 1-min df matching the same schema."""
    r = requests.get(SWPC_7DAY_URL, timeout=20)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        raise RuntimeError("SWPC 7-day JSON returned empty payload")

    df = pd.DataFrame(raw)
    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(columns={"flux": "short_flux"})
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(columns={"flux": "long_flux"})

    merged = pd.merge(long_df, short_df, on="time_tag", how="inner").sort_values("time_tag")
    merged["timestamp"] = pd.to_datetime(merged["time_tag"], utc=True).dt.floor("min")
    out = merged[["timestamp", "long_flux", "short_flux"]].copy()
    return out

def finalize_and_save(df: pd.DataFrame, out_path: str):
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows; first={df['timestamp'].iloc[0]}, last={df['timestamp'].iloc[-1]})")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(out_path: str):
    now = Time.now()
    start_90 = now - timedelta(days=90)
    split_recent = now - timedelta(days=7)

    print(f"Historical (SunPy/Fido): {start_90.iso} → {split_recent.iso}")
    print(f"Recent (SWPC JSON):      {split_recent.iso} → {now.iso}")

    # ---- 1) Historical 83 days via SunPy ----
    hist_files = []
    t0 = start_90
    while t0 < split_recent:
        t1 = min(t0 + timedelta(days=CHUNK_DAYS), split_recent)
        print(f"  chunk: {t0.iso} → {t1.iso}")
        files = fetch_chunk(t0.iso, t1.iso)
        hist_files.extend(files)
        t0 = t1

    df_hist = to_clean_df_from_timeseries(hist_files)

    # ---- 2) Recent 7 days via SWPC JSON ----
    df_recent = fetch_7day_swpc_json()

    # ---- 3) Concat and keep the most recent values on overlap ----
    if not df_hist.empty:
        df = pd.concat([df_hist, df_recent], axis=0, ignore_index=True)
    else:
        df = df_recent.copy()

    finalize_and_save(df, out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    main(args.out)
