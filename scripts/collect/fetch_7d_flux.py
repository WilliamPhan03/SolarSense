#!/usr/bin/env python3
"""
Download the most-recent 7 days of GOES X-ray data (1-min cadence)
and save as:  data/processed/goes_training_clean.csv

Columns:
    timestamp, long_flux, short_flux
"""

import requests
import pandas as pd
import os
from datetime import datetime

URL_7DAY  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
SAVE_PATH = "data/processed/goes_7day_clean.csv"

def fetch_training_data(save_path: str = SAVE_PATH) -> None:
    print("Fetching 7-day GOES X-ray data from NOAA SWPC …")

    resp = requests.get(URL_7DAY, timeout=20)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: could not fetch data.")
        return

    raw = resp.json()
    if not raw:
        print("NOAA returned an empty payload.")
        return

    df = pd.DataFrame(raw)

    # --- separate the two energy bands -------------------------------------
    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(
        columns={"flux": "short_flux"}
    )
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(
        columns={"flux": "long_flux"}
    )

    # --- merge and tidy -----------------------------------------------------
    merged = (
        pd.merge(long_df, short_df, on="time_tag", how="inner")
          .sort_values("time_tag")
    )

    # convert ISO strings → “YYYY-MM-DD HH:MM:SS”
    merged["time_tag"] = (
        pd.to_datetime(merged["time_tag"], utc=True)
          .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    merged = merged.rename(columns={"time_tag": "timestamp"})
    merged = merged[["timestamp", "long_flux", "short_flux"]]

    # --- save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(merged)} rows; "
          f"first = {merged['timestamp'].iloc[0]}, "
          f"last = {merged['timestamp'].iloc[-1]})")

if __name__ == "__main__":
    fetch_training_data()