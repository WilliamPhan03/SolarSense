import requests
import pandas as pd
import os

SAVE = "data/processed/goes_7day_clean.csv"
URL  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

# This script fetches GOES XRS data and saves to a CSV file.
# The data is for the last 7 days, fetched from NOAA SWPC.
# You can change the SAVE path to save the output CSV file to a different location
# with a different name depending on the date/timeframe being fetched.

# This only grabs the last 7 days of data from NOAA SWPC. In UTC time.
# For greater than 7 days, use the fetch_90d_flux.py script or SunPy Fido option seen in fetch_x_time.py.

def fetch_training_data(save_path: str = SAVE) -> None:
    print("Fetching 7-day GOES X-ray data from NOAA SWPC …")

    resp = requests.get(URL, timeout=20)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: could not fetch data.")
        return

    raw = resp.json()
    if not raw:
        print("NOAA returned an empty payload.")
        return

    df = pd.DataFrame(raw)

    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(
        columns={"flux": "short_flux"}
    )
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(
        columns={"flux": "long_flux"}
    )

    merged = (
        pd.merge(long_df, short_df, on="time_tag", how="inner")
          .sort_values("time_tag")
    )

    merged["time_tag"] = (
        pd.to_datetime(merged["time_tag"], utc=True)
          .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    merged = merged.rename(columns={"time_tag": "timestamp"})
    merged = merged[["timestamp", "long_flux", "short_flux"]]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(merged)} rows; "
          f"first = {merged['timestamp'].iloc[0]}, "
          f"last = {merged['timestamp'].iloc[-1]})")

if __name__ == "__main__":
    fetch_training_data()
