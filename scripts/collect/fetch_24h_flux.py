import requests
import pandas as pd
import os

SAVE = "data/processed/goes_24h_clean.csv"
URL  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

def fetch_live_flux(save_path: str = SAVE) -> None:
    print("Fetching live GOES XRS data from NOAA SWPC …")
    r = requests.get(URL, timeout=15)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: couldn’t fetch data.")
        return

    data = r.json()
    if not data:
        print("NOAA returned an empty payload.")
        return

    df = pd.DataFrame(data)

    # --- separate the two bands --------------------------------------------
    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(
        columns={"flux": "short_flux"}
    )
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(
        columns={"flux": "long_flux"}
    )

    # --- merge and format ---------------------------------------------------
    merged = (
        pd.merge(long_df, short_df, on="time_tag", how="inner")
          .sort_values("time_tag")
    )

    # convert ISO strings → nice timestamp
    merged["time_tag"] = (
        pd.to_datetime(merged["time_tag"], utc=True)
          .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    merged = merged.rename(columns={"time_tag": "timestamp"})
    merged = merged[["timestamp", "long_flux", "short_flux"]]

    # --- save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(merged)} rows)")

if __name__ == "__main__":
    fetch_live_flux()