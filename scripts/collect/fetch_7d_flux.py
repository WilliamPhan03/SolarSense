import requests
import pandas as pd
import os

SAVE = 'data/processed/goes_7day_clean.csv'
URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

def fetch_7d_flux(save_path=SAVE):
    print("📡 Fetching 7-day GOES XRS data from NOAA SWPC...")
    r = requests.get(URL)
    if r.status_code != 200:
        print(f"Failed to fetch data. HTTP status: {r.status_code}")
        return

    data = r.json()
    if not data:
        print("No data returned from NOAA.")
        return

    df = pd.DataFrame(data)

    # Filter for both energy bands and rename
    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(columns={"flux": "short_flux"})
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(columns={"flux": "long_flux"})

    # Merge on time_tag
    merged = pd.merge(long_df, short_df, on='time_tag', how='inner')

    # Format timestamps
    merged['timestamp'] = pd.to_datetime(merged['time_tag'])
    merged = merged[['timestamp', 'long_flux', 'short_flux']]

    # Resample to hourly averages
    merged.set_index('timestamp', inplace=True)
    hourly_avg = merged.resample('1H').mean().dropna().reset_index()

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hourly_avg.to_csv(save_path, index=False)
    print(f"✅ Saved hourly-averaged flux data to {save_path}")

if __name__ == "__main__":
    fetch_7d_flux()
