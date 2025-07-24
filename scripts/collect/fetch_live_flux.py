from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

SAVE = 'data/live/goes_recent_flux.csv'

def fetch_last_24_hours(save_path=SAVE):
    # Use timezone-aware UTC
    # Known past dates with GOES XRS data

    # set end first
    end = datetime.now(timezone.utc)

    # then subtract 48 hours
    start = end - timedelta(hours=48)    
    print(f"Fetching recent flux from {start} to {end}")

    # Capitalize instrument name
    results = Fido.search(a.Time(start, end), a.Instrument('XRS'))

    if len(results) == 0:
        print("⚠️ No GOES XRS data found in that time range.")
        return

    files = Fido.fetch(results)
    if not files:
        print("⚠️ No files downloaded.")
        return

    ts = TimeSeries(files, concatenate=True)
    df = ts.to_dataframe().reset_index()

    df = df.rename(columns={
        'index': 'timestamp',
        'xrsa': 'long_flux',
        'xrsb': 'short_flux'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'long_flux', 'short_flux']]
    df.set_index('timestamp', inplace=True)
    # Use '1min' instead of '1T' to avoid future warnings
    df = df.resample('1min').mean().dropna().reset_index()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved live flux data to {save_path}")

if __name__ == "__main__":
    fetch_last_24_hours()
