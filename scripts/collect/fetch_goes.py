from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
import pandas as pd
import os

START_TIME = '2022-01-01'
END_TIME = '2022-01-31'
SAVE_PATH = 'data/raw/goes_raw.csv'

def fetch_goes_xray(start=START_TIME, end=END_TIME, save_path=SAVE_PATH):
    print(f"Fetching GOES data from {start} to {end}...")

    time_range = a.Time(start, end)
    instrument = a.Instrument('xrs')

    results = Fido.search(time_range, instrument)
    downloaded_files = Fido.fetch(results)

    ts = TimeSeries(downloaded_files, concatenate=True)
    df = ts.to_dataframe().reset_index()

    df = df.rename(columns={
        'index': 'timestamp',
        'xrsa': 'long_flux',
        'xrsb': 'short_flux'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'long_flux', 'short_flux']]  

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved GOES data to {save_path}")

if __name__ == "__main__":
    fetch_goes_xray()
