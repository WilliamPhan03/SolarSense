
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
import pandas as pd
import os
from datetime import datetime, timedelta

SAVE = 'data/processed/goes_training_clean.csv'

def fetch_last_week(save_path=SAVE):
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    print(f"Fetching training data from {start} to {end}")

    results = Fido.search(a.Time(start, end), a.Instrument('xrs'))
    files = Fido.fetch(results)
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
    df = df.resample('1T').mean().dropna().reset_index()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved training data to {save_path}")

if __name__ == "__main__":
    fetch_last_week()
