import requests
import pandas as pd
import os

SAVE = "data/processed/goes_1day_clean.csv"
URL  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

CLASS_THRESH = [
    (1e-4, "X"),
    (1e-5, "M"),
    (1e-6, "C"),
    (1e-7, "B"),
    (0.0,  "A"),
]

def flux_to_class(f):
    
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"


def fetch_training_data(save_path: str = SAVE) -> None:
    print("Fetching 1-day GOES X-ray data from NOAA SWPC …")

    resp = requests.get(URL, timeout=20)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: could not fetch data.")
        return

    raw = resp.json()
    if not raw:
        print("NOAA returned an empty payload.")
        return

    df = pd.DataFrame(raw)

    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(
        columns={"flux": "long_flux"}
    )

    # Add the "goes_class" column
    long_df["goes_class"] = long_df["long_flux"].apply(flux_to_class)

    long_df["timestamp"] = (
        pd.to_datetime(long_df["time_tag"], utc=True)
          .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    final_df = long_df[["timestamp", "long_flux", "goes_class"]]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(final_df)} rows; "
          f"first = {final_df['timestamp'].iloc[0]}, "
          f"last = {final_df['timestamp'].iloc[-1]})")

if __name__ == "__main__":
    fetch_training_data()
