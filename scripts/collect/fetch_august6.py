import requests
import pandas as pd
import os

SAVE = "data/processed/goes_august7_clean.csv"
URL  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


# Classification thresholds for GOES long flux
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

    merged = merged[
        (merged["timestamp"] >= "2025-08-07 00:00:00") &
        (merged["timestamp"] <= "2025-08-07 23:59:59")
    ]

    merged["goes_class"] = merged["long_flux"].apply(flux_to_class)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(merged)} rows; "
          f"first = {merged['timestamp'].iloc[0]}, "
          f"last = {merged['timestamp'].iloc[-1]})")

if __name__ == "__main__":
    fetch_training_data()
