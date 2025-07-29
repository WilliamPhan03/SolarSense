import pandas as pd

def process_24h_long_flux(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Correct the timestamp format
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

# Example usage
file_path = "/Users/williamphan/Downloads/School/Summer2025/CMPT310/SolarSense/data/processed/goes_24h_long_flux.csv"
process_24h_long_flux(file_path)
