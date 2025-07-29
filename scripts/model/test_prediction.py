import pandas as pd
from sklearn.metrics import accuracy_score
import os

def test_prediction_accuracy():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    forecast_file = os.path.join(project_root, 'data', 'processed', 'forecast_2025_07_28-2025_07_29.csv')
    actual_file = os.path.join(project_root, 'data', 'processed', 'goes_1day_clean.csv')

    # file checks
    if not os.path.exists(forecast_file):
        print(f"Error: Forecast file not found at {forecast_file}")
        return
    if not os.path.exists(actual_file):
        print(f"Error: Actual data file not found at {actual_file}")
        return

    # load csv's
    try:
        forecast_df = pd.read_csv(forecast_file)
        actual_df = pd.read_csv(actual_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # convert timestamps for merging
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'])

    merged_df = pd.merge(forecast_df, actual_df, on='timestamp', how='inner')

    # merge check
    if merged_df.empty:
        print("No matching timestamps found between forecast and actual data. Cannot calculate accuracy.")
        return

    y_true = merged_df['goes_class']
    y_pred = merged_df['goes_class_pred']

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Number of comparable predictions: {len(merged_df)}")
    print(f"Prediction Accuracy for 'goes_class': {accuracy:.2%}")

if __name__ == '__main__':
    test_prediction_accuracy()
