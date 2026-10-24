import os
import time
import argparse
import pandas as pd
import requests


def load_raw_stream_data(filepath: str = "data/raw/green_tripdata_2023-03.parquet") -> pd.DataFrame:
    """Load and perform baseline cleaning on live simulation data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Simulation data file not found at: {filepath}")

    df = pd.read_parquet(filepath)

    pickup_col = "lpep_pickup_datetime" if "lpep_pickup_datetime" in df.columns else "tpep_pickup_datetime"
    dropoff_col = "lpep_dropoff_datetime" if "lpep_dropoff_datetime" in df.columns else "tpep_dropoff_datetime"

    df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    if "trip_distance" in df.columns:
        df = df[(df.trip_distance > 0) & (df.trip_distance <= 50)].copy()

    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def stream_to_api(
    df: pd.DataFrame,
    endpoint: str = "http://localhost:8085/predict",
    num_records: int = 100,
    delay: float = 0.05,
):
    """Simulate real-time live traffic by sending HTTP POST requests to FastAPI serving."""
    print(f"[Streamer] Streaming {num_records} rides to {endpoint}...")
    success = 0
    errors = 0

    records = df.head(num_records).to_dict(orient="records")
    for i, ride in enumerate(records):
        payload = {
            "PULocationID": int(ride["PULocationID"]),
            "DOLocationID": int(ride["DOLocationID"]),
            "trip_distance": float(ride["trip_distance"]),
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=2.0)
            if resp.status_code == 200:
                success += 1
            else:
                errors += 1
        except Exception:
            errors += 1

        if delay > 0:
            time.sleep(delay)

    print(f"[Streamer] Streaming complete: {success} successful, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Real-time Traffic Simulator")
    parser.add_argument("--data_path", type=str, default="data/raw/green_tripdata_2023-03.parquet")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8085/predict")
    parser.add_argument("--records", type=int, default=50)
    args = parser.parse_args()

    df_stream = load_raw_stream_data(args.data_path)
    stream_to_api(df_stream, endpoint=args.endpoint, num_records=args.records)
