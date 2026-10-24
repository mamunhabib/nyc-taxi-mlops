import os
import glob
import time
import signal
import argparse
import itertools
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


def get_stream_files(data_path: str) -> list:
    """Resolve data files from a file path or directory glob."""
    if os.path.isfile(data_path):
        return [data_path]
    elif os.path.isdir(data_path):
        # Look for Q2 2023 (or any monthly) parquet files
        q2_files = sorted(glob.glob(os.path.join(data_path, "green_tripdata_2023-0[456].parquet")))
        if q2_files:
            return q2_files
        all_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        if all_files:
            return all_files
    raise FileNotFoundError(f"No valid parquet stream files found at: {data_path}")


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


def stream_continuous(
    data_files: list,
    endpoint: str = "http://localhost:8085/predict",
    delay: float = 0.5,
    sample_per_month: int = 1000,
):
    """Continuously stream rides across multiple monthly datasets in a rolling loop."""
    print(f"[Streamer] Starting continuous traffic generator to endpoint: {endpoint}")
    print(f"[Streamer] Delay per request: {delay}s | Datasets: {[os.path.basename(f) for f in data_files]}")

    running = True

    def handle_shutdown(signum, frame):
        nonlocal running
        print(f"\n[Streamer] Received termination signal ({signum}). Gracefully stopping traffic generator...")
        running = False

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    total_requests = 0
    total_success = 0
    total_errors = 0
    start_time = time.time()

    # Infinite cyclic generator over monthly datasets
    for file_cycle, filepath in enumerate(itertools.cycle(data_files), start=1):
        if not running:
            break

        filename = os.path.basename(filepath)
        print(f"\n[Streamer] [Cycle #{file_cycle}] Loading batch from {filename}...")
        try:
            df = load_raw_stream_data(filepath)
            if sample_per_month and len(df) > sample_per_month:
                df = df.sample(n=sample_per_month, random_state=42 + file_cycle).reset_index(drop=True)
            records = df[["PULocationID", "DOLocationID", "trip_distance"]].to_dict(orient="records")
        except Exception as e:
            print(f"[Streamer] Error loading {filepath}: {e}. Retrying next file in 3s...")
            time.sleep(3)
            continue

        for i, ride in enumerate(records):
            if not running:
                break

            payload = {
                "PULocationID": int(ride["PULocationID"]),
                "DOLocationID": int(ride["DOLocationID"]),
                "trip_distance": float(ride["trip_distance"]),
            }

            try:
                resp = requests.post(endpoint, json=payload, timeout=3.0)
                if resp.status_code == 200:
                    total_success += 1
                else:
                    total_errors += 1
            except Exception:
                total_errors += 1

            total_requests += 1

            # Log periodic status report
            if total_requests % 50 == 0:
                elapsed = time.time() - start_time
                rps = total_requests / elapsed if elapsed > 0 else 0
                print(
                    f"[Streamer Telemetry] Sent: {total_requests} | "
                    f"Success: {total_success} | Errors: {total_errors} | "
                    f"Throughput: {rps:.1f} req/s | Current Source: {filename} ({i + 1}/{len(records)})"
                )

            if delay > 0:
                time.sleep(delay)

    print(
        f"\n[Streamer] Generator stopped. Total sent: {total_requests} "
        f"(Success: {total_success}, Errors: {total_errors})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Real-time Traffic Simulator")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw",
        help="Path to single parquet file or directory containing monthly parquet files",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("SERVING_ENDPOINT", "http://localhost:8085/predict"),
        help="FastAPI serving endpoint URL",
    )
    parser.add_argument("--records", type=int, default=100, help="Number of records for one-shot stream")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between requests")
    parser.add_argument("--continuous", action="store_true", help="Run infinitely in a rolling loop across datasets")
    args = parser.parse_args()

    files = get_stream_files(args.data_path)
    if args.continuous:
        stream_continuous(files, endpoint=args.endpoint, delay=args.delay)
    else:
        df_stream = load_raw_stream_data(files[0])
        stream_to_api(df_stream, endpoint=args.endpoint, num_records=args.records, delay=args.delay)
