import os
import pickle
import argparse
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    """Serialize Python object to a pickle file."""
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def read_dataframe(filename: str) -> pd.DataFrame:
    """Read raw taxi parquet file and apply baseline cleaning."""
    df = pd.read_parquet(filename)

    # Detect Green (lpep) vs Yellow (tpep) timestamp columns
    if "lpep_pickup_datetime" in df.columns:
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"
    else:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"

    # Compute trip duration in minutes
    df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0

    # Filter out outliers (trips between 1 minute and 60 minutes, distance between 0 and 50 miles)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    if "trip_distance" in df.columns:
        df = df[(df.trip_distance > 0) & (df.trip_distance <= 50)]

    # Cast location IDs to string for categorical encoding
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """Engineer route interaction feature and vectorize input matrix."""
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PULocationID", "DOLocationID", "PU_DO"]
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].to_dict(orient="records")

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    """Run full data preparation pipeline for train, val, and test months."""
    print(f"Reading raw data from: {raw_data_path}")
    print(f"Saving processed data to: {dest_path}")

    train_file = os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet")
    val_file = os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet")
    test_file = os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")

    # Load cleaned dataframes
    print(f"Loading {train_file}...")
    df_train = read_dataframe(train_file)
    print(f"Loading {val_file}...")
    df_val = read_dataframe(val_file)
    print(f"Loading {test_file}...")
    df_test = read_dataframe(test_file)

    # Extract target values
    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit DictVectorizer on training set and transform val & test
    print("Vectorizing features...")
    dv = DictVectorizer(sparse=True)
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Ensure output destination exists
    os.makedirs(dest_path, exist_ok=True)

    # Save artifacts
    print(f"Saving processed artifacts to {dest_path}...")
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    print(f"Done! Train rows: {X_train.shape[0]}, Val rows: {X_val.shape[0]}, Test rows: {X_test.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NYC Taxi Parquet data for MLOps training")
    parser.add_argument("--raw_data_path", type=str, default="data/raw", help="Directory of raw parquet files")
    parser.add_argument("--dest_path", type=str, default="data/processed", help="Directory to save processed pkl files")
    parser.add_argument("--dataset", type=str, default="green", help="Taxi dataset type: 'green' or 'yellow'")
    args = parser.parse_args()

    run_data_prep(
        raw_data_path=args.raw_data_path,
        dest_path=args.dest_path,
        dataset=args.dataset,
    )
