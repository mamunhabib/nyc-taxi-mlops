#!/usr/bin/env python3
"""
NYC Taxi Trip Duration — Continuous Training (CT) Pipeline
Automated retraining flow triggered by statistical data drift detection.
Evaluates production drift, ingests new data, retrains candidate models,
and promotes new Champion models to the MLflow Model Registry.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from monitoring.drift_detector import run_drift_detection
from monitoring.db import MetricsDB
from src.data_prep import read_dataframe, preprocess, dump_pickle
from src.train import run_training_pipeline


def load_config(config_path: str = "config/monitoring_config.yaml") -> dict:
    """Load monitoring and drift configuration."""
    with open(config_path, "r") as f_in:
        return yaml.safe_load(f_in)["monitoring"]


def check_drift_status(config_path: str = "config/monitoring_config.yaml") -> bool:
    """Run drift detection and determine if retraining is recommended."""
    print("\n[CT Pipeline] Step 1: Checking Production Data Drift...")
    drift_results = run_drift_detection(config_path)
    retrain_recommended = drift_results.get("retraining_recommended", False)
    drift_share = drift_results.get("drift_share", 0.0)

    print(f"\n[CT Pipeline] Drift Check Result: Share={drift_share:.2%}, Retrain Recommended={retrain_recommended}")
    return retrain_recommended


def execute_continuous_training(
    model_config_path: str = "config/model_config.yaml",
    monitoring_config_path: str = "config/monitoring_config.yaml",
):
    """Execute automated end-to-end retraining and champion promotion."""
    print("\n" + "=" * 65)
    print("CONTINUOUS TRAINING (CT) PIPELINE TRIGGERED")
    print("=" * 65)

    mon_cfg = load_config(monitoring_config_path)

    # 1. Update processed data using latest available batches
    print("\n[CT Pipeline] Step 2: Ingesting latest production batches & re-processing...")
    train_raw = "data/raw/green_tripdata_2023-01.parquet"
    val_raw = "data/raw/green_tripdata_2023-02.parquet"
    stream_raw = mon_cfg["current_data_path"]  # 2023-03

    # Merge historical validation with stream data for retraining
    df_val = read_dataframe(val_raw)
    df_stream = read_dataframe(stream_raw)
    df_retrain_val = pd.concat([df_val, df_stream], ignore_index=True)
    print(f"[CT Pipeline] Combined Retraining Set: {len(df_retrain_val):,} rows")

    # Vectorize and update data/processed/val.pkl with the fresh batch
    dv_path = "data/processed/dv.pkl"
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)

    # Ensure base training split exists
    train_processed_path = "data/processed/train.pkl"
    if not os.path.exists(train_processed_path):
        print(f"[CT Pipeline] Generating base training split '{train_processed_path}'...")
        df_train = read_dataframe(train_raw)
        X_train, _ = preprocess(df_train, dv, fit_dv=False)
        y_train = df_train["duration"].values
        dump_pickle((X_train, y_train), train_processed_path)
        print(f"[CT Pipeline] Saved base training split to '{train_processed_path}' (Shape: {X_train.shape})")

    X_val, _ = preprocess(df_retrain_val, dv, fit_dv=False)
    y_val = df_retrain_val["duration"].values
    val_processed_path = "data/processed/val.pkl"
    dump_pickle((X_val, y_val), val_processed_path)
    print(f"[CT Pipeline] Saved updated validation split to '{val_processed_path}' (Shape: {X_val.shape})")

    # 2. Trigger Unified Model Training with MLflow tracking
    print("\n[CT Pipeline] Step 3: Retraining Candidate Models (Ridge, RF, XGBoost)...")
    run_training_pipeline(model_config_path)

    # 3. Update Baseline Reference Dataset for future monitoring
    print("\n[CT Pipeline] Step 4: Updating Golden Baseline Reference Dataset...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8087")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    registry_name = "nyc-taxi-duration-model"
    alias = "champion"

    champion_model_version = client.get_model_version_by_alias(registry_name, alias)
    print(f"[CT Pipeline] Promoted Model: {registry_name} version {champion_model_version.version} (@{alias})")

    # 4. Log successful CT event to PostgreSQL Metrics DB
    db = MetricsDB(
        host=mon_cfg["database"]["host"],
        port=mon_cfg["database"]["port"],
        db_name=mon_cfg["database"]["name"],
        user=mon_cfg["database"]["user"],
        password=mon_cfg["database"]["password"],
    )
    db.log_drift_metrics({
        "timestamp": datetime.now(timezone.utc),
        "dataset_name": "continuous-training-retrain-run",
        "drift_score": 0.0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
        "retraining_recommended": False,
    })

    print("\n" + "=" * 65)
    print("CONTINUOUS TRAINING (CT) COMPLETED SUCCESSFULLY")
    print(f"New Champion Version: {champion_model_version.version}")
    print(f"Timestamp:            {datetime.now(timezone.utc).isoformat()}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Continuous Training (CT) Orchestrator")
    parser.add_argument("--config", default="config/monitoring_config.yaml", help="Path to monitoring config")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Path to model config")
    parser.add_argument("--force", action="store_true", help="Force retraining loop regardless of drift status")
    args = parser.parse_args()

    if args.force:
        print("[CT Pipeline] --force flag enabled. Bypassing drift check and initiating retraining.")
        execute_continuous_training(args.model_config, args.config)
        return

    should_retrain = check_drift_status(args.config)

    if should_retrain:
        print("[CT Pipeline] 🚨 Production Drift Detected exceeding thresholds. Launching Retraining.")
        execute_continuous_training(args.model_config, args.config)
    else:
        print("[CT Pipeline] ✅ Model is healthy. Drift thresholds not breached. No retraining required.")


if __name__ == "__main__":
    main()
