#!/usr/bin/env python3
"""
NYC Taxi Trip Duration — Multi-Model Training & Experiment Tracking
Trains candidate regression models (Ridge, Random Forest, XGBoost), logs metrics
to MLflow, registers the Champion model, and exports the Evidently baseline dataset.
"""

import os
import time
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

import mlflow
from mlflow.tracking import MlflowClient

# Silence optional agent telemetry logs
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Default MinIO credentials for local artifact streaming
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")


def load_pickle(file_path: str) -> Any:
    """Load serialized object from disk."""
    with open(file_path, "rb") as f_in:
        return pickle.load(f_in)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML model configuration file."""
    with open(config_path, "r") as f_in:
        return yaml.safe_load(f_in)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def train_ridge(
    X_train, y_train, X_val, y_val, params: Dict[str, Any]
) -> Tuple[Any, float, float, float]:
    """Train Ridge regression baseline."""
    start_time = time.time()
    model = Ridge(**params)
    model.fit(X_train, y_train)
    duration = time.time() - start_time

    y_pred = model.predict(X_val)
    rmse, mae = compute_metrics(y_val, y_pred)
    return model, rmse, mae, duration


def train_random_forest(
    X_train, y_train, X_val, y_val, params: Dict[str, Any]
) -> Tuple[Any, float, float, float]:
    """Train Random Forest regressor ensemble."""
    start_time = time.time()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    duration = time.time() - start_time

    y_pred = model.predict(X_val)
    rmse, mae = compute_metrics(y_val, y_pred)
    return model, rmse, mae, duration


def train_xgboost(
    X_train, y_train, X_val, y_val, params: Dict[str, Any]
) -> Tuple[Any, float, float, float]:
    """Train XGBoost gradient boosting regressor."""
    start_time = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    duration = time.time() - start_time

    y_pred = model.predict(X_val)
    rmse, mae = compute_metrics(y_val, y_pred)
    return model, rmse, mae, duration


def export_reference_dataset(
    raw_val_path: str,
    output_path: str,
    champion_model: Any,
    X_val: Any,
):
    """
    Export validation data with Champion predictions as baseline reference for Evidently AI.
    """
    print(f"\n[Artifact] Creating Evidently Baseline Reference: {output_path}")
    df_raw = pd.read_parquet(raw_val_path)

    pickup_col = "lpep_pickup_datetime" if "lpep_pickup_datetime" in df_raw.columns else "tpep_pickup_datetime"
    dropoff_col = "lpep_dropoff_datetime" if "lpep_dropoff_datetime" in df_raw.columns else "tpep_dropoff_datetime"

    df_raw["duration"] = (df_raw[dropoff_col] - df_raw[pickup_col]).dt.total_seconds() / 60.0
    df_clean = df_raw[(df_raw.duration >= 1) & (df_raw.duration <= 60)].copy()
    if "trip_distance" in df_clean.columns:
        df_clean = df_clean[(df_clean.trip_distance > 0) & (df_clean.trip_distance <= 50)].copy()

    df_clean["PULocationID"] = df_clean["PULocationID"].astype(str)
    df_clean["DOLocationID"] = df_clean["DOLocationID"].astype(str)
    df_clean["PU_DO"] = df_clean["PULocationID"] + "_" + df_clean["DOLocationID"]

    # Generate Champion predictions
    df_clean["prediction"] = champion_model.predict(X_val)

    # Keep key monitoring columns
    cols_to_keep = [
        "PULocationID",
        "DOLocationID",
        "PU_DO",
        "trip_distance",
        pickup_col,
        "duration",
        "prediction",
    ]
    df_reference = df_clean[[c for c in cols_to_keep if c in df_clean.columns]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_reference.to_parquet(output_path, index=False)
    print(f"[Artifact] Reference dataset saved successfully ({df_reference.shape[0]} rows).")


def run_training_pipeline(config_path: str):
    """Orchestrate end-to-end model training, logging, champion selection, and artifact export."""
    config = load_config(config_path)

    # Setup MLflow Tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["experiment"]["tracking_uri"])
    experiment_name = config["experiment"]["name"]
    registry_name = config["experiment"]["model_registry_name"]
    champion_alias = config["experiment"].get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment:   {experiment_name}")

    # Load Data Splits
    processed_dir = config["data"]["processed_dir"]
    print(f"\nLoading processed splits from '{processed_dir}'...")
    X_train, y_train = load_pickle(os.path.join(processed_dir, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(processed_dir, "val.pkl"))
    print(f"Loaded: Train shape={X_train.shape}, Val shape={X_val.shape}")

    models_config = config["models"]
    results = []

    # 1. Candidate 1: Ridge Regression
    if models_config.get("ridge", {}).get("enabled", False):
        print("\n--- Training Candidate 1: Ridge Baseline ---")
        ridge_params = models_config["ridge"]["params"]
        with mlflow.start_run(run_name="ridge-baseline") as run:
            mlflow.log_params(ridge_params)
            mlflow.set_tag("model_family", "linear_model")
            model, rmse, mae, dur = train_ridge(X_train, y_train, X_val, y_val, ridge_params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "train_duration_sec": dur})
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")
            print(f"Ridge Complete | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f} | Time: {dur:.2f}s")
            results.append({"name": "Ridge", "model": model, "rmse": rmse, "mae": mae, "run_id": run.info.run_id})

    # 2. Candidate 2: Random Forest
    if models_config.get("random_forest", {}).get("enabled", False):
        print("\n--- Training Candidate 2: Random Forest ---")
        rf_params = models_config["random_forest"]["params"]
        with mlflow.start_run(run_name="random-forest") as run:
            mlflow.log_params(rf_params)
            mlflow.set_tag("model_family", "ensemble_rf")
            model, rmse, mae, dur = train_random_forest(X_train, y_train, X_val, y_val, rf_params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "train_duration_sec": dur})
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")
            print(f"Random Forest Complete | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f} | Time: {dur:.2f}s")
            results.append({"name": "RandomForest", "model": model, "rmse": rmse, "mae": mae, "run_id": run.info.run_id})

    # 3. Candidate 3: XGBoost
    if models_config.get("xgboost", {}).get("enabled", False):
        print("\n--- Training Candidate 3: XGBoost ---")
        xgb_params = models_config["xgboost"]["params"]
        with mlflow.start_run(run_name="xgboost") as run:
            mlflow.log_params(xgb_params)
            mlflow.set_tag("model_family", "gradient_boosting")
            model, rmse, mae, dur = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "train_duration_sec": dur})
            mlflow.xgboost.log_model(model, artifact_path="model")
            print(f"XGBoost Complete | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f} | Time: {dur:.2f}s")
            results.append({"name": "XGBoost", "model": model, "rmse": rmse, "mae": mae, "run_id": run.info.run_id})

    # Select Champion: Preferred model (Ridge per Zoomcamp lin_reg) or lowest RMSE
    preferred_name = config["experiment"].get("preferred_model")
    matched = [r for r in results if preferred_name and r["name"].lower() == preferred_name.lower()]
    best_result = matched[0] if matched else min(results, key=lambda x: x["rmse"])
    print("\n" + "=" * 60)
    print("EXPERIMENT SCOREBOARD")
    print("=" * 60)
    for res in sorted(results, key=lambda x: x["rmse"]):
        winner_mark = " <-- CHAMPION WINNER!" if res["name"] == best_result["name"] else ""
        print(f"Model: {res['name']:<14} | RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f}{winner_mark}")
    print("=" * 60)

    # Register Champion in MLflow Model Registry
    client = MlflowClient()
    champion_run_id = best_result["run_id"]
    model_uri = f"runs:/{champion_run_id}/model"
    print(f"\n[Registry] Registering Champion model from run: {champion_run_id}")
    reg_model = mlflow.register_model(model_uri=model_uri, name=registry_name)
    client.set_registered_model_alias(name=registry_name, alias=champion_alias, version=reg_model.version)
    print(f"[Registry] Model '{registry_name}' version {reg_model.version} promoted with alias '@{champion_alias}'")

    # Export Production Binary weights
    models_dir = config["data"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_export_path = os.path.join(models_dir, "model.pkl")
    with open(model_export_path, "wb") as f_out:
        pickle.dump(best_result["model"], f_out)
    print(f"[Artifact] Champion binary saved to '{model_export_path}'")

    # Export Baseline Reference Dataset (data/reference.parquet)
    raw_val_file = "data/raw/green_tripdata_2023-02.parquet"
    ref_parquet_path = config["data"]["reference_parquet_path"]
    if os.path.exists(raw_val_file):
        export_reference_dataset(
            raw_val_path=raw_val_file,
            output_path=ref_parquet_path,
            champion_model=best_result["model"],
            X_val=X_val,
        )

    print("\nPhase 3 Model Training & Experiment Tracking successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Duration Multi-Model Training Pipeline")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model config YAML")
    args = parser.parse_args()

    run_training_pipeline(args.config)
