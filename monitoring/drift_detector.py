import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import yaml
import pandas as pd
from datetime import datetime, timezone

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace
from evidently.sdk.panels import counter_panel, line_plot_panel, text_panel
from evidently.sdk.models import PanelMetric

from monitoring.db import MetricsDB
from monitoring.streamer import load_raw_stream_data


def load_config(config_path: str = "config/monitoring_config.yaml") -> dict:
    with open(config_path, "r") as f_in:
        return yaml.safe_load(f_in)["monitoring"]


def prepare_current_dataset(
    raw_path: str,
    dv_path: str = "data/processed/dv.pkl",
    model_path: str = "models/model.pkl",
    sample_size: int = 5000,
) -> pd.DataFrame:
    """Load March data, generate PU_DO, and compute Champion model predictions."""
    print(f"[DriftDetector] Loading live batch from: {raw_path}")
    df = load_raw_stream_data(raw_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).copy()

    # Load artifacts to compute predictions
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)

    dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    X = dv.transform(dicts)
    df["prediction"] = model.predict(X)

    return df


def setup_project_and_dashboard(ws: RemoteWorkspace, project_name: str):
    """Ensure project and monitoring dashboard panels exist in Evidently UI."""
    projects = ws.search_project(project_name)
    if projects:
        project = projects[0]
    else:
        project = ws.create_project(project_name)
        project.description = "NYC Taxi Trip Duration production drift tracking."
        project.save()

    try:
        current_panels = project.dashboard.model().panels
        if not current_panels:
            project.dashboard.add_panel(text_panel(title="NYC Taxi Trip Duration Drift Monitoring"))
            project.dashboard.add_panel(
                counter_panel(
                    title="Share of Drifted Features",
                    size="half",
                    values=[
                        PanelMetric(
                            legend="Share",
                            metric="evidently:metric_v2:DriftedColumnsCount",
                            metric_labels={"value_type": "share"},
                        ),
                    ],
                    aggregation="last",
                )
            )
            project.dashboard.add_panel(
                counter_panel(
                    title="Number of Drifted Features",
                    size="half",
                    values=[
                        PanelMetric(
                            legend="Count",
                            metric="evidently:metric_v2:DriftedColumnsCount",
                            metric_labels={"value_type": "count"},
                        ),
                    ],
                    aggregation="last",
                )
            )
            project.dashboard.add_panel(
                line_plot_panel(
                    title="Data Drift Over Time (Share of Drifted Features)",
                    values=[
                        PanelMetric(
                            legend="Share of Drifted Features",
                            metric="evidently:metric_v2:DriftedColumnsCount",
                            metric_labels={"value_type": "share"},
                        ),
                    ],
                    size="full",
                )
            )
            ws.save_dashboard(project.id, project.dashboard.model())
    except Exception as e:
        print(f"[DriftDetector] Dashboard panels check: {e}")

    return project


def push_to_evidently_ui(ws_url: str, project_name: str, snapshot) -> str:
    """Push report snapshot directly to Evidently UI server."""
    print(f"[DriftDetector] Connecting to Evidently UI at {ws_url}...")
    ws = RemoteWorkspace(ws_url)
    project = setup_project_and_dashboard(ws, project_name)
    run_ref = ws.add_run(project.id, snapshot)
    report_url = f"{ws_url}/projects/{project.id}/reports/{run_ref.id}"
    dashboard_url = f"{ws_url}/projects/{project.id}/dashboard"
    print(f"[DriftDetector] Snapshot successfully published to Evidently Web UI!")
    print(f"[DriftDetector] Direct Report URL:    {report_url}")
    print(f"[DriftDetector] Project Dashboard URL: {dashboard_url}")
    return report_url


def run_drift_detection(config_path: str = "config/monitoring_config.yaml") -> dict:
    """Execute end-to-end drift detection, publish to Evidently UI, and log to DB."""
    config = load_config(config_path)

    features_num = config["features"]["numerical"]
    features_cat = config["features"]["categorical"]
    features = features_num + features_cat

    # 1. Load Baseline Reference Data
    ref_path = config["reference_data_path"]
    print(f"\n[DriftDetector] Loading Baseline Reference from: {ref_path}")
    df_ref = pd.read_parquet(ref_path)[features].copy()
    if len(df_ref) > 10000:
        df_ref = df_ref.sample(10000, random_state=42)

    # 2. Load & Score Current Production Batch
    curr_path = config["current_data_path"]
    df_curr = prepare_current_dataset(curr_path, sample_size=5000)[features].copy()

    # 3. Configure DataDefinition
    data_def = DataDefinition(
        numerical_columns=features_num,
        categorical_columns=features_cat,
    )
    ref_ds = Dataset.from_pandas(df_ref, data_definition=data_def)
    curr_ds = Dataset.from_pandas(df_curr, data_definition=data_def)

    # 4. Generate Evidently Report
    print("[DriftDetector] Running Evidently Data Drift analysis...")
    drift_threshold = config["thresholds"]["drift_share_threshold"]
    p_val = config.get("thresholds", {}).get("p_value", 0.05)

    drift_report = Report(metrics=[
        DataDriftPreset(drift_share=drift_threshold, threshold=p_val),
        DataSummaryPreset(),
    ])
    snapshot = drift_report.run(
        current_data=curr_ds,
        reference_data=ref_ds,
        name=f"drift_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )

    # 5. Push directly to Evidently Web UI Server
    ui_url = config["evidently_ui"]["url"]
    report_ui_url = push_to_evidently_ui(ui_url, config["project_name"], snapshot)

    # 6. Extract Summary Metrics
    results = snapshot.dict()
    metrics = results.get("metrics", [])
    drift_metric_val = metrics[0].get("value", {}) if metrics else {}
    drift_share = drift_metric_val.get("share", 0.0)
    n_drifted = int(drift_metric_val.get("count", 0))

    dataset_drift = drift_share >= drift_threshold
    retrain_recommended = dataset_drift

    print("\n" + "=" * 60)
    print("EVIDENTLY AI DRIFT REPORT SUMMARY")
    print("=" * 60)
    print(f"Dataset Drift Detected:      {dataset_drift}")
    print(f"Drifted Feature Share:       {drift_share:.2%} (Threshold: {drift_threshold:.0%})")
    print(f"Number of Drifted Features:  {n_drifted}")
    print(f"Retraining Recommended:      {'🚨 YES (Trigger Phase 7 CT)' if retrain_recommended else '✅ NO (Model is Healthy)'}")
    print("=" * 60)

    # 7. Persist to PostgreSQL Metrics DB
    db = MetricsDB(
        host=config["database"]["host"],
        port=config["database"]["port"],
        db_name=config["database"]["name"],
        user=config["database"]["user"],
        password=config["database"]["password"],
    )
    db.log_drift_metrics({
        "timestamp": datetime.now(timezone.utc),
        "dataset_name": "green-tripdata-2023-03",
        "drift_score": drift_share,
        "number_of_drifted_columns": n_drifted,
        "share_of_drifted_columns": drift_share,
        "dataset_drift": dataset_drift,
        "retraining_recommended": retrain_recommended,
    })

    return {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "retraining_recommended": retrain_recommended,
        "report_ui_url": report_ui_url,
    }


if __name__ == "__main__":
    run_drift_detection("config/monitoring_config.yaml")
