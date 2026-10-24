# NYC Taxi Trip Duration — MLOps Implementation Plan

## 1. Project Directory Structure

```text
04-NYC-Taxi-MLOps/
├── data/
│   ├── raw/
│   │   ├── green_tripdata_2023-01.parquet  (Jan: Train split)
│   │   ├── green_tripdata_2023-02.parquet  (Feb: Validation split)
│   │   ├── green_tripdata_2023-03.parquet  (Mar: Initial production test)
│   │   ├── green_tripdata_2023-04.parquet  (Apr: Production stream)
│   │   ├── green_tripdata_2023-05.parquet  (May: Production stream)
│   │   └── green_tripdata_2023-06.parquet  (Jun: Production stream)
│   ├── processed/
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   ├── test.pkl
│   │   └── dv.pkl
│   ├── reference.parquet
│   └── q2_samples.json                      (750 genuine Q2 2023 trips for streamer pod)
├── models/
│   └── model.pkl
├── config/
│   ├── model_config.yaml
│   └── monitoring_config.yaml
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── ui/
│   ├── app.py                              (Streamlit Interactive Frontend)
│   ├── Dockerfile
│   └── requirements.txt
├── pipelines/
│   └── training_flow.py                    (Automated Continuous Training Orchestrator)
├── tests/
│   ├── test_data_prep.py
│   └── test_predict.py
├── serving/
│   ├── app.py
│   ├── schemas.py
│   └── Dockerfile
├── monitoring/
│   ├── streamer.py                         (Multi-month continuous CLI traffic generator)
│   ├── drift_detector.py                   (Evidently AI 5-panel drift analysis engine)
│   └── db.py
├── k8s/
│   ├── streamer-pod.yaml                   (Standalone Kubernetes traffic generator pod)
│   ├── kibana-dashboard.ndjson             (10 saved objects: metrics, charts, histogram)
│   ├── platform/
│   │   ├── 00-namespace.yaml
│   │   ├── 01-configmap.yaml               (Cluster endpoints & Asia/Kuala_Lumpur TZ)
│   │   ├── 02-postgres.yaml
│   │   ├── 03-minio.yaml
│   │   ├── 04-mlflow.yaml
│   │   ├── 05-evidently.yaml
│   │   └── 06-kibana-dashboard.yaml        (Automated K8s Job for Kibana dashboard import)
│   └── apps/
│       ├── serving.yaml                    (FastAPI Serving Deployment & Service :8085)
│       ├── ui.yaml                         (Streamlit UI Deployment & Service :8089)
│       └── sealed-git-secret.yaml
├── Jenkinsfile                             (Automated CI/CD build & deploy pipeline)
├── Makefile
└── requirements.txt
```

---

## 2. Implementation Phases

### PART 1: Platform Foundation (MLOps Track)

#### Pre-existing Infrastructure [ALREADY INSTALLED]
* **Kubernetes Cluster (OrbStack / Mac):** Local container & K8s runtime [ALREADY INSTALLED].
* **Jenkins Automation Server:** CI/CD orchestration server configured with `Asia/Kuala_Lumpur` timezone [ALREADY INSTALLED].
* **Kibana & Elasticsearch:** Centralized logging stack with browser-matched timezone [ALREADY INSTALLED].

#### Layer 1: Dedicated Platform Services (`k8s/platform/`) [COMPLETED]
* `k8s/platform/00-namespace.yaml`: Dedicated namespace `nyc-taxi-mlops` [COMPLETED].
* `k8s/platform/01-configmap.yaml`: Cluster DNS endpoints, S3 bucket, ports, and `TZ: Asia/Kuala_Lumpur` [COMPLETED].
* `k8s/platform/02-postgres.yaml`: PostgreSQL 15 deployment with 2 GiB PVC for MLflow & drift metrics [COMPLETED].
* `k8s/platform/03-minio.yaml`: MinIO S3 object store with 5 GiB PVC (S3 port 9000 & console port 8088) [COMPLETED].
* `k8s/platform/04-mlflow.yaml`: MLflow Tracking Server & Model Registry on port 8087 [COMPLETED].
* `k8s/platform/05-evidently.yaml`: Evidently Web UI Deployment & Service on port 8086 with persistent workspace [COMPLETED].
* `k8s/platform/06-kibana-dashboard.yaml`: Automated Kubernetes setup Job importing 10 saved objects (index patterns, metric gauges, request histogram, and donut charts) into Kibana [COMPLETED].

---

### PART 2: Model & Data Development (ML Engineering Track)

#### Phase 0: Scaffolding & Setup [COMPLETED]
* Initialized repository directory structure and established role boundaries.
* Created local Python virtual environment (`venv/`) and installed baseline dependencies.

#### Phase 1: Data Acquisition [COMPLETED]
* Acquired Green Taxi Q1 & Q2 2023 datasets into `data/raw/` and synced to MinIO S3 (`s3://nyc-taxi-mlops/raw/`):
  * `green_tripdata_2023-01.parquet` (Baseline training set, ~68k rides)
  * `green_tripdata_2023-02.parquet` (Validation & tuning set, ~65k rides)
  * `green_tripdata_2023-03.parquet` (Initial production test batch, ~72k rides)
  * `green_tripdata_2023-04.parquet` (April production stream, ~69k rides)
  * `green_tripdata_2023-05.parquet` (May production stream, ~74k rides)
  * `green_tripdata_2023-06.parquet` (June production stream, ~68k rides)

#### Phase 2: Data Preprocessing & Feature Engineering [COMPLETED]
* `src/data_prep.py`: Filters outliers ($1 \le \text{duration} \le 60$ min), generates `PU_DO` interaction feature, fits `DictVectorizer`.
* Output artifacts in `data/processed/`: `dv.pkl`, `train.pkl` (65,946 rows), `val.pkl` (62,574 rows), `test.pkl` (69,392 rows).

#### Phase 3: Model Experimentation & Champion Selection [COMPLETED]
* `config/model_config.yaml`: Candidate model parameters for Ridge, Random Forest, and XGBoost.
* `src/train.py`: Unified training engine with MLflow run tracking.
* Crowned Champion: Selected by lowest validation RMSE and registered as `@champion` in MLflow Model Registry.

#### The Hand-off Deliverables [COMPLETED]
* `models/model.pkl`: Champion model binary exported to disk and MinIO S3.
* `data/processed/dv.pkl`: Fitted DictVectorizer feature encoder.
* `data/reference.parquet`: Golden baseline dataset for drift monitoring.

---

### PART 3: Serving, Observability & CI/CD/CT (MLOps Track)

#### Milestone 3.5: Data Pipeline Unit Testing [COMPLETED]
* `tests/test_data_prep.py`: Unit tests for `read_dataframe`, duration calculation, outlier filtering ($1 \le \text{duration} \le 60$), and `PU_DO` feature vectorization.

#### Phase 4: Containerized Model Serving [COMPLETED]
* `serving/schemas.py`: Pydantic payload validation (`PULocationID`, `DOLocationID`, `trip_distance`).
* `serving/app.py`: FastAPI application exposing `POST /predict` and `GET /health`.
* `tests/test_predict.py`: Integration test validating FastAPI inference endpoint and payload schema.
* `serving/Dockerfile`: Container image build for GitHub Container Registry (GHCR).

#### Phase 5: Kubernetes Serving Workload [COMPLETED]
* `k8s/apps/serving.yaml`: FastAPI Deployment & LoadBalancer Service on port 8085 pulling image and `@champion` weights from MinIO on boot.

#### Phase 5.5: Decoupled User Interface (Streamlit Frontend) [COMPLETED]
* `ui/app.py`: Modern Streamlit passenger/dispatcher web app querying FastAPI for real-time trip predictions.
* `ui/requirements.txt`: Lightweight frontend dependency list (Streamlit, Requests, Plotly).
* `ui/Dockerfile`: Dedicated container image build for the Streamlit UI.
* `k8s/apps/ui.yaml`: Streamlit Deployment & LoadBalancer Service on port 8089 communicating with FastAPI via internal K8s DNS (`http://nyc-taxi-serving:8000`).

#### Phase 6: Drift Monitoring, Continuous Streaming & Observability [COMPLETED]
* `config/monitoring_config.yaml`: Evidently drift thresholds (30% share, $p=0.05$), test suites, and alert tolerances.
* `monitoring/streamer.py`: CLI traffic generator supporting multi-month rolling loop across April, May, and June datasets (`--continuous`).
* `k8s/streamer-pod.yaml`: Standalone, self-contained Kubernetes traffic generator pod running `python:3.11-slim` with 750 genuine Q2 2023 taxi trips in a ConfigMap, streaming to `http://nyc-taxi-serving:8000/predict` at ~2 req/sec.
* `monitoring/drift_detector.py`: Evidently AI service computing 3-tier drift and generating 5 live dashboard panels in Evidently UI:
  1. Share of Drifted Features
  2. Number of Drifted Features
  3. Production Batch Row Count
  4. Mean Predicted Duration (mins)
  5. Data Drift Trend Over Time
* `monitoring/db.py`: PostgreSQL client recording drift metrics to trigger automated retraining loops.
* **Cluster Kibana Dashboard:** Automated setup via `06-kibana-dashboard.yaml` displaying live inference metrics, log search, status breakdown, and volume charts at `http://localhost:8084`.

#### Phase 7: Automated CI/CD & Continuous Training (CT) [COMPLETED]
* `Jenkinsfile`: Production pipeline automating Docker builds for both images (`nyc-taxi-serving` and `nyc-taxi-ui`), pushing to GHCR, zero-downtime rolling restart on Kubernetes, and smoke tests.
* `pipelines/training_flow.py`: Automated Continuous Training orchestrator. Checks Evidently drift status ($> 30\%$), ingests fresh production data, vectorizes and updates `data/processed/val.pkl`, retrains candidate models (Ridge, Random Forest, XGBoost), and promotes the winning model to `@champion` in the MLflow Model Registry.
* **Timezone Synchronization:** Cluster-wide alignment of Jenkins, platform services, and application workloads to `Asia/Kuala_Lumpur` (`UTC+8`).


