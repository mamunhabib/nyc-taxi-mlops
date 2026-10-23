# NYC Taxi Trip Duration — MLOps Implementation Plan

## 1. Project Directory Structure

```text
04-NYC-Taxi-MLOps/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference.parquet
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
├── pipelines/
│   └── training_flow.py
├── tests/
│   ├── test_data_prep.py
│   └── test_predict.py
├── serving/
│   ├── app.py
│   ├── schemas.py
│   └── Dockerfile
├── monitoring/
│   ├── streamer.py
│   ├── drift_detector.py
│   └── db.py
├── k8s/
│   ├── platform/
│   │   ├── 00-namespace.yaml
│   │   ├── 01-configmap.yaml
│   │   ├── 02-postgres.yaml
│   │   ├── 03-minio.yaml
│   │   ├── 04-mlflow.yaml
│   │   └── 05-evidently.yaml
│   └── apps/
│       └── serving.yaml
├── Jenkinsfile
├── Makefile
└── requirements.txt
```

---

## 2. Implementation Phases

### PART 1: Platform Foundation (MLOps Track)

#### Pre-existing Infrastructure [ALREADY INSTALLED]
* **Kubernetes Cluster (OrbStack / Mac):** Local container & K8s runtime [ALREADY INSTALLED].
* **Jenkins Automation Server:** CI/CD orchestration server [ALREADY INSTALLED].
* **Kibana & Elasticsearch:** Cluster-wide centralized logging & trace search [ALREADY INSTALLED].

#### Layer 1: Dedicated Platform Services (`k8s/platform/`) [COMPLETED]
* `k8s/platform/00-namespace.yaml`: Dedicated namespace `nyc-taxi-mlops` [COMPLETED].
* `k8s/platform/01-configmap.yaml`: Cluster DNS endpoints (`postgres`, `minio`), ports, and credentials [COMPLETED].
* `k8s/platform/02-postgres.yaml`: PostgreSQL 15 deployment with 2 GiB PVC for MLflow & drift metrics [COMPLETED].
* `k8s/platform/03-minio.yaml`: MinIO S3 object store with 5 GiB PVC (S3 port 9000 & console port 8088) [COMPLETED].
* `k8s/platform/04-mlflow.yaml`: MLflow Tracking Server & Model Registry on port 8087 [COMPLETED].
* `k8s/platform/05-evidently.yaml`: Evidently Web UI Deployment & Service on port 8086 [COMPLETED].

---

### PART 2: Model & Data Development (ML Engineering Track)

#### Phase 0: Scaffolding & Setup [COMPLETED]
* Initialized repository directory structure and established role boundaries.
* Created local Python virtual environment (`venv/`) and installed baseline dependencies.

#### Phase 1: Data Acquisition [COMPLETED]
* Acquired Green Taxi Q1 2023 datasets into `data/raw/`:
  * `green_tripdata_2023-01.parquet` (Baseline training set, ~68k rides)
  * `green_tripdata_2023-02.parquet` (Validation & tuning set, ~65k rides)
  * `green_tripdata_2023-03.parquet` (Production streaming simulation, ~72k rides)

#### Phase 2: Data Preprocessing & Feature Engineering [COMPLETED]
* `src/data_prep.py`: Filters outliers ($1 \le \text{duration} \le 60$ min), generates `PU_DO` interaction feature, fits `DictVectorizer`.
* Output artifacts in `data/processed/`: `dv.pkl`, `train.pkl` (65,946 rows), `val.pkl` (62,574 rows), `test.pkl` (69,392 rows).

#### Phase 3: Model Experimentation & Champion Selection [COMPLETED]
* `config/model_config.yaml`: Candidate model parameters for Ridge, Random Forest, and XGBoost.
* `src/train.py`: Unified training engine with MLflow run tracking.
* Crowned Champion: Selected by lowest validation RMSE and registered as `@champion` (version 1) in MLflow Model Registry.

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

#### Phase 6: Drift Monitoring & Observability [COMPLETED]
* `config/monitoring_config.yaml`: Evidently drift thresholds, test suites, and alert tolerances.
* `monitoring/streamer.py`: Replays March 2023 data as a simulated live production ride stream into FastAPI.
* `monitoring/drift_detector.py`: Evidently AI service computing 3-tier drift and generating Evidently UI workspace snapshots.
* `monitoring/db.py`: PostgreSQL client recording drift metrics to trigger automated retraining loops.
* **Cluster Kibana Integration:** Centralized log search across `nyc-taxi-serving` pods using `kubernetes.labels.app: "nyc-taxi-serving"`.

#### Phase 7: Automated CI/CD & Continuous Training (CT) [COMPLETED]
* `pipelines/training_flow.py`: Automated retraining script triggered when drift exceeds thresholds ($MAPE > 30\%$).
* `Jenkinsfile`: Pipeline automating linting, testing, Docker builds for BOTH images (`nyc-taxi-serving` and `nyc-taxi-ui`), K8s deployment of both services, and CT retraining loop.


