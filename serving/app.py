import os
import pickle
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
import mlflow

from serving.schemas import RideInput, PredictionOutput

# Global references for loaded artifacts
model = None
dv = None
model_version_tag = "champion"


def load_artifacts():
    """Load preprocessor (dv.pkl) and Champion model from disk or MLflow registry."""
    global model, dv, model_version_tag

    dv_path = os.getenv("DV_PATH", "data/processed/dv.pkl")
    if not os.path.exists(dv_path):
        raise FileNotFoundError(f"DictVectorizer not found at {dv_path}")

    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)

    # Check if local model binary exists first
    local_model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    if os.path.exists(local_model_path):
        with open(local_model_path, "rb") as f_in:
            model = pickle.load(f_in)
        model_version_tag = "local-champion"
        print(f"Loaded champion model from local path: '{local_model_path}'")
        return

    # Fallback to MLflow registry
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "nyc-taxi-duration-model")
    model_alias = os.getenv("MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{model_alias}"
    print(f"Loading champion model from MLflow registry: {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    model_version_tag = f"{model_name}@{model_alias}"
    print("Successfully loaded model from MLflow registry.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load artifacts once into memory
    load_artifacts()
    yield


app = FastAPI(
    title="NYC Taxi Trip Duration Serving Service",
    description="Production model serving API using FastAPI and MLflow Champion Model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Kubernetes liveness and readiness probe endpoint."""
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded")
    return {
        "status": "healthy",
        "model_version": model_version_tag,
    }


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(ride: RideInput) -> PredictionOutput:
    """Predict trip duration in minutes for a given taxi ride."""
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    try:
        features = [{
            "PU_DO": f"{ride.PULocationID}_{ride.DOLocationID}",
            "trip_distance": ride.trip_distance,
        }]
        X = dv.transform(features)
        prediction = model.predict(X)
        pred_val = float(prediction[0])
        return PredictionOutput(trip_duration=round(pred_val, 2), model_version=model_version_tag)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )
