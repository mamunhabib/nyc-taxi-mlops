import os
import pickle
from typing import Dict, Any, Tuple


def load_artifacts(
    dv_path: str = "data/processed/dv.pkl",
    model_path: str = "models/model.pkl",
) -> Tuple[Any, Any]:
    """Load fitted DictVectorizer preprocessor and trained Champion model artifact."""
    if not os.path.exists(dv_path):
        raise FileNotFoundError(f"DictVectorizer artifact not found: {dv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)

    return dv, model


def prepare_features(ride: Dict[str, Any]) -> list:
    """Format single raw ride dictionary into engineered feature dictionary."""
    pu = str(ride.get("PULocationID", ""))
    do = str(ride.get("DOLocationID", ""))
    trip_distance = float(ride.get("trip_distance", 0.0))

    feature_dict = {
        "PU_DO": f"{pu}_{do}",
        "trip_distance": trip_distance,
    }
    return [feature_dict]


def predict_duration(ride: Dict[str, Any], dv: Any = None, model: Any = None) -> float:
    """Predict trip duration in minutes for a single taxi ride."""
    if dv is None or model is None:
        dv, model = load_artifacts()

    features = prepare_features(ride)
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])
