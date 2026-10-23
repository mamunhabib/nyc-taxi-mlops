import pytest
from fastapi.testclient import TestClient

from serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 OK and reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_version" in payload


def test_predict_endpoint_valid_ride(client):
    """Verify /predict returns status 200 and a realistic duration estimate."""
    ride_data = {
        "PULocationID": 43,
        "DOLocationID": 151,
        "trip_distance": 2.5,
    }
    response = client.post("/predict", json=ride_data)
    assert response.status_code == 200
    payload = response.json()
    assert "trip_duration" in payload
    assert isinstance(payload["trip_duration"], float)
    assert 1.0 <= payload["trip_duration"] <= 60.0


def test_predict_endpoint_invalid_distance(client):
    """Verify /predict rejects zero or negative distances with 422 Unprocessable Entity."""
    invalid_ride = {
        "PULocationID": 43,
        "DOLocationID": 151,
        "trip_distance": -5.0,
    }
    response = client.post("/predict", json=invalid_ride)
    assert response.status_code == 422
