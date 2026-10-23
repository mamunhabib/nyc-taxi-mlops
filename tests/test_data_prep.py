import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from src.data_prep import read_dataframe, preprocess


def test_read_dataframe_duration_and_outliers(tmp_path):
    """Test trip duration calculation and outlier filtering (1 <= duration <= 60)."""
    data = {
        "lpep_pickup_datetime": pd.to_datetime([
            "2023-01-01 10:00:00",
            "2023-01-01 10:00:00",
            "2023-01-01 10:00:00",
        ]),
        "lpep_dropoff_datetime": pd.to_datetime([
            "2023-01-01 10:00:30",  # 0.5 minutes (drop: < 1 min)
            "2023-01-01 10:15:00",  # 15.0 minutes (keep: valid ride)
            "2023-01-01 11:15:00",  # 75.0 minutes (drop: > 60 min)
        ]),
        "PULocationID": [43, 43, 43],
        "DOLocationID": [151, 151, 151],
        "trip_distance": [0.2, 3.5, 20.0],
    }
    df = pd.DataFrame(data)
    test_file = tmp_path / "mock_taxi_trips.parquet"
    df.to_parquet(test_file)

    df_cleaned = read_dataframe(str(test_file))

    # Assert exactly 1 row kept (the 15-minute ride)
    assert len(df_cleaned) == 1
    assert df_cleaned.iloc[0]["duration"] == 15.0
    # Assert categorical columns cast to string
    assert isinstance(df_cleaned.iloc[0]["PULocationID"], str)
    assert isinstance(df_cleaned.iloc[0]["DOLocationID"], str)


def test_preprocess_feature_engineering():
    """Test route PU_DO interaction feature creation and DictVectorizer transformation."""
    df = pd.DataFrame({
        "PULocationID": ["43"],
        "DOLocationID": ["151"],
        "trip_distance": [3.5],
    })
    dv = DictVectorizer()
    X, dv_fitted = preprocess(df, dv, fit_dv=True)

    # Verify PU_DO column created
    assert "PU_DO" in df.columns
    assert df.iloc[0]["PU_DO"] == "43_151"
    # Verify vectorizer matrix shape
    assert X.shape[0] == 1
    assert X.shape[1] > 0
