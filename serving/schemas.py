from pydantic import BaseModel, Field


class RideInput(BaseModel):
    PULocationID: int = Field(..., ge=1, le=265, description="Pick-up TLC Taxi Zone ID (1-265)")
    DOLocationID: int = Field(..., ge=1, le=265, description="Drop-off TLC Taxi Zone ID (1-265)")
    trip_distance: float = Field(..., gt=0.0, description="Trip distance in miles, must be positive")

    model_config = {
        "json_schema_extra": {
            "example": {
                "PULocationID": 43,
                "DOLocationID": 151,
                "trip_distance": 2.5,
            }
        }
    }


class PredictionOutput(BaseModel):
    trip_duration: float = Field(..., description="Predicted trip duration in minutes")
    model_version: str = Field(default="champion", description="Identifier of serving model")
