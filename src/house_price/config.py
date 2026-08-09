import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(BASE_DIR, "models", "v1")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# Used by the API to validate incoming requests before they hit the model
FEATURE_RANGES = {
    "MedInc": (0.5, 15.0),
    "HouseAge": (1, 52),
    "AveRooms": (1.0, 20.0),
    "AveBedrms": (0.5, 5.0),
    "Population": (3, 40000),
    "AveOccup": (0.5, 20.0),
    "Latitude": (32.0, 42.0),
    "Longitude": (-125.0, -114.0),
}

XGB_PARAMS = dict(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)