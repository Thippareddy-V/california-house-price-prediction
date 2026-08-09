import joblib
import pandas as pd

from .config import MODEL_PATH, FEATURE_NAMES_PATH

_model = None
_feature_names = None


def _load():
    global _model, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _feature_names = joblib.load(FEATURE_NAMES_PATH)
    return _model, _feature_names


def predict(features: dict) -> float:
    """features: dict with keys matching FEATURE_NAMES. Returns price in $100,000s."""
    model, feature_names = _load()
    row = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)
    return float(model.predict(row)[0])