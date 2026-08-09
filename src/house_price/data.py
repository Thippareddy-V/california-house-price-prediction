import pandas as pd

from .config import FEATURE_NAMES

# Fallback source if sklearn's direct download is blocked (locked-down networks).
# Rebuilds the identical 8 features + target using the same transform sklearn applies.
RAW_CSV_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset as a single DataFrame with a 'Price' column."""
    try:
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df["Price"] = housing.target
        return df

    except Exception:
        raw = pd.read_csv(RAW_CSV_URL).dropna(subset=["total_bedrooms"])
        df = pd.DataFrame({
            "MedInc": raw["median_income"],
            "HouseAge": raw["housing_median_age"],
            "AveRooms": raw["total_rooms"] / raw["households"],
            "AveBedrms": raw["total_bedrooms"] / raw["households"],
            "Population": raw["population"],
            "AveOccup": raw["population"] / raw["households"],
            "Latitude": raw["latitude"],
            "Longitude": raw["longitude"],
            "Price": raw["median_house_value"] / 100_000.0,
        }).reset_index(drop=True)
        return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df["Price"]
    return X, y
    