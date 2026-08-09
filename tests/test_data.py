from src.house_price.data import load_data, get_X_y
from src.house_price.config import FEATURE_NAMES


def test_load_data_shape():
    df = load_data()
    assert len(df) > 0
    assert "Price" in df.columns


def test_get_X_y_columns():
    df = load_data()
    X, y = get_X_y(df)
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == len(y)