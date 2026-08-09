from src.house_price.predict import predict

SAMPLE = {
    "MedInc": 3.5, "HouseAge": 20, "AveRooms": 5.0, "AveBedrms": 1.0,
    "Population": 1000, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0,
}


def test_predict_returns_float():
    price = predict(SAMPLE)
    assert isinstance(price, float)
    assert price > 0