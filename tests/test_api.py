import pytest

from api.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid(client):
    payload = {
        "MedInc": 3.5, "HouseAge": 20, "AveRooms": 5.0, "AveBedrms": 1.0,
        "Population": 1000, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price_usd" in response.get_json()


def test_predict_missing_field(client):
    response = client.post("/predict", json={"MedInc": 3.5})
    assert response.status_code == 400