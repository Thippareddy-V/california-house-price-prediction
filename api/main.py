from flask import Flask, request, jsonify

from src.house_price.predict import predict
from src.house_price.config import FEATURE_NAMES, FEATURE_RANGES

app = Flask(__name__)


def validate_input(data: dict) -> list[str]:
    errors = []
    for name in FEATURE_NAMES:
        if name not in data:
            errors.append(f"missing field: {name}")
            continue
        value = data[name]
        if not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
            continue
        low, high = FEATURE_RANGES[name]
        if not (low <= value <= high):
            errors.append(f"{name} must be between {low} and {high}")
    return errors


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    errors = validate_input(data)
    if errors:
        return jsonify({"errors": errors}), 400

    price = predict(data)
    return jsonify({"predicted_price_usd": round(price * 100_000, 2)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)