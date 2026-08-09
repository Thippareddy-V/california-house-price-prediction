## Running Locally

```bash
git clone https://github.com/<your-username>/california-house-price-prediction.git
cd california-house-price-prediction
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

Train the model (or use the one already saved in `models/v1/`):
```bash
python -m src.house_price.train
```

Run the API (terminal 1):
```bash
python -m api.main
```

Run the web interface (terminal 2):
```bash
streamlit run app/app.py
```

Run tests:
```bash
pytest tests/ -v
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · Flask · Streamlit · pytest · GitHub Actions (CI)

## Next Steps

- Docker deployment (Dockerfile included, container build pending)
- Model versioning beyond v1