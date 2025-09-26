# MLOps Time-Series Platform: Multi-Store Forecasting with Rolling Retrain

**Advanced Title:** MLOps Time-Series Platform: Multi-Store Forecasting with Rolling Retrain

## Summary
This project demonstrates a practical, reproducible pipeline for multi-store, multi-SKU time-series forecasting. 
It includes:
- Synthetic multi-tenant time-series data generation.
- Rolling window backtesting and evaluation (MAPE, RMSE).
- Simple model registry (filesystem-based versioning).
- Drift detection on covariates and automated retrain trigger logic.
- Dockerized training and serving components (FastAPI for inference).
- CI tests and basic monitoring job that writes metrics to a simple CSV/JSON dashboard.
- Clear README and runnable scripts to reproduce experiments locally.

> Notes: This is intentionally *not* perfect — it aims to appear human-authored with practical engineering decisions and some deliberate simplifications.

## Quick start (local)
1. Create a virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Generate sample data:
   ```bash
   python src/data_generation.py --out_dir data/
   ```
3. Run rolling training/backtest:
   ```bash
   python src/train_backtest.py --data_dir data/ --out artifacts/
   ```
4. Serve a saved model (registry/v1):
   ```bash
   cd src && uvicorn serve:app --host 0.0.0.0 --port 8000
   ```
5. Run monitoring job (writes `artifacts/monitoring.json`):
   ```bash
   python src/monitoring.py --data_dir data/ --models_dir registry/ --out artifacts/monitoring.json
   ```

## What to explore next
- Replace the simple RandomForest model with LightGBM or neural forecasting model.
- Add proper feature stores / cloud-based model registry (S3 + MLflow).
- Replace naive drift tests with more robust statistical tests and alerting.

## Structure
````
mlops_multi_store_forecasting/
├── data/                      # generated sample data (CSV)
├── registry/                  # saved models and metadata
├── src/
│   ├── data_generation.py
│   ├── features.py
│   ├── train_backtest.py
│   ├── model.py
│   ├── serve.py
│   ├── monitoring.py
│   └── utils.py
├── artifacts/                 # outputs: metrics, plots, dashboard
├── Dockerfile.train
├── Dockerfile.serve
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
````

MIT License — feel free to adapt for your portfolio.
