# 🛒 MLOps Multi-Store Forecasting

An **end-to-end MLOps project** for **retail sales forecasting across multiple stores & SKUs**.  
This repository demonstrates how to design, build, and deploy a **production-grade ML pipeline** including:

- 📊 **Feature Engineering** with rolling aggregates & lag-based predictors  
- 🔁 **Rolling Backtesting** for robust model evaluation  
- 🌲 **Random Forest Regressor (baseline)** – extendable to advanced models (XGBoost, LSTM, Transformer)  
- ⚡ **FastAPI Model Serving** with REST endpoints  
- 📈 **Streamlit Dashboard** for interactive forecasting exploration  
- 🐳 **Dockerized Workflows** for training & serving pipelines  
- 🔧 **MLOps Best Practices**: experiment tracking, artifact management, reproducible builds  

---

**🚀 Project Structure**
```bash
mlops-multi-store-forecasting/
│
├── data/                # Input datasets (per store & SKU)
├── artifacts/           # Trained models, metrics, logs
│
├── src/
│   ├── features.py      # Feature engineering pipeline
│   ├── train_backtest.py # Rolling backtesting & training
│   ├── serve.py         # FastAPI inference service
│   ├── dashboard.py     # Streamlit dashboard for exploration
│   └── utils.py         # Helper functions
│
├── Dockerfile.train     # Docker image for training jobs
├── Dockerfile.serve     # Docker image for serving API
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
---
**⚡ Quickstart**
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
   python -m src.train_backtest --data_dir data/ --out artifacts/
   ```
4. Serve a saved model (registry/v1):
   ```bash
   cd src && uvicorn serve:app --host 0.0.0.0 --port 8000
   ```
   Endpoint: POST http://127.0.0.1:8000/predict

   Example request:
   {
     "store_id": 1,
     "sku": 42,
     "date": "2025-09-01",
     "features": {...}
   }
5. Run monitoring job (writes `artifacts/monitoring.json`):
   ```bash
   python src/monitoring.py --data_dir data/ --models_dir registry/ --out artifacts/monitoring.json
   ```

6. Explore with Streamlit Dashboard
   ```bash
   streamlit run src/dashboard.py
   ```
**🐳 Dockerized Workflows**
1. Training
   ```bash
   docker build -f Dockerfile.train -t mlops-forecast-train:latest .
   docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/artifacts:/app/artifacts mlops-forecast-train:latest
   ```
2. Serving
   ```bash
   docker build -f Dockerfile.serve -t mlops-forecast-serve:latest .
   docker run -p 8000:8000 mlops-forecast-serve:latest
   ```
---
**🛠️ Tech Stack**  
- Python (pandas, scikit-learn, numpy)  
- FastAPI for model serving  
- Streamlit for dashboarding  
- Docker for containerized workflows  
- MLflow / JSON Artifacts for tracking experiments (extensible)  
---
**🔮 Future Scope**  
✅ Integrate XGBoost / LightGBM / Deep Learning models  
✅ Add MLflow experiment tracking  
✅ CI/CD with GitHub Actions  
✅ Kubernetes deployment for scalable inference  

---
**🤝 Contributing**  
PRs are welcome! For major changes, please open an issue first to discuss what you’d like to change.  

---
**👨‍💻 Author**  
Kshitij Maurya  
Data Scientist | AI/ML Engineer | Product Thinker  
