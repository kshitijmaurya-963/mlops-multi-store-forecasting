"""FastAPI-based simple model server that loads latest model from registry and serves predictions.
Endpoint: POST /predict with JSON {"store_id":"store_000","sku":"sku_001","features":{...}}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import joblib, json, os

app = FastAPI(title='Forecast Serve')

class PredictRequest(BaseModel):
    store_id: str
    sku: str
    features: dict

def load_latest_model():
    reg = Path('registry')
    versions = sorted([p for p in reg.iterdir() if p.is_dir() and p.name.startswith('v')])
    if not versions:
        raise FileNotFoundError('No model in registry')
    latest = versions[-1]
    model = joblib.load(str(latest / 'model.joblib'))
    with open(latest / 'metadata.json') as f:
        meta = json.load(f)
    return model, meta

@app.on_event('startup')
def startup_event():
    try:
        app.state.model, app.state.meta = load_latest_model()
    except Exception as e:
        app.state.model = None
        app.state.meta = None

@app.post('/predict')
def predict(req: PredictRequest):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    X = np.array([list(req.features.values())])
    preds = model.predict(X)
    return {'prediction': float(preds[0]), 'meta': app.state.meta}
