# src/predict_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from src.features import extract_features


app = FastAPI(title='Phishing Detector')


MODEL_PATH = os.path.join('models', 'rf_phish_pipeline.pkl')
if os.path.exists(MODEL_PATH):
model = joblib.load(MODEL_PATH)
else:
model = None


class URLItem(BaseModel):
url: str


@app.on_event("startup")
def startup_event():
global model
if model is None:
if not os.path.exists(MODEL_PATH):
raise RuntimeError('Model not found. Train model first using src/train.py')
model = joblib.load(MODEL_PATH)


@app.post('/predict')
def predict(item: URLItem):
url = item.url
if not isinstance(url, str) or len(url.strip()) == 0:
raise HTTPException(status_code=400, detail='Invalid URL')


feats = extract_features(url)
X = pd.DataFrame([feats])
prob = float(model.predict_proba(X)[0,1])
label = int(prob > 0.5)
return {"url": url, "phishing_prob": prob, "label": label}


@app.get('/')
def root():
return {"status": "ok", "model_loaded": model is not None}
