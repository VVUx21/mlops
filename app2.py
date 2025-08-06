from fastapi import FastAPI, HTTPException
import joblib # type: ignore
import numpy as np # type: ignore
from pydantic import BaseModel, validator
import uvicorn
from pyngrok import ngrok # type: ignore
from typing import List
import os

# Load the trained model with error handling
try:
    model = joblib.load("car_price_model.pkl")
except FileNotFoundError:
    print("Error: car_price_model.pkl not found!")
    model = None

app = FastAPI(title="Car Price Prediction API", version="1.0.0")

class CarFeatures(BaseModel):
    features: List[float]  # More specific typing
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError('Features list cannot be empty')
        return v

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running!"}

@app.post("/predict")
def predict(data: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

def start_ngrok():
    url = ngrok.connect(8000).public_url
    print(f"Public URL: {url}")
    return url

if __name__ == "__main__":
    public_url = start_ngrok()
    print(f"Access API here: {public_url}/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)