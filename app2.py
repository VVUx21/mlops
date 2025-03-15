from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

# Load the trained model
model = joblib.load("car_price_model.pkl")

app = FastAPI()

class CarFeatures(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running!"}

@app.post("/predict")
def predict(data: CarFeatures):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

def start_ngrok():
    url = ngrok.connect(8000).public_url
    print(f"Public URL: {url}")
    return url

if __name__ == "__main__":
    public_url = start_ngrok()
    print(f"Access API here: {public_url}/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
