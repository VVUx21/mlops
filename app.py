from flask import Flask, request, jsonify # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
from flask_cors import CORS # type: ignore

model = joblib.load("car_price_model.pkl")

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return "Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
