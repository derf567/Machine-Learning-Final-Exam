from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load pre-trained models and encoders
yield_model = joblib.load('yield_prediction_model.pkl')
yield_scaler = joblib.load('yield_scaler.pkl')
disease_model = joblib.load('disease_prediction_model.pkl')
irrigation_model = joblib.load('irrigation_model.pkl')
farming_insights_model = joblib.load('farming_insights_model.pkl')
farming_insights_scaler = joblib.load('farming_insights_scaler.pkl')
soil_type_encoder = joblib.load('soil_type_encoder.pkl')
crop_type_encoder = joblib.load('crop_type_encoder.pkl')

# Route to serve HTML
@app.route('/')
def index():
    return render_template('index.html')

# API routes
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.json
    rainfall = float(data['rainfall'])
    temperature = float(data['temperature'])
    input_data = np.array([[rainfall, temperature]])
    input_scaled = yield_scaler.transform(input_data)
    prediction = yield_model.predict(input_scaled)[0]
    return jsonify({'prediction': round(prediction, 2)})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    humidity = float(data['humidity'])
    temperature = float(data['temperature'])
    input_data = np.array([[humidity, temperature]])
    prediction = disease_model.predict(input_data)[0]
    risk = "High" if prediction == 1 else "Low"
    return jsonify({'risk': risk})

@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    data = request.json
    soil_moisture = float(data['soil_moisture'])
    rain_forecast = float(data['rain_forecast'])
    input_data = np.array([[soil_moisture, rain_forecast]])
    prediction = irrigation_model.predict(input_data)[0]
    advice = "Irrigate" if prediction == 1 else "Do Not Irrigate"
    return jsonify({'advice': advice})

@app.route('/predict_farming_insights', methods=['POST'])
def predict_farming_insights():
    data = request.json
    soil_type = data['soil_type']
    planting_date = data['planting_date']
    crop_type = data['crop_type']

    # Process input
    planting_month = pd.to_datetime(planting_date).month

    # Encode categorical variables
    soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
    crop_type_encoded = crop_type_encoder.transform([crop_type])[0]

    # Simulate additional features (use real data collection in practice)
    rainfall = np.random.uniform(200, 1500)
    temperature = np.random.uniform(10, 40)

    # Prepare input data
    input_data = np.array([[soil_type_encoded, crop_type_encoded, planting_month, rainfall, temperature]])

    # Scale input
    input_scaled = farming_insights_scaler.transform(input_data)

    # Predict yield
    predicted_yield = farming_insights_model.predict(input_scaled)[0]

    # Months array for harvest month calculation
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    # Crop duration mapping (expand as needed)
    crop_duration_map = {
        'Wheat': 7,
        'Corn': 4,
        'Rice': 4,
        'Potato': 3,
        'Soybean': 5
    }

    # Calculate harvest month
    estimated_duration = crop_duration_map.get(crop_type, 5)
    harvest_month = (planting_month + estimated_duration - 1) % 12

    # Prepare result
    result = {
        'estimated_yield': round(predicted_yield, 2),
        'estimated_harvest_month': months[harvest_month],
        'soil_suitability': f"{soil_type.capitalize()} soil is suitable for {crop_type}"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
