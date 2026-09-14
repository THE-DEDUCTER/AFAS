import os
import joblib
import pandas as pd

MODEL_PATH = "models/sensor_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_sensor_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Failed to load sensor model: {e}")
    return None, None

model, scaler = load_sensor_model()

def predict_sensor(sensor_data: dict) -> float:
    """Returns probability of fire (0.0 to 1.0) based on sensor data."""
    if model is None or scaler is None:
        # Fallback for UI testing based on temp > 50 as a simple mock
        temp = float(sensor_data.get('Temperature', 20))
        return 0.95 if temp > 50 else 0.05

    try:
        # Define the exact features used during training
        features = [
            'Temperature', 'Humidity', 'TVOC', 'eCO2', 
            'Raw H2', 'Raw Ethanol', 'Pressure', 'PM1.0', 'PM2.5'
        ]
        # Define realistic normal baselines for missing sensors to prevent anomaly false positives
        default_values = {
            'Temperature': 20.0,
            'Humidity': 50.0,
            'TVOC': 0.0,
            'eCO2': 400.0,
            'Raw H2': 12900.0,
            'Raw Ethanol': 19400.0,
            'Pressure': 939.0,
            'PM1.0': 0.0,
            'PM2.5': 0.0
        }
        
        # Ensure data is numeric and fill missing with realistic defaults
        data_df = pd.DataFrame([{f: float(sensor_data.get(f, default_values[f])) for f in features}])
        
        # Scale
        scaled_data = scaler.transform(data_df)
        
        # Predict probability of class 1 (Fire)
        prob = model.predict_proba(scaled_data)[0][1]
        return float(prob)
    except Exception as e:
        print(f"Error predicting sensor: {e}")
        return 0.0
