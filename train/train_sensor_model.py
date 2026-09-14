import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'smoke_detection', 'smoke_telemetry.csv')
ALT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'smoke_detection', 'smoke_detection_iot.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'sensor_model.pkl')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def train_model():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset {DATA_PATH} not found. Please download the Kaggle Smoke Detection dataset and place it here.")
        # Sometimes the kaggle file is named smoke_detection_iot.csv
        if os.path.exists(ALT_DATA_PATH):
            print(f"Found {ALT_DATA_PATH}, using that instead.")
            csv_path = ALT_DATA_PATH
        else:
            return
    else:
        csv_path = DATA_PATH

    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Clean up column names and select features
    # Typical columns: 'Unnamed: 0', 'UTC', 'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 
    # 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT', 'Fire Alarm'
    
    # We rename to map our UI inputs directly to model features for simplicity
    feature_mapping = {
        'Temperature[C]': 'Temperature',
        'Humidity[%]': 'Humidity',
        'TVOC[ppb]': 'TVOC',
        'eCO2[ppm]': 'eCO2',
        'Raw H2': 'Raw H2',
        'Raw Ethanol': 'Raw Ethanol',
        'Pressure[hPa]': 'Pressure',
        'PM1.0': 'PM1.0',
        'PM2.5': 'PM2.5'
    }
    
    # Check if we need to rename
    if 'Temperature[C]' in df.columns:
        df = df.rename(columns=feature_mapping)

    features = [
        'Temperature', 'Humidity', 'TVOC', 'eCO2', 
        'Raw H2', 'Raw Ethanol', 'Pressure', 'PM1.0', 'PM2.5'
    ]
    target = 'Fire Alarm'

    # Filter out missing features if any
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Error: Missing columns in dataset: {missing}")
        return

    X = df[features]
    y = df[target]

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    xgb_preds = xgb_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test_scaled)

    xgb_acc = accuracy_score(y_test, xgb_preds)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"\nXGBoost Accuracy: {xgb_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Select the better model
    if xgb_acc >= rf_acc:
        print("\nSelecting XGBoost model.")
        best_model = xgb_model
        preds = xgb_preds
    else:
        print("\nSelecting Random Forest model.")
        best_model = rf_model
        preds = rf_preds

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nSaved model to {MODEL_SAVE_PATH} and scaler to {SCALER_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
