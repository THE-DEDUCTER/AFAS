import json
from flask import Flask, render_template, request, jsonify
from ml.image_model import predict_image
from ml.sensor_model import predict_sensor
from ml.fusion import fuse_predictions
from ml.alert_engine import get_alert_payload

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get occupant profile
        disability_type = request.form.get('occupant_profile', 'none')
        
        # Parse sensor data if provided
        sensor_data_str = request.form.get('sensor_data', '{}')
        sensor_data = json.loads(sensor_data_str)
        
        # Get thermal mode flag
        thermal_mode = request.form.get('thermal_mode') == 'true'
        
        # Get Image if provided
        image_file = request.files.get('image')
        
        img_prob = None
        if image_file and image_file.filename != '':
            image_bytes = image_file.read()
            if thermal_mode:
                from ml.image_model import predict_thermal
                img_prob = predict_thermal(image_bytes)
            else:
                img_prob = predict_image(image_bytes)
            
        sensor_prob = None
        if sensor_data:
            sensor_prob = predict_sensor(sensor_data)
            
        if img_prob is None and sensor_prob is None:
            return jsonify({"error": "No data provided"}), 400
            
        # Fusion
        fusion_result = fuse_predictions(img_prob, sensor_prob, thermal_mode=thermal_mode)
        
        # Alert mapping
        alert_payload = get_alert_payload(
            is_fire=fusion_result['is_fire'],
            severity=fusion_result['severity'],
            disability_type=disability_type
        )
        
        return jsonify({
            "status": "success",
            "img_prob": img_prob,
            "sensor_prob": sensor_prob,
            "fusion": fusion_result,
            "alert": alert_payload
        })
        
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
