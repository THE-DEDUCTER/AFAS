def fuse_predictions(img_prob: float = None, sensor_prob: float = None, thermal_mode: bool = False) -> dict:
    """Fuses predictions to decide if there is a fire and what the severity is."""
    final_prob = 0.0
    
    if img_prob is not None and sensor_prob is not None:
        # Calibrate sensor_prob to remove the ~0.35 baseline of normal rooms
        calibrated_sensor = max(0.0, (sensor_prob - 0.35) / 0.65)
        
        # We take the maximum of the two probabilities. This acts as an "OR" gate:
        # If either the camera OR the sensors are highly confident, the alarm triggers.
        # This prevents a 'blind' camera from watering down a spiked temperature sensor,
        # and vice-versa.
        final_prob = max(img_prob, calibrated_sensor)
    elif img_prob is not None:
        final_prob = img_prob
    elif sensor_prob is not None:
        # If camera is unavailable, trust the sensor model directly
        final_prob = sensor_prob
    else:
        final_prob = 0.0

    threshold = 0.5
    is_fire = final_prob >= threshold

    severity = "Low"
    if is_fire:
        if final_prob > 0.8:
            severity = "High"
        else:
            severity = "Medium"

    return {
        "is_fire": is_fire,
        "probability": round(final_prob, 4),
        "severity": severity
    }
