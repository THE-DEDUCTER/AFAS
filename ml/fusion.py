def fuse_predictions(img_prob: float, sensor_prob: float) -> dict:
    """
    Fuses probabilities and makes a final decision.
    img_prob: float (0 to 1) or None (if no image)
    sensor_prob: float (0 to 1) or None (if no sensor data)
    """
    if img_prob is not None and sensor_prob is not None:
        # Calibrate sensor_prob to remove the ~0.35 baseline of normal rooms
        calibrated_sensor = max(0.0, (sensor_prob - 0.35) / 0.65)
        
        # STRICT MULTI-MODAL FUSION:
        # Due to lack of indoor training data, the camera cannot be trusted to trigger 
        # an alarm entirely on its own (max contribution is 0.4). It requires at least 
        # some corroboration from the physical sensors (Temp/TVOC) to cross the 0.5 threshold.
        final_prob = (img_prob * 0.4) + (calibrated_sensor * 0.6)
    elif img_prob is not None:
        final_prob = img_prob
    elif sensor_prob is not None:
        calibrated_sensor = max(0.0, (sensor_prob - 0.35) / 0.65)
        final_prob = calibrated_sensor
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
