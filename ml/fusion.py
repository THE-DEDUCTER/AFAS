def fuse_predictions(img_prob: float, sensor_prob: float):
    # Simple weighted average
    return (img_prob * 0.5) + (sensor_prob * 0.5)
