import os
import io
from PIL import Image

MODEL_PATH = "models/yolov8_fire.pt"
_yolo_model = None

def load_image_model():
    global _yolo_model
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        
        if not os.path.exists("models"):
            os.makedirs("models")
            
        if not os.path.exists(MODEL_PATH):
            print("Downloading YOLOv8 Fire model from Hugging Face...")
            ckpt_path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
            import shutil
            shutil.copy(ckpt_path, MODEL_PATH)
            
        _yolo_model = YOLO(MODEL_PATH)
        return _yolo_model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None

def predict_image(image_bytes: bytes) -> float:
    """Returns probability of fire (0.0 to 1.0)"""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = load_image_model()
        if _yolo_model is None:
            return 0.0
            
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run YOLO inference
        results = _yolo_model(image, verbose=False)
        
        max_prob = 0.0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > max_prob:
                    max_prob = conf
                    
        return max_prob
    except Exception as e:
        print(f"Error predicting image: {e}")
        return 0.0

def predict_image_detailed(image_bytes: bytes) -> dict:
    """Returns rich detection dictionary with classes, boxes, and max confidence."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = load_image_model()
        if _yolo_model is None:
            return {"probability": 0.0, "detections": [], "model": "YOLOv8n"}
            
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        results = _yolo_model(image, verbose=False)
        
        max_prob = 0.0
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = _yolo_model.names.get(cls_id, f"Class {cls_id}")
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(float(c), 1) for c in box.xyxy[0].tolist()]
                })
                if conf > max_prob:
                    max_prob = conf
                    
        return {
            "probability": round(max_prob, 4),
            "detections": detections,
            "detections_count": len(detections),
            "model": "YOLOv8n"
        }
    except Exception as e:
        print(f"Error in detailed prediction: {e}")
        return {"probability": 0.0, "detections": [], "error": str(e)}

_mobilenet_model = None

def predict_image_mobilenet(image_bytes: bytes) -> float:
    """Alternative inference using MobileNetV2 transfer-learned checkpoint."""
    global _mobilenet_model
    try:
        import torch
        from torchvision import models, transforms
        import torch.nn as nn

        mb_path = "models/mobilenetv2_fire.pt"
        if not os.path.exists(mb_path):
            return 0.0

        if _mobilenet_model is None:
            mb = models.mobilenet_v2()
            mb.classifier[1] = nn.Linear(mb.classifier[1].in_features, 1)
            mb.load_state_dict(torch.load(mb_path, map_location='cpu'))
            mb.eval()
            _mobilenet_model = mb

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = _mobilenet_model(tensor)
            # Probability calculation (accounting for inverted ImageFolder alphabetical indexing)
            prob = 1.0 - torch.sigmoid(output).item()
        return round(float(prob), 4)
    except Exception as e:
        print(f"Error in MobileNetV2 prediction: {e}")
        return 0.0

def predict_thermal(image_bytes: bytes) -> float:
    """Simulates thermal imaging by detecting extreme bright heat signatures."""
    try:
        import numpy as np
        image = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to Grayscale (heat map)
        img_array = np.array(image)
        
        # Count pixels that are extremely bright (assumed to be fire/extreme heat)
        # This is a basic simulation of a FLIR camera isolating high-temperature signatures
        bright_pixels = np.sum(img_array > 245)
        total_pixels = img_array.size
        
        bright_ratio = bright_pixels / total_pixels
        
        # If more than 0.5% of the image is extreme heat, return a high confidence
        if bright_ratio > 0.005:
            return 0.99
        else:
            return 0.01
            
    except Exception as e:
        print(f"Error in thermal simulation: {e}")
        return 0.0
