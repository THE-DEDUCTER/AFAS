import os
import torch
from torchvision import models, transforms
from PIL import Image
import io

MODEL_PATH = "models/mobilenetv2_fire.pt"

# Define transformations (must match training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load image model: {e}")
        return None

# Global instance
model = load_image_model()

def predict_image(image_bytes: bytes) -> float:
    """Returns probability of fire (0.0 to 1.0)"""
    if model is None:
        # Fallback for UI testing before model is trained (simple deterministic mock)
        return 0.9 if len(image_bytes) > 50000 else 0.1 
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Calculate brightness (average pixel value)
        from PIL import ImageStat
        stat = ImageStat.Stat(image.convert('L'))
        brightness = stat.mean[0]

        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            non_fire_prob = torch.sigmoid(output).item()
            prob = 1.0 - non_fire_prob # Invert because fire_images is class 0
            
            # Suppress false positives on Out-Of-Distribution dark indoor images
            # Fires emit light, so if the frame is very dark, penalize the probability.
            if brightness < 60:
                prob = prob * (brightness / 60.0)
                
        return prob
    except Exception as e:
        print(f"Error predicting image: {e}")
        return 0.0
