import torch 
import os
from app.core.config import get_settings


def load_model():
    settings = get_settings()
    MODEL_PATH = settings.model_path
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Load model architecture
    from torchvision import models
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: Defective / OK
    
    # Load trained weights
    state_dict = torch.load(
        MODEL_PATH, 
        map_location="cuda" if torch.cuda.is_available() else "cpu", 
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return model
