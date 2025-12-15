from fastapi import HTTPException
import io
import torch
from PIL import Image
import torchvision.transforms as transforms


# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Preprocess image bytes for model inference.
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        Preprocessed tensor ready for model input
        
    Raises:
        HTTPException: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, "Invalid image file")
    
    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)

    return tensor
