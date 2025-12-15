import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = "artifacts/defect_detection_resnet_casting_data.pth"

# -------------------------
# 1. Load the model
# -------------------------
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: OK / Defective

    checkpoint = torch.load(MODEL_PATH, map_location="cuda")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# -------------------------
# 2. Preprocessing with ImageNet normalization
# -------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# 3. Run inference on an image
# -------------------------
def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs

# -------------------------
# 4. Main test flow
# -------------------------
if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")

    # Test 1: Random Image
    random_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(random_input)
    print("Random image test passed. Output:", out)
    
    # Test 2: Test on real images
    TEST_FOLDER = "test_samples"  # create and add 2–3 casting images

    if os.path.exists(TEST_FOLDER):
        for img_name in os.listdir(TEST_FOLDER):
            path = os.path.join(TEST_FOLDER, img_name)
            probs = predict(model, path)
            cls = torch.argmax(probs, dim=1).item()
            print(f"{img_name}: probs={probs}, predicted_class={cls}")
    else:
        print("No test_samples/ folder found. Skipping image tests.")
