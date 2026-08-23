import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Resolved from this file's location, not the working directory, so the script runs from
# anywhere rather than only from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "backend" / "app" / "defect_detection_resnet_casting_data.pth"
TEST_FOLDER = REPO_ROOT / "test_samples"

# Same automatic detection as backend/app/core/model.py: one device for the weights, the
# model and the input tensors, so this runs on CPU-only machines and actually uses the GPU
# on machines that have one.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. Load the model
# -------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. See the 'Model weights' section of the "
            f"README - the checkpoint is gitignored and must be downloaded separately."
        )

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: OK / Defective

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(DEVICE)
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
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs

# -------------------------
# 4. Main test flow
# -------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Weights: {MODEL_PATH}")
    model = load_model()
    print("Model loaded successfully.")

    # Test 1: Random Image
    random_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        out = model(random_input)
    print("Random image test passed. Output:", out)

    # Test 2: Test on real images
    if TEST_FOLDER.is_dir():
        for path in sorted(TEST_FOLDER.iterdir()):
            if not path.is_file():
                continue
            probs = predict(model, path)
            cls = torch.argmax(probs, dim=1).item()
            print(f"{path.name}: probs={probs}, predicted_class={cls}")
    else:
        print(f"No {TEST_FOLDER} folder found. Skipping image tests.")
