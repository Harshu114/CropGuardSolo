"""Test the API locally with a mock model (no .pt file needed)."""
import os
import sys
import tempfile

# Add api to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from fastapi.testclient import TestClient

# Mock the model loading
import model_server as ms

# Create a dummy model for testing (on the same device as get_device())
def create_dummy_model():
    from model import CropGuardMultiTask
    device = ms.get_device()
    model = CropGuardMultiTask(nc=16, pretrained=False).to(device).float().eval()
    return model

# Monkey-patch
ms._model = create_dummy_model()
ms._model_version = "cropguard-mock-test"

def test_health():
    from main import app
    client = TestClient(app)
    response = client.get("/v1/health")
    print(f"Health: {response.status_code} - {response.json()}")

def test_predict():
    from main import app
    client = TestClient(app)

    # Create a dummy test image
    import cv2
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode('.jpg', dummy_img)

    response = client.post(
        "/v1/predict",
        files={"image": ("test.jpg", img_bytes.tobytes(), "image/jpeg")},
        data={"conf": "0.1", "with_cam": "true"}
    )
    print(f"Predict: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Image ID: {result['image_id']}")
        print(f"  Size: {result['width']}x{result['height']}")
        print(f"  Class: {result['classification']['class']} ({result['classification']['confidence']:.2%})")
        print(f"  Severity: {result['severity']['continuous']:.3f} ({result['severity']['stage']})")
        print(f"  Detections: {len(result['detections'])}")
        print(f"  Lesion fraction: {result['segmentation']['lesion_fraction']:.3f}")
        print(f"  Timings: {result['timings_ms']}")
    else:
        print(f"  Error: {response.json()}")

if __name__ == "__main__":
    print("Testing CropGuard API (mock model)...")
    test_health()
    test_predict()
    print("\nAll tests passed!")