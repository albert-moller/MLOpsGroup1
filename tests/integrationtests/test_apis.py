import pytest
import glob
from pathlib import Path
from fastapi.testclient import TestClient
from mlops_project.api import app as regular_app
from mlops_project.api_onnx import app as onnx_app

@pytest.fixture(scope="module", params=[regular_app, onnx_app])
def test_client(request):
    """Fixture to provide a TestClient for each API."""
    with TestClient(request.param) as client:
        yield client

@pytest.fixture
def sample_image():
    """Find an image in the sample directory."""
    sample_dir = Path("tests/sample")
    image_paths = glob.glob(f"{sample_dir}/**/*.*", recursive=True)
    for image_path in image_paths:
        if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            return image_path
    raise FileNotFoundError("No valid image found in the sample directory.")

def test_predict(test_client, sample_image):
    """Test the predict endpoint using a sample image."""
    with open(sample_image, "rb") as img_file:
        response = test_client.post("/predict/", files={"file": ("sample_image.jpg", img_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    label = json_response["predicted_class"]
    assert isinstance(label, int)
    assert 0 <= label <= 38

def test_predict_invalid_file(test_client):
    """Test the predict endpoint with an invalid file type."""
    response = test_client.post("/predict/", files={"file": ("test.txt", b"This is not an image.", "text/plain")})
    assert response.status_code == 500
    json_response = response.json()
    assert "error" in json_response

def test_predict_no_file(test_client):
    """Test the predict endpoint without providing a file."""
    response = test_client.post("/predict/")
    assert response.status_code == 422
    json_response = response.json()
    assert "detail" in json_response

def test_predict_corrupted_image(test_client):
    """Test the predict endpoint with a corrupted image file."""
    corrupted_image = b"This is a corrupt image file"

    response = test_client.post("/predict/", files={"file": ("corrupt_image.jpg", corrupted_image, "image/jpeg")})
    assert response.status_code == 500
    json_response = response.json()
    assert "error" in json_response