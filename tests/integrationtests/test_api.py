import pytest
import glob
from pathlib import Path
from fastapi.testclient import TestClient
from mlops_project.api import app

# Initialize test client.
client = TestClient(app)


@pytest.fixture
def sample_image():
    """Find an image in the sample directory."""
    sample_dir = Path("tests/sample")
    image_paths = glob.glob(f"{sample_dir}/**/*.*", recursive=True)
    for image_path in image_paths:
        if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            return image_path
    raise FileNotFoundError("No valid image found in the sample directory.")


def test_predict(sample_image):
    """Test the predict endpoint using a sample image."""
    with TestClient(app) as test_client:
        with open(sample_image, "rb") as img_file:
            response = test_client.post("/predict/", files={"file": ("sample_image.jpg", img_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    label = json_response["predicted_class"]
    assert isinstance(label, int)
    assert 0 <= label <= 38


def test_predict_invalid_file():
    """Test the predict endpoint with an invalid file type."""
    with TestClient(app) as test_client:
        response = test_client.post("/predict/", files={"file": ("test.txt", b"This is not an image.", "text/plain")})
    assert response.status_code == 500
    json_response = response.json()
    assert "error" in json_response


def test_predict_no_file():
    """Test the predict endpoint without providing a file."""
    with TestClient(app) as test_client:
        response = test_client.post("/predict/")
    assert response.status_code == 422
    json_response = response.json()
    assert "detail" in json_response


def test_predict_corrupted_image():
    """Test the predict endpoint with a corrupted image file."""
    corrupted_image = b"This is a corrupt image file"

    with TestClient(app) as test_client:
        response = test_client.post("/predict/", files={"file": ("corrupt_image.jpg", corrupted_image, "image/jpeg")})
    assert response.status_code == 500
    json_response = response.json()
    assert "error" in json_response
