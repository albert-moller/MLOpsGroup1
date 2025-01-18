from locust import HttpUser, task, between
from PIL import Image
import io


class ImageClassificationUser(HttpUser):
    """Simulates users making predictions with image uploads."""

    wait_time = between(1, 3)  # Random wait between requests

    def on_start(self):
        """Setup before tests begin - create a test image."""
        # Create a simple test image in memory
        img = Image.new("RGB", (224, 224), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        self.test_image_bytes = img_byte_arr.getvalue()

    @task
    def predict_image(self):
        """Task to test the prediction endpoint."""
        files = {"file": ("test_image.jpg", self.test_image_bytes, "image/jpeg")}

        with self.client.post("/predict/", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if "predicted_class" in json_response:
                        response.success()
                    else:
                        response.failure("Response missing predicted_class")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")
