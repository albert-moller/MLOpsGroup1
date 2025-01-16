import torch
import uvicorn
from hydra import compose, initialize
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
from torchvision import transforms

from mlops_project.model import MobileNetV3


# Define the lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model and configurations on startup and shutdown."""
    global model, transform, device

    # Load Hydra configuration.
    initialize(version_base=None, config_path="../../configs", job_name="fastapi_app")
    cfg = compose(config_name="config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model.
    model = MobileNetV3(cfg)
    model.load_state_dict(torch.load("models/mobilenetv3_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Define transformations.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    yield

    del model, transform, device

# Initialize the FastAPI app.
app = FastAPI(lifespan=lifespan)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Perform inference on the uploaded image."""
    try:
        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device) 

        # Perform inference.
        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax(dim=1).item()

        return JSONResponse(content={
            "predicted_class": predicted_class
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def main():
    """Run the FastAPI server."""
    uvicorn.run("mlops_project.api:app", host="0.0.0.0", port=8000, reload=True)

