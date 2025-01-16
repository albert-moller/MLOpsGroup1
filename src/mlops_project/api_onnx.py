import numpy as np
import uvicorn
from onnxruntime import InferenceSession
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
from torchvision import transforms

# Define the lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model and configurations on startup and shutdown."""
    global model, transform

    # Load the ONNX model.
    onnx_path = "models/mobilenetv3_model.onnx"
    model = InferenceSession(onnx_path)

    # Define transformations.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    yield

    del model, transform


# Initialize the FastAPI app.
app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Perform inference on the uploaded image."""
    global transform
    try:
        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).numpy()

        # Run ONNX inference
        inputs = {model.get_inputs()[0].name: image}
        outputs = model.run(None, inputs)
        predicted_class = int(np.argmax(outputs[0]))

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def main():
    """Run the FastAPI server."""
    uvicorn.run("mlops_project.api:app", host="0.0.0.0", port=8000, reload=True)
