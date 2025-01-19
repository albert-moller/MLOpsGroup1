import wandb
import os
import time
import torch
from mlops_project.model import MobileNetV3  # Corrected import
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load environment variables from .env file
load_dotenv()  # This loads the variables from your .env file

# Function to load the model from the WandB artifact
def load_model(artifact_path):
    print("Starting the model loading process...")
    
    # Initialize the WandB API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    
    print("WandB API initialized successfully.")
    
    # Fetch the artifact from the artifact path provided
    artifact = api.artifact(artifact_path)
    print(f"Fetching artifact: {artifact_path}...")
    
    # Download the artifact to the local directory
    artifact.download(root="./models")  # or any directory where you'd like to download
    print("Artifact downloaded successfully.")
    
    # Load the model checkpoint from the downloaded files
    model_file = artifact.files()[0].name  # Assuming the checkpoint is the first file
    print(f"Model checkpoint file: {model_file}")
    
    # Dynamically determine the number of classes from the checkpoint
    checkpoint_path = f"./models/{model_file}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract number of classes if it exists in the checkpoint (or set a default value)
    num_classes = checkpoint.get("hyper_parameters", {}).get("num_classes", 38)  # Default to 38 if not found
    print(f"Number of classes determined: {num_classes}")
    
    # Load the configuration for the model
    cfg = OmegaConf.create({
        "model": {
            "model_name": "mobilenetv3_large_100",
            "pretrained": True,
            "num_classes": num_classes  # Use dynamic value
        },
        "optimizer": {
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }
    })
    print("Configuration loaded successfully.")
    
    # Load the model using the class method and pass the checkpoint
    print("Loading model from checkpoint...")
    model = MobileNetV3.load_from_checkpoint(checkpoint_path, cfg=cfg)
    
    # Model already handles classifier replacement, no need to do it here
    print("Model loaded successfully.")
    
    return model


# Function to test the model's speed
def test_model_speed():
    print("Starting the model speed test...")
    model = load_model(os.getenv("MODEL_NAME"))
    
    print("Model loaded successfully, starting inference tests...")
    
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 3, 224, 224))  # Using a batch of 1 with 3 channels and 224x224 input size
    end = time.time()
    
    time_taken = end - start
    print(f"Time taken for 100 inferences: {time_taken:.4f} seconds")
    
    # Assert that the model takes less than 1 second for 100 inferences (if appropriate)
    assert time_taken < 5, f"Test failed! Time taken was {time_taken:.5f} seconds, which is more than 5 seconds."

    print("Model speed test completed successfully.")

# Call the test function
if __name__ == "__main__":
    test_model_speed()






