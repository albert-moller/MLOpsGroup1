import os
import pytest
import torch
from pathlib import Path
from src.mlops_project.data import PlantVillageDataset, get_transforms
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the sample dataset path
SAMPLE_DATASET_DIR = "./tests/sample"

def test_sample_dataset_loading():
    """Test loading and verifying the small dataset in tests/data/sample."""
    # Ensure the sample dataset exists
    assert os.path.exists(SAMPLE_DATASET_DIR), "Sample dataset directory does not exist."

    # Get image transformations
    transform = get_transforms()

    # Create an instance of the dataset with transformations
    dataset = PlantVillageDataset(raw_data_path=Path(SAMPLE_DATASET_DIR), transform=transform)

    # Check that the dataset has exactly 10 samples
    assert len(dataset) == 10, f"Expected 10 images, but found {len(dataset)}."

    # Iterate through the dataset and verify each sample
    for img, label in dataset:
        # Ensure the image is a torch.Tensor
        assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor."
        # Ensure the image shape is correct
        assert img.shape == (3, 224, 224), f"Image tensor shape is incorrect: {img.shape}."
        # Ensure the label is within range
        assert 0 <= label < len(dataset.class_to_idx), f"Label out of range: {label}."

def test_invalid_dataset():
    """Test handling of invalid datasets."""
    invalid_dataset_dir = "./tests/data/invalid"
    # Ensure the invalid directory does not exist
    assert not os.path.exists(invalid_dataset_dir), "Invalid dataset directory should not exist."
    
    # Attempt to load the invalid dataset and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        PlantVillageDataset(raw_data_path=Path(invalid_dataset_dir))
