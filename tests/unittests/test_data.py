import os
import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlops_project.data import PlantVillageDataset, get_transforms
from mlops_project.utils.download_dataset import download_dataset, authenticate_kaggle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the sample dataset path
SAMPLE_DATASET_DIR = "./tests/sample"


def test_missing_env_vars():
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
        with pytest.raises(ValueError, match="Kaggle API credentials not found in .env file"):
            authenticate_kaggle()


@patch("kaggle.api.kaggle_api_extended.KaggleApi")
def test_download_dataset(mock_kaggle_api, tmp_path):
    # Convert the configuration dictionary to a DictConfig
    cfg = OmegaConf.create(
        {
            "dataset": {
                "dataset_dir": str(tmp_path / "dataset"),
                "raw_dir": str(tmp_path / "raw"),
                "dataset_name": "plant-village-dataset",
            }
        }
    )

    mock_api = MagicMock()
    mock_kaggle_api.return_value = mock_api

    # Mock environment variables
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"}):
        download_dataset(cfg)

    mock_api.authenticate.assert_called_once()
    mock_api.dataset_download_files.assert_called_with(
        cfg.dataset.dataset_name, path=cfg.dataset.dataset_dir, unzip=True
    )
    assert os.path.exists(cfg.dataset.raw_dir)


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
