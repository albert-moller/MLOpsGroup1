import os
import pytest
import torch
import shutil
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from src.mlops_project.data import PlantVillageDataset, get_dataloaders
from omegaconf import OmegaConf

load_dotenv()

# Define test configuration for Hydra
TEST_CONFIG = OmegaConf.create(
    {
        "dataset": {
            "dataset_dir": "./test_dataset_dir",
            "raw_dir": "./test_dataset_dir/raw",
            "dataset_name": "mohitsingh1804/plantvillage",
        },
        "train_split": 0.7,
        "val_split": 0.2,
        "batch_size": 16,
    }
)


@pytest.fixture(scope="module")
def test_dataset_dir():
    """Fixture to set up and tear down a test dataset directory."""
    os.makedirs(TEST_CONFIG.dataset.raw_dir, exist_ok=True)
    yield TEST_CONFIG.dataset.raw_dir
    if os.path.exists(TEST_CONFIG.dataset.dataset_dir):
        shutil.rmtree(TEST_CONFIG.dataset.dataset_dir)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipif(
    not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"),
    reason="Kaggle credentials are not set in the environment.",
)
def test_download_and_count_images():
    """Test downloading the PlantVillage dataset and counting images."""
    dataset_name = TEST_CONFIG.dataset.dataset_name
    download_dir = TEST_CONFIG.dataset.dataset_dir

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Download and unzip dataset
    api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

    # Count images
    image_count = 0
    for root, _, files in os.walk(download_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_count += 1

    assert image_count > 0, "Dataset should contain images."
    print(f"Total number of images in the dataset: {image_count}")


def test_dataloaders():
    """Test the get_dataloaders function."""
    train_loader, val_loader, test_loader = get_dataloaders(TEST_CONFIG)

    # Check if loaders return data
    for loader, split_name in zip([train_loader, val_loader, test_loader], ["train", "val", "test"]):
        assert len(loader) > 0, f"{split_name.capitalize()} DataLoader should not be empty."
        for img, label in loader:
            assert img.shape[1:] == (3, 224, 224), "Image tensor shape is incorrect."
            assert label.dtype == torch.int64, "Label tensor should have dtype torch.int64."


def test_invalid_dataset():
    """Test handling of invalid datasets."""
    with pytest.raises(FileNotFoundError):
        PlantVillageDataset(raw_data_path=Path("./invalid_path"))
