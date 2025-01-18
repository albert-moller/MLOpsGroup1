import os
import pytest
import torch
from torch.utils.data import Subset
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlops_project.data import PlantVillageDataset, get_transforms, get_dataloaders
from mlops_project.utils.download_dataset import authenticate_kaggle, download_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the sample dataset path
SAMPLE_DATASET_DIR = "./tests/sample"

@pytest.fixture
def mock_cfg():
    return OmegaConf.create({
        "dataset": {
            "dataset_dir": "./tests/mock_dataset",
            "raw_dir": "./tests/mock_dataset/raw",
            "dataset_name": "plant-village-dataset"
        },
        "train_split": 0.7, 
        "val_split": 0.2,   
        "batch_size": 16  
    })

def test_authenticate_kaggle_missing_env_vars():
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
        with pytest.raises(ValueError, match="Kaggle API credentials not found in .env file"):
            authenticate_kaggle()

def test_authenticate_kaggle_success():
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "valid_user", "KAGGLE_KEY": "valid_key"}):
        with patch("kaggle.KaggleApi") as mock_kaggle_api:
            mock_api_instance = MagicMock()
            mock_kaggle_api.return_value = mock_api_instance

            api = authenticate_kaggle()
            assert api is not None

def test_download_dataset_already_exists(mock_cfg):
    # Mock the file existence check and logging
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["file1", "file2"]), \
         patch("logging.Logger.info") as mock_log, \
         patch("mlops_project.utils.download_dataset.authenticate_kaggle") as mock_auth, \
         patch("kaggle.KaggleApi") as mock_kaggle_api:

        # Mock the authentication to return a mock API instance
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance
        mock_auth.return_value = mock_api_instance

        # Call the function under test
        download_dataset(mock_cfg)

        # Verify that the specific log message is present among the calls
        mock_log.assert_any_call(f"Dataset already prepared in: {mock_cfg.dataset.raw_dir}")

        # Ensure the Kaggle API's `dataset_download_files` is not called since the dataset exists
        mock_api_instance.dataset_download_files.assert_not_called()

def test_download_dataset_fresh_download(mock_cfg):
    
    raw_path = mock_cfg.dataset.raw_dir

    with patch("os.path.exists", side_effect=lambda path: path == raw_path), \
         patch("os.makedirs") as mock_makedirs, \
         patch("os.listdir", return_value=[]), \
         patch("kaggle.KaggleApi") as mock_kaggle_api, \
         patch("pathlib.Path.iterdir", return_value=[Path("train/class1"), Path("train/class2")]), \
         patch("mlops_project.utils.download_dataset.authenticate_kaggle") as mock_auth:
        
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance
        mock_auth.return_value = mock_api_instance

        download_dataset(mock_cfg)

        mock_makedirs.assert_any_call(mock_cfg.dataset.dataset_dir, exist_ok=True)
        mock_makedirs.assert_any_call(mock_cfg.dataset.raw_dir, exist_ok=True)

        mock_api_instance.dataset_download_files.assert_called_once_with(
            mock_cfg.dataset.dataset_name,
            path=mock_cfg.dataset.dataset_dir,
            unzip=True
        )

def test_sample_dataset_loading():
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
    invalid_dataset_dir = "./tests/data/invalid"
    # Ensure the invalid directory does not exist
    assert not os.path.exists(invalid_dataset_dir), "Invalid dataset directory should not exist."

    # Attempt to load the invalid dataset and expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        PlantVillageDataset(raw_data_path=Path(invalid_dataset_dir))

def test_dataset_initialization():
    dataset = PlantVillageDataset(raw_data_path=SAMPLE_DATASET_DIR)
    assert len(dataset) == 10 
    assert isinstance(dataset.class_to_idx, dict)


def test_dataset_len():
    dataset = PlantVillageDataset(raw_data_path=SAMPLE_DATASET_DIR)
    assert len(dataset) == len(dataset.image_paths)


def test_dataset_getitem():
    dataset = PlantVillageDataset(raw_data_path=SAMPLE_DATASET_DIR)
    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)

    transform = get_transforms()
    dataset_with_transform = PlantVillageDataset(raw_data_path=SAMPLE_DATASET_DIR, transform=transform)
    image, label = dataset_with_transform[0]
    assert image.shape == (3, 224, 224) 

def test_get_dataloaders(mock_cfg):
    with patch("mlops_project.data.download_dataset") as mock_download, \
         patch("mlops_project.data.random_split") as mock_split:
       
        dataset = PlantVillageDataset(raw_data_path=SAMPLE_DATASET_DIR)
        total_size = len(dataset)
        train_size = int(mock_cfg.train_split * total_size)
        val_size = int(mock_cfg.val_split * total_size)
        test_size = total_size - train_size - val_size

        train_dataset = Subset(dataset, list(range(train_size)))
        val_dataset = Subset(dataset, list(range(train_size, train_size + val_size)))
        test_dataset = Subset(dataset, list(range(train_size + val_size, total_size)))
        mock_split.return_value = (train_dataset, val_dataset, test_dataset)

        train_loader, val_loader, test_loader = get_dataloaders(mock_cfg)

        mock_download.assert_called_once_with(mock_cfg)

        assert len(train_loader.dataset) == train_size
        assert len(val_loader.dataset) == val_size
        assert len(test_loader.dataset) == test_size

        assert train_loader.batch_size == mock_cfg.batch_size
        assert val_loader.batch_size == mock_cfg.batch_size
        assert test_loader.batch_size == mock_cfg.batch_size