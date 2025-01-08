import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import shutil


# Define the paths for raw and processed data
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


# Function to download dataset using Kaggle API
def download_plantvillage_data():
    """
    Downloads and unzips the PlantVillage dataset if it doesn't already exist.
    """
    if not RAW_DATA_PATH.exists():
        print(f"Creating directory {RAW_DATA_PATH}")
        RAW_DATA_PATH.mkdir(parents=True)

    dataset_path = RAW_DATA_PATH / "plantvillage"
    
    if not dataset_path.exists():
        print("Downloading the dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()

        # Download and unzip PlantVillage dataset
        api.dataset_download_files('mohitsingh1804/plantvillage', path=str(RAW_DATA_PATH), unzip=True)
        print("Dataset downloaded and unzipped.")
    else:
        print("Dataset already exists.")


# Function to define image transformations (Albumentations)
def get_transforms():
    """
    Define the image transformations for preprocessing.
    """
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),  # Random crop
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast adjustments
        A.Rotate(limit=30, p=0.5),  # Random rotation
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),  # ImageNet normalization
        ToTensorV2()  # Convert to PyTorch tensor
    ])
    return transform


# Custom Dataset class to load images
class MyDataset(Dataset):
    def __init__(self, data_path: Path, transform=None):
        """
        Initialize dataset with path to images and transformations.
        """
        self.data_path = data_path
        self.transform = transform
        # Get all image file paths (assumes images are in subfolders)
        self.image_paths = list(self.data_path.glob('**/*.jpg'))  # Modify this for other image formats like .png, .jpeg
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_path}. Please check the folder structure.")
        
        # Assuming folder names are the labels
        self.labels = [p.parent.name for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get image path and label
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformations if defined
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        
        return img, label


# Function to preprocess and save images
def preprocess(raw_data_path: Path, output_folder: Path):
    """
    Preprocess the raw data and save it to the output folder.
    """
    # Check if output folder exists, if not, create it
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    
    # Create subfolders for train and validation
    train_folder = output_folder / "train"
    val_folder = output_folder / "val"
    
    if not train_folder.exists():
        train_folder.mkdir()
    
    if not val_folder.exists():
        val_folder.mkdir()

    # Define transformations
    transform = get_transforms()

    # Create dataset
    dataset = MyDataset(raw_data_path, transform=transform)

    # Split data (80% for training, 20% for validation)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    
    # Split data into training and validation (no need for slicing, just split by index)
    train_data = dataset.image_paths[:train_size]
    val_data = dataset.image_paths[train_size:]

    # Save images to processed folder
    print("Processing and saving train images...")
    for i, img_path in enumerate(train_data):
        img = Image.open(img_path).convert("RGB")
        img_save_path = train_folder / f"img_{i}.jpg"
        img.save(img_save_path)  # Save the image after transformation
    
    print("Processing and saving validation images...")
    for i, img_path in enumerate(val_data):
        img = Image.open(img_path).convert("RGB")
        img_save_path = val_folder / f"img_{i}.jpg"
        img.save(img_save_path)  # Save the image after transformation

    print(f"Training data saved to {train_folder}")
    print(f"Validation data saved to {val_folder}")


# Main function to run the entire preprocessing process
def main():
    print(f"Starting preprocessing for raw data in {RAW_DATA_PATH}...")
    
    # Step 1: Download data if it doesn't exist
    download_plantvillage_data()

    # Step 2: Preprocess data
    preprocess(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()



