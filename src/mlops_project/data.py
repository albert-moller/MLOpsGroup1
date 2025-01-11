
import os
import numpy as np
import hydra
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from typing import Tuple, Callable, Optional, List, Dict
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader, random_split

from utils.download_dataset import download_dataset

logger = logging.getLogger('Data')
logger.setLevel(logging.INFO)
      
class PlantVillageDataset(Dataset):
    """Custom Dataset for PlantVillage"""

    def __init__(self, raw_data_path: Path, transform: Optional[Callable] = None) -> None:
        self.raw_data_path = raw_data_path
        self.transform: Optional[Callable] = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {}
        self.load_and_store_data()

    def load_and_store_data(self) -> None:
        for class_name in sorted(os.listdir(self.raw_data_path)):
            class_dir = os.path.join(self.raw_data_path, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = len(self.class_to_idx)
                for image_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, label
    

# Function to define image transformations (Albumentations)
def get_transforms() -> A.Compose:
    """
    Define the image transformations for preprocessing.
    """
    transform = A.Compose([
        A.Resize(224, 224),  # Resize to (224, 224).
        A.RandomBrightnessContrast(p=0.1),  # Adjust brightness and contrast.
        A.HorizontalFlip(p=0.2),  # Random horizontal flip.
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.1),  # Slight shifts, scaling, and rotations
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ToTensorV2()  # Convert to PyTorch tensor
    ])
    return transform

def get_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for the PlantVillage dataset.

    Args:
        cfg (DictConfig): Hydra configuration object with experiment parameters.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    # Download dataset if not already downloaded.
    download_dataset(cfg)

    # Transformations
    transform = get_transforms()

    # Create the dataset
    dataset = PlantVillageDataset(raw_data_path=cfg.dataset.raw_dir, transform=transform)

    # Split the dataset
    total_size = len(dataset)
    train_size = int(cfg.train_split * total_size)
    val_size = int(cfg.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    # Log DataLoader info
    logger.info(f"Train Loader: {len(train_loader.dataset)} samples")
    logger.info(f"Validation Loader: {len(val_loader.dataset)} samples")
    logger.info(f"Test Loader: {len(test_loader.dataset)} samples")

if __name__ == "__main__":
    main()
