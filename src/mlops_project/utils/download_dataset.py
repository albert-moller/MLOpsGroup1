import os
import shutil
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
from dotenv import load_dotenv

logger = logging.Logger('Data')
logger.setLevel(logging.INFO)
      
def download_dataset(cfg: DictConfig) -> None:
    """
    Downloads and organizes a dataset from Kaggle

    Args:
        cfg (DictConfig): Hydra configuration object with dataset parameters.
    """
    cfg = cfg.dataset
    # Load environment variables
    load_dotenv()
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle API credentials not found in .env file")

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    # Authenticate the Kaggle API.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Set dataset paths.
    dataset_path = os.path.join(cfg.dataset_dir, "PlantVillage") 
    raw_path = cfg.raw_dir

    # Check if dataset has already been downloaded.
    if os.path.exists(raw_path) and len(os.listdir(raw_path)) > 1:
        logger.info(f"Dataset already prepared in: {raw_path}")
        return
    
    # Create dataset directories
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.raw_dir, exist_ok=True)

    # Download and unzip the dataset.
    if not os.path.exists(dataset_path):
        api.dataset_download_files(cfg.dataset_name, path=cfg.dataset_dir, unzip=True)
        logger.info(f"Dataset downloaded and extracted to: {cfg.raw_dir}")
    
    logger.info("Downloading and processing the Plant Village Dataset")
    # Move the contents of 'train' and 'val' folders into the raw directory.
    for folder_name in ["train", "val"]:
        source_dir = Path(os.path.join(dataset_path, folder_name))
        if source_dir.exists():
            for subfolder in source_dir.iterdir():
                if subfolder.is_dir():
                    destination = Path(os.path.join(raw_path, subfolder.name))
                    if not destination.exists():
                        shutil.move(str(subfolder), str(destination))
                        logger.info(f"Moved: {subfolder} -> {destination}")

    # Clean up the original dataset directory.
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    download_dataset(cfg)
    
if __name__ == "__main__":
    main()