import os
import torch
import hydra
import wandb
import typer
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from mlops_project.model import MobileNetV3
from mlops_project.data import get_dataloaders
from mlops_project.config import MainConfig
from loguru import logger
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.debug(f"Using device: {DEVICE}")

# Create a Typer app
app = typer.Typer()


@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    cfg: MainConfig = OmegaConf.structured(cfg)
    # Set random seed.
    seed_everything(cfg.seed)
    # Load Wandb API key.
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.info("Wandb API key not found in .env file or environment variables.")
        return
    wandb.login(key=wandb_api_key)

    # Initialize Wandb for evaluation
    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.experiment_name}_evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=f"{cfg.experiment_name}_evaluation")

    # Load dataloaders (only test_loader is needed)
    _, _, test_loader = get_dataloaders(cfg)

    # Load the trained model
    model = MobileNetV3(cfg).to(DEVICE)
    model_path = "models/mobilenetv3_model.pth"
    if not os.path.exists(model_path):
        logger.info(f"Model file '{model_path}' not found. Please ensure the model is trained and saved.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    logger.info(f"Loaded model from '{model_path}'.")

    # Define the Trainer for evaluation
    trainer = Trainer(logger=wandb_logger, precision=cfg.precision, log_every_n_steps=10)

    # Evaluate the model on the test dataset
    logger.info("Starting model evaluation on the test dataset.")
    test_metrics = trainer.test(model, test_loader)

    # Log evaluation metrics to Wandb
    wandb.log({"test_metrics": test_metrics})
    logger.info(f"Evaluation metrics: {test_metrics}")

    # Finish Wandb session
    wandb.finish()


@app.command()
def main():
    evaluate()


if __name__ == "__main__":
    typer.run(main)
