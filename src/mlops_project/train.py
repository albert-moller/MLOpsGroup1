import torch
import os
import hydra
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf
from model import MobileNetV3
from data import get_dataloaders
from loguru import logger

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {os.getenv('WANDB_API_KEY')}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.debug(f"Using device: {DEVICE}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # Set random seed.
    seed_everything(cfg.seed)
    # Initialize Wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True)  
    )
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment_name, log_model=True)

    # Load dataloaders.
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    # Intialize MobileNetV3 model.
    model = MobileNetV3(cfg).to(DEVICE)
    # Define callbacks.
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor=cfg.monitor,
        mode=cfg.mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.patience,
        mode=cfg.mode
    )
    # Define trainer.
    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        precision=cfg.precision,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )
    # Log initial model parameters
    wandb.watch(model, log="all", log_freq=100)
    # Train model
    logger.info(f"Starting training with {cfg.num_epochs} epochs and batch size {cfg.batch_size}")
    trainer.fit(model, train_loader, val_loader)
    # Test model
    trainer.test(model, test_loader)
    # Save model
    torch.save(model.state_dict(), "mobilenetv3_model.pth")
    artifact = wandb.Artifact(
        "mobilenetv3_model",
        type="model",
        description="A fine-tuned MobileNetV3 model for plant disease classification",
        metadata={"num_epochs": cfg.num_epochs, "batch_size": cfg.batch_size}
    )
    artifact.add_file("mobilenetv3_model.pth")
    wandb.run.log_artifact(artifact)
    logger.info("Model artifact logged to Wandb.")

    # Finish Wandb session
    wandb.finish()

if __name__ == "__main__":
    train()