import torch
import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig

from model import MobileNetV3
from data import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # Set random seed.
    seed_everything(cfg.seed)
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
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    # Train model
    trainer.fit(model, train_loader, val_loader)
    # Test model
    trainer.test(model, test_loader)

if __name__ == "__main__":
    train()