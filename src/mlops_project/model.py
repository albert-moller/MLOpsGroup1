import hydra
import torch
import timm
import logging
import pytorch_lightning as pl
from torch import nn
from typing import Tuple
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)


class MobileNetV3(pl.LightningModule):
    """Implementation of the MobileNetV3 model using a pre-trained model from the TIMM framework."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the MobileNetV3 Model.

        Args:
            cfg (DictConfig): Config containing model and optimizer parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.model_cfg = self.cfg.model
        self.optimizer_cfg = self.cfg.optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.model = self.load_model()
        self.prepare_model_for_finetuning()

    def load_model(self) -> torch.nn.Module:
        """
        Initializes the MobileNetV3 Model using the TIMM framework.
        """
        model = timm.create_model(
            model_name=self.model_cfg.model_name,
            pretrained=self.model_cfg.pretrained,
            num_classes=self.model_cfg.num_classes,
        )
        return model

    def prepare_model_for_finetuning(self) -> None:
        """
        Freezes all layers except the output classification layer.
        Replaces the classification layer of the pre-trained
        model such that it accounts for the correct number
        of classes.
        """
        num_classes = self.cfg.model.num_classes
        # Freeze all layers.
        for param in self.model.parameters():
            param.requires_grad = False
        # Replace the output classification layer.
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            # Unfreeze the linear output layer.
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "classifier"):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            # Unfreeze the classification layer.
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"The selected model {self.model_cfg.model_name} does not have a linear classification layer."
            )

    @staticmethod
    def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.size(0) == 0 or target.size(0) == 0:
            raise ValueError("Predictions and targets must not be empty.")
        if pred.size(0) != target.size(0):
            raise ValueError("Predictions and targets must have the same batch size.")
        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
        return accuracy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Unpack images and labels.
        img, label = batch
        # Perform forward pass.
        y_pred = self(img)
        # Compute the training loss
        loss = self.criterion(y_pred, label)
        # Compute training accuracy.
        accuracy = MobileNetV3.compute_accuracy(y_pred, label)
        # Log train metrics.
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Unpack images and labels.
        img, label = batch
        # Perform forward pass.
        y_pred = self(img)
        # Compute the training loss
        loss = self.criterion(y_pred, label)
        # Compute training accuracy.
        accuracy = MobileNetV3.compute_accuracy(y_pred, label)
        # Log validation metrics.
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Unpack images and labels.
        img, label = batch
        # Perform forward pass.
        y_pred = self(img)
        # Compute the training loss
        loss = self.criterion(y_pred, label)
        # Compute training accuracy.
        accuracy = MobileNetV3.compute_accuracy(y_pred, label)
        # Log test metrics.
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Create optimizer dynamically based on optimizer config.
        optimizer_class = getattr(torch.optim, self.optimizer_cfg.optimizer.type)
        optimizer = optimizer_class(self.parameters(), **self.optimizer_cfg.optimizer.params)
        return optimizer


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Display the config object.
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    # Initialize model for fine-tuning
    model = MobileNetV3(cfg)
    # Calculate total number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}")
    # Calculate the number of trainable parameters
    finetune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters being fine-tuned: {finetune_params}")
    # Dummy input for testing.
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
