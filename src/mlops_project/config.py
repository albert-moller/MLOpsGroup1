from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_dir: Path
    raw_dir: Path
    train_folder: str
    val_folder: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclass
class ModelConfig:
    model_name: str
    num_classes: int
    pretrained: bool

    def __post_init__(self):
        if not isinstance(self.num_classes, int) or self.num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer.")
        if not isinstance(self.pretrained, bool):
            raise ValueError("Pre-trained must be a boolean.")


@dataclass
class OptimizerParams:
    lr: float
    weight_decay: float

    def __post_init__(self):
        if not (0 < self.lr <= 1):
            raise ValueError("Learning rate must be a float between 0 and 1.")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be a non-negative float.")


@dataclass
class OptimizerConfig:
    type: str
    params: OptimizerParams

    def __post_init__(self):
        if self.type not in ["Adam", "SGD", "RMSprop"]:
            raise ValueError(f"Optimizer must be one of ['Adam', 'SGD', 'RMSprop'], got '{self.type}'.")


@dataclass
class WandbConfig:
    project: str
    run_name: str

    def __post_init__(self):
        if not self.project or not isinstance(self.project, str):
            raise ValueError("Wandb project must be a non-empty string.")
        if not self.run_name or not isinstance(self.run_name, str):
            raise ValueError("Wandb run name must be a non-empty string.")


@dataclass
class MainConfig:
    defaults: list
    experiment_name: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    train_split: float
    val_split: float
    test_split: float
    seed: int
    precision: int
    monitor: str
    patience: int
    mode: str
    wandb: WandbConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig

    def __post_init__(self):
        total_split = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total_split <= 1.01):
            raise ValueError("Train split, Validation split and test split must equate to 1.")
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be either min or max.")
        if self.precision not in [16, 32]:
            raise ValueError("Precision must be 16 or 32.")
