import numpy as np
import typer
from hydra import initialize, compose
from mlops_project.data import get_dataloaders

# Create a Typer app
app = typer.Typer()


def main():
    # Load Hydra configuration.
    with initialize(version_base=None, config_path="../../configs", job_name="test_model"):
        cfg = compose(config_name="config")
    # Initialize dataloaders.
    train_loader, validation_loader, test_loader = get_dataloaders(cfg)
    # Compute and print dataset statistics.
    for name, loader in [("Train", train_loader), ("Validation", validation_loader), ("Test", test_loader)]:
        dataset = loader.dataset
        # Compute class distribution
        labels = [label for _, label in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        # Find most and least frequent classes
        most_frequent = max(class_distribution.items(), key=lambda x: x[1])
        least_frequent = min(class_distribution.items(), key=lambda x: x[1])
        # Print dataset statistics.
        print(f"Plant Village dataset: {name}")
        print(f"Number of images: {len(dataset)}")
        print(f"Image shape: {dataset[0][0].shape}")
        print(f"Most frequent class: {most_frequent[0]} ({most_frequent[1]} samples)")
        print(f"Least frequent class: {least_frequent[0]} ({least_frequent[1]} samples)\n")


if __name__ == "__main__":
    typer.run(main)
