import os
import torch
import typer
import hydra
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig, OmegaConf
from mlops_project.model import MobileNetV3
from mlops_project.data import get_dataloaders
from mlops_project.config import MainConfig
from loguru import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.debug(f"Using device: {DEVICE}")

# Create a Typer app
app = typer.Typer()

@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../configs", config_name="config")
def visualize(cfg: DictConfig) -> None:
    cfg: MainConfig = OmegaConf.structured(cfg)

    # Load the trained model.
    model = MobileNetV3(cfg).to(DEVICE)
    model_path = "mobilenetv3_model.pth"
    if not os.path.exists(model_path):
        logger.info(f"Model file '{model_path}' not found. Please ensure the model is trained and saved.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() 
    logger.info(f"Loaded model from '{model_path}'.")

    # Load the test dataloader.
    _, _, test_loader = get_dataloaders(cfg)
    logger.info("Loaded test dataset for visualization.")

    # Collect predictions and true labels.
    all_preds = []
    all_labels = []
    all_images = []
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_images.extend(images.cpu())

    # Calculate confusion matrix.
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.model.num_classes)))
    class_wise_errors = cm.sum(axis=1) - cm.diagonal() 

    # Identify struggling classes (top 3 by error count).
    struggling_classes = sorted(
        range(len(class_wise_errors)),
        key=lambda x: class_wise_errors[x],
        reverse=True
    )[:3]
    logger.info(f"Classes the model struggles with: {struggling_classes}")

    # Transform for displaying images.
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(cfg.dataset.mean, cfg.dataset.std)],
                             std=[1 / s for s in cfg.dataset.std]),
        transforms.ToPILImage(),
    ])

    # Visualize examples for struggling classes.
    fig, axes = plt.subplots(len(struggling_classes), 5, figsize=(15, len(struggling_classes) * 4)) 
    for i, cls in enumerate(struggling_classes):
        cls_images = [(img, lbl, pred) for img, lbl, pred in zip(all_images, all_labels, all_preds) if lbl == cls and lbl != pred]
        for j, (img, lbl, pred) in enumerate(cls_images[:5]):
            ax = axes[i, j] if len(struggling_classes) > 1 else axes[j]
            img = inv_transform(img) 
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"True: {lbl}, Pred: {pred}")
    plt.tight_layout()
    plt.savefig("visualizations/struggling_classes.png")
    logger.info("Saved visualization to 'visualizations/struggling_classes.png'.")

@app.command()
def main():
    visualize()

if __name__ == "__main__":
    typer.run(main)