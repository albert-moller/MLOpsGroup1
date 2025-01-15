import os
import torch
import tempfile
import pytest
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from hydra import initialize, compose

from mlops_project.model import MobileNetV3


def test_model_initialization():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    model = MobileNetV3(cfg)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "forward")


def test_model_forward_pass_batch():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    model = MobileNetV3(cfg)
    x = torch.randn(8, 3, 224, 224)  # Batch of 8 images
    y = model(x)
    assert y.shape == (8, 38), "Output shape mismatch for batch input"


def test_model_invalid_input():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    model = MobileNetV3(cfg)
    x = torch.randn(1, 1, 224, 224)
    with pytest.raises(RuntimeError, match="expected input\\[1, 1, 224, 224\\] to have 3 channels"):
        model(x)


def test_training_step():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    model = MobileNetV3(cfg)
    # Prepare the model for fine-tuning.
    model.prepare_model_for_finetuning()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()
    # Initialize dummy data.
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, cfg.model.num_classes, (4,))
    # Perform forward pass.
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, labels)
    # Perform backward pass and optimizer step.
    initial_params = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
    loss.backward()
    # Check gradients only for unfrozen layers.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.sum(param.grad.abs()) > 0, f"Gradient for {name} is zero"

    optimizer.step()
    # Verify that only unfrozen layers are updated.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param), f"Parameter {name} was not updated"


def test_model_saving_loading():
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config")
    # Intialize the model.
    model = MobileNetV3(cfg)
    # Prepare model for fine-tuning.
    model.prepare_model_for_finetuning()
    # Save the model.
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.pth")
        torch.save(model.state_dict(), save_path)
        # Load the model.
        loaded_model = MobileNetV3(cfg)
        loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
    # Compare parameters.
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)
