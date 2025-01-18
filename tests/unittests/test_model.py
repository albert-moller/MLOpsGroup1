import os
import torch
import tempfile
import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from hydra import initialize, compose

from mlops_project.model import MobileNetV3

# Load Hydra configuration.
with initialize(version_base=None, config_path="../../configs", job_name="test_model"):
    cfg = compose(config_name="config")


def test_model_initialization():
    model = MobileNetV3(cfg)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "forward")


def test_model_forward_pass_batch():
    model = MobileNetV3(cfg)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    assert y.shape == (8, 38), "Output shape mismatch for batch input"


def test_model_invalid_input():
    model = MobileNetV3(cfg)
    x = torch.randn(1, 1, 224, 224)
    with pytest.raises(RuntimeError, match="expected input\\[1, 1, 224, 224\\] to have 3 channels"):
        model(x)


@pytest.mark.parametrize(
    "step_name, log_keys, returns_loss",
    [
        ("training_step", ["train_loss", "train_accuracy"], True),
        ("validation_step", ["val_loss", "val_accuracy"], False),
        ("test_step", ["test_loss", "test_accuracy"], False),
    ],
)
def test_model_steps(step_name, log_keys, returns_loss):
    # Prepare the model.
    model = MobileNetV3(cfg)
    model.prepare_model_for_finetuning()
    model.log = MagicMock()

    # Verify model.
    assert hasattr(model.model, "classifier")
    assert model.model.classifier.out_features == cfg.model.num_classes
    assert all(p.requires_grad for p in model.model.classifier.parameters())

    # Prepare dummy data.
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, cfg.model.num_classes, (4,))
    batch = (x, labels)

    # Call the step function dynamically
    step_fn = getattr(model, step_name)
    loss = step_fn(batch, 0)

    # Verify loss.
    if returns_loss:
        assert isinstance(loss, torch.Tensor), f"{step_name} should return a loss tensor"
        assert loss.ndim == 0, "Loss should be a scalar tensor"
        model.log.assert_any_call(log_keys[0], loss)
    else:
        assert loss is None
        logged_loss = next(arg[0][1] for arg in model.log.call_args_list if arg[0][0] == log_keys[0])
        assert isinstance(logged_loss, torch.Tensor), f"{step_name} should log a loss tensor"

    # Compute expected accuracy.
    with torch.no_grad():
        outputs = model(x)
        expected_accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()

    # Verify accuracy logging.
    logged_accuracy = next(arg[0][1] for arg in model.log.call_args_list if arg[0][0] == log_keys[1])
    assert logged_accuracy == expected_accuracy, f"{step_name} logged accuracy does not match expected accuracy"


def test_model_saving_loading():
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


def test_invalid_model_config():
    invalid_cfg = OmegaConf.create({"invalid_param": True})
    with pytest.raises(ConfigAttributeError, match="Missing key model"):
        MobileNetV3(invalid_cfg)


def test_compute_accuracy():
    preds = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )
    targets = torch.tensor([1, 0, 1])
    accuracy = MobileNetV3.compute_accuracy(preds, targets)
    assert accuracy == pytest.approx(2 / 3, 0.01)

def test_configure_optimizers():
    model = MobileNetV3(cfg)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.defaults["lr"] == cfg.optimizer.optimizer.params.lr

def test_load_model():
    model = MobileNetV3(cfg)
    loaded_model = model.load_model()
    assert isinstance(loaded_model, torch.nn.Module)
    assert loaded_model.num_classes == cfg.model.num_classes
    assert loaded_model.default_cfg["input_size"] == (3, 224, 224)
