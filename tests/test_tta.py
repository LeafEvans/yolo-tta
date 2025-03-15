# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import pytest
import torch
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS_DIR


@pytest.fixture
def model():
    """Fixture to provide a new YOLO model instance for each test."""
    return YOLO("yolov8n.pt")


@pytest.mark.parametrize(
    "alpha, tau1, tau2, momentum",
    [
        (0.01, 1.1, 1.05, 0.99),  # Default values
        (0.001, 1.05, 1.02, 0.999),  # Conservative settings
        (0.05, 1.2, 1.1, 0.95),  # Aggressive settings
    ],
)
def test_tta_configuration(model, alpha, tau1, tau2, momentum):
    """Test TTA configuration with various parameter combinations."""
    # Enable TTA
    model.enable_tta()

    # Configure TTA with specific parameters
    model.configure_tta(alpha=alpha, tau1=tau1, tau2=tau2, momentum=momentum)

    # Verify parameters are correctly set
    assert hasattr(model.model, "tta_strategy"), "TTA strategy not found in model"
    assert model.model.tta_strategy.alpha == alpha, "Alpha parameter not set correctly"
    assert model.model.tta_strategy.tau1 == tau1, "Tau1 parameter not set correctly"
    assert model.model.tta_strategy.tau2 == tau2, "Tau2 parameter not set correctly"
    assert model.model.tta_strategy.momentum == momentum, "Momentum parameter not set correctly"


@pytest.mark.slow
def test_tta_inference(model):
    """Test that inference works correctly with TTA enabled."""
    # Use a standard test image
    img_path = str(Path(ASSETS_DIR) / "bus.jpg")
    if not os.path.exists(img_path):
        img_path = str(Path("ultralytics/assets") / "bus.jpg")

    # Run standard inference as baseline
    results_standard = model.predict(img_path, verbose=False)

    # Enable and configure TTA
    model.enable_tta()
    model.configure_tta(alpha=0.01, tau1=1.1, tau2=1.05)

    # Run inference with TTA
    results_tta = model.predict(img_path, verbose=False)

    # Verify results are produced
    assert results_standard is not None, "Standard inference failed"
    assert results_tta is not None, "TTA inference failed"

    # Verify both produce detection boxes
    assert hasattr(results_standard[0], "boxes"), "Standard results missing boxes"
    assert hasattr(results_tta[0], "boxes"), "TTA results missing boxes"


@pytest.mark.slow
def test_tta_adaptation():
    """Test that TTA adapts model across different distribution shifts."""
    model = YOLO("yolov8n.pt")
    model.enable_tta()
    model.configure_tta(alpha=0.02, tau1=1.1, tau2=1.05)

    # Test sequence of different images to trigger adaptation
    test_images = [str(Path(ASSETS_DIR) / "bus.jpg"), str(Path(ASSETS_DIR) / "zidane.jpg")]

    if not os.path.exists(test_images[0]):
        test_images = [str(Path("ultralytics/assets") / "bus.jpg"), str(Path("ultralytics/assets") / "zidane.jpg")]

    # First inference triggers initial adaptation
    results1 = model.predict(test_images[0], verbose=False)

    # Ensure model can process different images after adaptation
    results2 = model.predict(test_images[1], verbose=False)

    assert len(results1) > 0, "Failed to process first image"
    assert len(results2) > 0, "Failed to process second image after adaptation"


def test_tta_disable(model):
    """Test that TTA can be properly disabled."""
    # Enable TTA
    model.enable_tta()
    assert hasattr(model.model, "tta_strategy"), "TTA was not enabled"

    # Disable TTA
    model.disable_tta()

    # Check if properly disabled
    assert not hasattr(model.model, "tta_strategy") or not model.model.tta_enabled, "TTA was not properly disabled"


def test_tta_chaining(model):
    """Test that TTA methods can be chained."""
    # Test method chaining
    result = model.enable_tta().configure_tta(alpha=0.02)

    # Verify chaining returns model
    assert result is model, "Method chaining does not return model instance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tta_device_compatibility():
    """Test TTA compatibility with different devices."""
    # Test on GPU if available
    model = YOLO("yolov8n.pt").to("cuda")
    model.enable_tta()

    img_path = str(Path(ASSETS_DIR) / "bus.jpg")
    if not os.path.exists(img_path):
        img_path = str(Path("ultralytics/assets") / "bus.jpg")

    results = model.predict(img_path, verbose=False)
    assert results is not None, "TTA failed on CUDA device"


def test_tta_validation():
    """Test that TTA parameter validation works."""
    model = YOLO("yolov8n.pt")
    model.enable_tta()

    # Test invalid parameters
    with pytest.raises((ValueError, AssertionError, TypeError)):
        model.configure_tta(alpha=-0.1)  # Negative alpha should raise error

    with pytest.raises((ValueError, AssertionError, TypeError)):
        model.configure_tta(tau1=0.9)  # tau1 < 1.0 should raise error

    with pytest.raises((ValueError, AssertionError, TypeError)):
        model.configure_tta(tau2=0.9)  # tau2 < 1.0 should raise error

    with pytest.raises((ValueError, AssertionError, TypeError)):
        model.configure_tta(momentum=1.1)  # momentum > 1.0 should raise error


if __name__ == "__main__":
    pytest.main(["-v", __file__])
