"""Shared pytest fixtures for all test modules."""
import numpy as np
import pytest


@pytest.fixture
def tiny_image():
    """32x32 uint8 grayscale image with uniform speckle-like noise."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (32, 32), dtype=np.uint8)


@pytest.fixture
def tiny_image_float(tiny_image):
    """tiny_image normalised to float32 [0, 1]."""
    return tiny_image.astype(np.float32) / 255.0


@pytest.fixture
def tiny_mask():
    """32x32 binary mask with a 8x8 lesion region."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True
    return mask


@pytest.fixture
def empty_mask():
    """32x32 all-False mask (no lesion)."""
    return np.zeros((32, 32), dtype=bool)


@pytest.fixture
def mock_model():
    """Callable mock that returns a fixed all-0.9 probability map for any input."""
    class _MockModel:
        def __call__(self, x):
            import torch
            return torch.full_like(x, 0.9)
    return _MockModel()
