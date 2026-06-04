"""Lightweight U-Net wrapper for lesion segmentation.

ISP Characteristics (W6):
  C1 - image ndim: {2 (grayscale), 3 (HxWxC), other (invalid)}
  C2 - model: {real torch model, injected mock}
  C3 - threshold: {0.0, 0.5 (default), 1.0, out-of-range}
"""
import numpy as np


def build_model():
    """Build and return an untrained lightweight U-Net.

    Returns a torch.nn.Module suitable for breast-ultrasound segmentation.
    """
    raise NotImplementedError


def predict(model, image):
    """Run model inference and return a probability map.

    Args:
        model: torch.nn.Module or callable mock that accepts a (1,1,H,W) tensor
               and returns a (1,1,H,W) probability tensor.
        image: 2-D or 3-D numpy array (H x W) or (H x W x C).

    Returns:
        2-D float32 numpy array of shape (H, W) in [0, 1].

    Raises:
        ValueError: if image has fewer than 2 or more than 3 dimensions.
    """
    raise NotImplementedError


def postprocess_mask(prob_map, threshold=0.5):
    """Threshold probability map to binary segmentation mask.

    Args:
        prob_map: 2-D float array in [0, 1].
        threshold: binarisation cutoff in [0, 1].

    Returns:
        2-D bool numpy array.

    Raises:
        ValueError: if threshold is outside [0, 1].
    """
    raise NotImplementedError
