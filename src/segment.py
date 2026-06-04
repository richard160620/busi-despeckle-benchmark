"""Lightweight U-Net wrapper for lesion segmentation.

ISP Characteristics (W6):
  C1 - image ndim: {2 (grayscale), 3 (HxWxC), other (invalid)}
  C2 - model: {real torch model, injected mock}
  C3 - threshold: {0.0, 0.5 (default), 1.0, out-of-range}
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight U-Net (2 encoder levels + bottleneck + 2 decoder levels)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _LightUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = _ConvBlock(1, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        # Interpolate before cat to handle odd spatial dims (skip-connection size mismatch)
        up2 = F.interpolate(self.up2(b), size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def build_model():
    """Build and return an untrained lightweight U-Net."""
    return _LightUNet()


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

    ISP C1 edges (W6):
      2-D  -> accepted
      3-D  -> accepted (take mean across channels)
      1-D  -> ValueError
      4-D+ -> ValueError
    """
    image = np.asarray(image)

    # Edge: invalid ndim < 2
    if image.ndim < 2:
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    # Edge: invalid ndim > 3
    if image.ndim > 3:
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")

    # Normalise to [0,1] float32
    img = image.astype(np.float32)
    if img.ndim == 3:
        # HxWxC -> grayscale by averaging channels
        img = img.mean(axis=-1)
    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape

    # Pad H and W to next multiple of 16 so encoder/decoder strides never mismatch
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H_pad,W_pad)

    with torch.no_grad():
        prob = model(tensor)  # (1,1,H_pad,W_pad)

    # Crop back to original spatial size before returning
    return prob.squeeze().numpy().astype(np.float32)[:h, :w]


def postprocess_mask(prob_map, threshold=0.5):
    """Threshold probability map to binary segmentation mask.

    Args:
        prob_map: 2-D float array in [0, 1].
        threshold: binarisation cutoff; must be in [0, 1].

    Returns:
        2-D bool numpy array.

    Raises:
        ValueError: if threshold is outside [0, 1].

    BVA thresholds (W6): 0.0 (all True), 0.5 (default), 1.0 (all False for p<1),
                         -0.1 (ValueError), 1.1 (ValueError).
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return np.asarray(prob_map) >= threshold
