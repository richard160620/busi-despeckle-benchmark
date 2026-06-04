"""BUSI dataset loader: image/mask pairing, label extraction, validation.

ISP Characteristics (W6):
  C1 - root path: {exists, missing}
  C2 - folder structure: {complete, missing subfolder}
  C3 - image/mask pair: {matched, size mismatch}
"""
import pathlib

import numpy as np
import skimage.io as skio

_LABELS = ("benign", "malignant", "normal")


def load_busi(root):
    """Load BUSI dataset from root directory.

    Returns list of (image, mask, label) tuples where
    label in {'benign', 'malignant', 'normal'}.

    Raises:
        FileNotFoundError: if root or a required subfolder is missing.
        ValueError: if any image/mask pair has mismatched spatial dimensions.

    Graph Coverage edges (W7):
    [START] -> check_root -> [root_missing? -> FileNotFoundError] -> iterate_labels
    iterate_labels -> [label_dir_missing? -> FileNotFoundError] -> load_pairs
    load_pairs -> [size_mismatch? -> ValueError] -> [END]
    """
    root = pathlib.Path(root)

    # Edge: check_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records = []

    # Edge: iterate_labels
    for label in _LABELS:
        label_dir = root / label

        # Edge: label_dir_missing
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label subdirectory: {label_dir}")

        image_paths = sorted(
            p for p in label_dir.glob("*.png") if "_mask" not in p.name
        )

        # Edge: load_pairs
        for img_path in image_paths:
            mask_path = label_dir / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue

            image = skio.imread(str(img_path))
            mask = skio.imread(str(mask_path))

            # Edge: size_mismatch
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Image/mask size mismatch for {img_path.name}: "
                    f"{image.shape[:2]} vs {mask.shape[:2]}"
                )

            records.append((image, mask, label))

    return records
