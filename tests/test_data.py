"""Tests for src/data.py — load_busi.

Testing methods applied:
  - Input Space Partitioning (W6): C1={exists,missing} x C2={complete,missing subfolder}
    x C3={matched,size mismatch}
  - Boundary Value Analysis: smallest valid dataset (1 image per label)
  - Graph Coverage (W7): all edges of load_busi control-flow graph
"""
import numpy as np
import pytest

from src.data import load_busi


# ---------------------------------------------------------------------------
# ISP: C1 – root path missing
# ---------------------------------------------------------------------------

class TestLoadBusiMissingRoot:
    def test_missing_root_raises_file_not_found(self, tmp_path):
        """Edge: check_root -> FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_busi(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# ISP: C2 – missing required subfolder
# ---------------------------------------------------------------------------

class TestLoadBusiMissingSubfolder:
    def test_missing_label_dir_raises_file_not_found(self, tmp_path):
        """Edge: iterate_labels -> FileNotFoundError (missing 'benign' dir)"""
        root = tmp_path / "busi"
        root.mkdir()
        (root / "malignant").mkdir()
        (root / "normal").mkdir()
        # 'benign' folder absent
        with pytest.raises(FileNotFoundError):
            load_busi(root)


# ---------------------------------------------------------------------------
# ISP: C3 – image/mask size mismatch
# ---------------------------------------------------------------------------

class TestLoadBusiSizeMismatch:
    def test_size_mismatch_raises_value_error(self, tmp_path):
        """Edge: load_pairs -> ValueError (H/W mismatch)"""
        import skimage.io as skio
        root = _make_minimal_busi_root(tmp_path)
        # Overwrite one mask with wrong size
        benign_dir = root / "benign"
        masks = sorted(benign_dir.glob("*_mask.png"))
        bad_mask = np.zeros((16, 32), dtype=np.uint8)  # wrong shape
        skio.imsave(str(masks[0]), bad_mask)
        with pytest.raises(ValueError):
            load_busi(root)


# ---------------------------------------------------------------------------
# Happy path: correct dataset returns correct structure
# ---------------------------------------------------------------------------

class TestLoadBusiHappyPath:
    def test_returns_list_of_triples(self, tmp_path):
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        assert isinstance(result, list)
        assert len(result) == 3  # one image per label
        for item in result:
            assert len(item) == 3

    def test_labels_are_valid(self, tmp_path):
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        labels = {item[2] for item in result}
        assert labels <= {"benign", "malignant", "normal"}

    def test_image_mask_same_shape(self, tmp_path):
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        for image, mask, _ in result:
            assert image.shape[:2] == mask.shape[:2]

    def test_image_is_numpy_array(self, tmp_path):
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        for image, mask, _ in result:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)


# ---------------------------------------------------------------------------
# BVA: boundary – single image per label (smallest valid dataset)
# ---------------------------------------------------------------------------

class TestLoadBusiBoundary:
    def test_single_image_per_label(self, tmp_path):
        """BVA: minimum-size dataset (1 image per label folder)."""
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mutation killers (W10 Syntax-Based Testing)
# ---------------------------------------------------------------------------

class TestDataMutationKillers:
    """W10: targeted tests to kill surviving mutmut mutants in src/data.py."""

    def test_missing_root_error_contains_root(self, tmp_path):
        """Kill M292: error message for missing root must contain 'root'."""
        with pytest.raises(FileNotFoundError, match="root"):
            load_busi(tmp_path / "does_not_exist")

    def test_missing_label_error_contains_label(self, tmp_path):
        """Kill M297: error for missing label dir must contain 'label'."""
        root = tmp_path / "busi"
        root.mkdir()
        (root / "malignant").mkdir()
        (root / "normal").mkdir()
        with pytest.raises(FileNotFoundError, match="label"):
            load_busi(root)

    def test_all_three_labels_loaded(self, tmp_path):
        """Kill M287: _LABELS mutation → 'malignant' missing → FileNotFoundError or wrong count."""
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        labels_found = {item[2] for item in result}
        assert "malignant" in labels_found   # kills M287 which mutates 'malignant'

    def test_image_files_not_masks(self, tmp_path):
        """Kill M299+M300: image list must exclude _mask files."""
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        # With M300 (inverted filter), only mask PNGs would be loaded → empty result
        assert len(result) == 3

    def test_images_are_different_from_masks(self, tmp_path):
        """Kill M300: images loaded should not be the mask arrays."""
        import skimage.io as skio
        root = _make_minimal_busi_root(tmp_path)
        result = load_busi(root)
        for image, mask, label in result:
            # Image was saved as random, mask as a white rectangle
            # They should not be identical
            assert not np.array_equal(image, mask)

    def test_size_mismatch_detected_not_accepted(self, tmp_path):
        """Kill M311: != must not become == (accept mismatch / reject match)."""
        import skimage.io as skio
        root = _make_minimal_busi_root(tmp_path)
        benign_dir = root / "benign"
        masks = sorted(benign_dir.glob("*_mask.png"))
        bad_mask = np.zeros((16, 32), dtype=np.uint8)
        skio.imsave(str(masks[0]), bad_mask)
        with pytest.raises(ValueError):
            load_busi(root)

    def test_multiple_images_per_label_all_loaded(self, tmp_path):
        """Kill M307: continue→break would stop after first missing mask."""
        import skimage.io as skio
        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            d = root / label
            d.mkdir(parents=True)
            for i in range(1, 4):  # 3 images per label
                img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
                mask = np.zeros((16, 16), dtype=np.uint8)
                skio.imsave(str(d / f"{label}_{i:03d}.png"), img)
                skio.imsave(str(d / f"{label}_{i:03d}_mask.png"), mask)
        result = load_busi(root)
        assert len(result) == 9   # 3 labels × 3 images


def _make_minimal_busi_root(tmp_path):
    """Create a minimal BUSI directory with one 32x32 image+mask per label."""
    try:
        import skimage.io as skio
    except ImportError:
        pytest.skip("scikit-image not available")

    root = tmp_path / "busi"
    for label in ("benign", "malignant", "normal"):
        d = root / label
        d.mkdir(parents=True)
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        skio.imsave(str(d / f"{label}_001.png"), img)
        skio.imsave(str(d / f"{label}_001_mask.png"), mask)
    return root
