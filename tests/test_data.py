"""Tests for src/data.py — load_busi.

Testing methods applied:
  - Input Space Partitioning (W6): C1={exists,missing} x C2={complete,missing subfolder}
    x C3={matched,size mismatch}
  - Boundary Value Analysis: smallest valid dataset (1 image per label)
  - Graph Coverage (W7): all edges of load_busi control-flow graph
"""
import re

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

    def test_missing_root_message_is_anchored(self, tmp_path):
        """Kill mutant: f-string 'Dataset root not found: ...' wrapped in XX markers.

        `match="root"` alone is insensitive to XX-prefix/suffix mangling because
        re.search still finds 'root' inside 'XXDataset root not found...XX'.
        Anchoring at the start kills the mangled-string mutant.
        """
        with pytest.raises(FileNotFoundError, match=r"^Dataset root not found:"):
            load_busi(tmp_path / "does_not_exist")

    def test_missing_label_message_is_anchored(self, tmp_path):
        """Kill mutant: f-string 'Missing label subdirectory: ...' wrapped in XX markers."""
        root = tmp_path / "busi"
        root.mkdir()
        (root / "malignant").mkdir()
        (root / "normal").mkdir()
        with pytest.raises(FileNotFoundError, match=r"^Missing label subdirectory:"):
            load_busi(root)

    def test_mask_files_excluded_from_image_list(self, tmp_path):
        """Kill mutant: '_mask' substring check mangled to 'XX_maskXX' (never matches,
        so mask files would be treated as images too).

        Chains image -> mask -> "mask of mask" so that, if mask files were wrongly
        accepted as images, an extra spurious pair would be produced.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        mask_of_mask = np.full((16, 16), 7, dtype=np.uint8)
        skio.imsave(str(benign / "a.png"), img)
        skio.imsave(str(benign / "a_mask.png"), mask)
        skio.imsave(str(benign / "a_mask_mask.png"), mask_of_mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        result = load_busi(root)
        benign_records = [r for r in result if r[2] == "benign"]
        # Only 'a.png' is a real image; 'a_mask.png' must be excluded from
        # the image list (it should only ever be loaded as a*'s mask).
        assert len(benign_records) == 1

    def test_missing_mask_skips_image_not_aborts_label(self, tmp_path):
        """Kill mutant: continue -> break in the 'mask missing' branch.

        With `continue`, an image lacking a mask is skipped and the loop moves
        on; with `break` it would abort the whole label directory, dropping
        images that come after it.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        for name in ("a", "b", "c"):
            img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            skio.imsave(str(benign / f"{name}.png"), img)
        # 'b' deliberately has no mask; 'a' and 'c' (which sorts after 'b') do.
        for name in ("a", "c"):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            skio.imsave(str(benign / f"{name}_mask.png"), mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        result = load_busi(root)
        benign_count = sum(1 for r in result if r[2] == "benign")
        # 'continue' loads both 'a' and 'c' (skipping only 'b'); 'break' would
        # stop at 'b' and drop 'c' too, yielding only 1.
        assert benign_count == 2

    def test_rgb_image_with_grayscale_mask_same_hw_is_accepted(self, tmp_path):
        """Kill mutant: image.shape[:2] -> image.shape[:3] in the size check.

        A 3-channel image and a single-channel mask that agree on (H, W) must
        be accepted — the comparison only cares about the spatial dimensions.
        Comparing shape[:3] vs shape[:2] would spuriously raise ValueError.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        rgb_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        gray_mask = np.zeros((16, 16), dtype=np.uint8)
        gray_mask[4:12, 4:12] = 255
        skio.imsave(str(benign / "a.png"), rgb_img)
        skio.imsave(str(benign / "a_mask.png"), gray_mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        result = load_busi(root)  # must not raise
        assert any(r[2] == "benign" for r in result)

    def test_grayscale_image_with_rgb_mask_same_hw_is_accepted(self, tmp_path):
        """Kill mutant: mask.shape[:2] -> mask.shape[:3] in the size check.

        Mirror of the RGB-image case: a single-channel image paired with a
        3-channel mask that agree on (H, W) must be accepted.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        gray_img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        rgb_mask = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb_mask[4:12, 4:12, :] = 255
        skio.imsave(str(benign / "a.png"), gray_img)
        skio.imsave(str(benign / "a_mask.png"), rgb_mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        result = load_busi(root)  # must not raise
        assert any(r[2] == "benign" for r in result)

    def test_size_mismatch_message_format_is_exact(self, tmp_path):
        """Kill mutants: mismatch-message text wrapped in XX markers (prefix/suffix).

        Anchors both ends of the message so a mangled 'XXImage/mask size
        mismatch...' prefix or '...(H, W)XX' suffix fails to match.
        """
        import skimage.io as skio

        root = _make_minimal_busi_root(tmp_path)
        benign_dir = root / "benign"
        masks = sorted(benign_dir.glob("*_mask.png"))
        bad_mask = np.zeros((16, 32), dtype=np.uint8)
        skio.imsave(str(masks[0]), bad_mask)
        with pytest.raises(
            ValueError,
            match=r"^Image/mask size mismatch for .+: \(\d+, \d+\) vs \(\d+, \d+\)$",
        ):
            load_busi(root)

    def test_size_mismatch_message_reports_2d_image_shape(self, tmp_path):
        """Kill mutant: image.shape[:2] -> image.shape[:3] inside the message f-string.

        Forces the mismatch with a 3-D (RGB) image so a [:3] slice would be
        observably different ((H, W, 3) vs (H, W)) from the correct [:2] form.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        rgb_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        bad_mask = np.zeros((20, 20), dtype=np.uint8)
        bad_mask[4:12, 4:12] = 255
        skio.imsave(str(benign / "a.png"), rgb_img)
        skio.imsave(str(benign / "a_mask.png"), bad_mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        with pytest.raises(ValueError, match=re.escape("(16, 16) vs (20, 20)")):
            load_busi(root)

    def test_size_mismatch_message_reports_2d_mask_shape(self, tmp_path):
        """Kill mutant: mask.shape[:2] -> mask.shape[:3] inside the message f-string.

        Mirror of the image case, forcing the mismatch with a 3-D (RGB) mask.
        """
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)

        benign = root / "benign"
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        rgb_mask = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb_mask[4:12, 4:12, :] = 255
        skio.imsave(str(benign / "a.png"), img)
        skio.imsave(str(benign / "a_mask.png"), rgb_mask)

        for label in ("malignant", "normal"):
            d = root / label
            im = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            mk = np.zeros((16, 16), dtype=np.uint8)
            mk[4:12, 4:12] = 255
            skio.imsave(str(d / f"{label}_001.png"), im)
            skio.imsave(str(d / f"{label}_001_mask.png"), mk)

        with pytest.raises(ValueError, match=re.escape("(16, 16) vs (20, 20)")):
            load_busi(root)


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
