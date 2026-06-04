"""Tests for src/segment.py.

Testing methods applied:
  - Input Space Partitioning (W6): image ndim x model type x threshold
  - Boundary Value Analysis: threshold at 0.0, 1.0, out-of-range
  - Mock injection: real GPU model replaced by lightweight callable
"""
import numpy as np
import pytest

from src.segment import build_model, postprocess_mask, predict


# ---------------------------------------------------------------------------
# predict — ISP on image dimensionality
# ---------------------------------------------------------------------------

class TestPredict:
    def test_2d_image_accepted(self, tiny_image_float, mock_model):
        """ISP C1: 2-D input is valid."""
        result = predict(mock_model, tiny_image_float)
        assert result.ndim == 2
        assert result.shape == tiny_image_float.shape

    def test_3d_image_accepted(self, tiny_image_float, mock_model):
        """ISP C1: 3-D HxWxC input is valid."""
        img3d = np.stack([tiny_image_float] * 3, axis=-1)
        result = predict(mock_model, img3d)
        assert result.ndim == 2

    def test_1d_image_raises(self, mock_model):
        """ISP C1 invalid: 1-D input raises ValueError."""
        with pytest.raises(ValueError):
            predict(mock_model, np.zeros(32))

    def test_4d_image_raises(self, mock_model):
        """ISP C1 invalid: 4-D input raises ValueError."""
        with pytest.raises(ValueError):
            predict(mock_model, np.zeros((1, 32, 32, 1)))

    def test_output_is_float32_probability_map(self, tiny_image_float, mock_model):
        result = predict(mock_model, tiny_image_float)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mock_model_returns_expected_values(self, tiny_image_float, mock_model):
        """Mock model returns all-0.9; verify downstream sees that value."""
        result = predict(mock_model, tiny_image_float)
        np.testing.assert_allclose(result, 0.9, atol=1e-5)


# ---------------------------------------------------------------------------
# postprocess_mask — BVA on threshold
# ---------------------------------------------------------------------------

class TestPostprocessMask:
    def test_threshold_zero_all_true(self, tiny_image_float):
        """BVA: threshold=0.0 should mark everything as positive."""
        prob = np.full_like(tiny_image_float, 0.5)
        mask = postprocess_mask(prob, threshold=0.0)
        assert mask.all()

    def test_threshold_one_all_false(self, tiny_image_float):
        """BVA: threshold=1.0 should mark nothing positive (prob < 1.0)."""
        prob = np.full_like(tiny_image_float, 0.9)
        mask = postprocess_mask(prob, threshold=1.0)
        assert not mask.any()

    def test_threshold_default_binarises(self, tiny_image_float):
        """Default threshold=0.5; values > 0.5 become True."""
        prob = tiny_image_float.copy()
        prob[:16, :] = 0.8
        prob[16:, :] = 0.2
        mask = postprocess_mask(prob)
        assert mask[:16, :].all()
        assert not mask[16:, :].any()

    def test_threshold_negative_raises(self, tiny_image_float):
        """BVA: threshold < 0 raises ValueError."""
        with pytest.raises(ValueError):
            postprocess_mask(tiny_image_float, threshold=-0.1)

    def test_threshold_above_one_raises(self, tiny_image_float):
        """BVA: threshold > 1 raises ValueError."""
        with pytest.raises(ValueError):
            postprocess_mask(tiny_image_float, threshold=1.1)

    def test_output_is_bool_array(self, tiny_image_float):
        mask = postprocess_mask(tiny_image_float)
        assert mask.dtype == bool
        assert mask.shape == tiny_image_float.shape


# ---------------------------------------------------------------------------
# build_model + real U-Net forward pass
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_returns_torch_module(self):
        import torch.nn as nn
        model = build_model()
        assert isinstance(model, nn.Module)

    def test_real_model_predict_shape(self, tiny_image_float):
        """ISP C2: real torch model (not mock) returns correct shape."""
        model = build_model()
        result = predict(model, tiny_image_float)
        assert result.ndim == 2
        assert result.shape == tiny_image_float.shape

    def test_real_model_output_in_zero_one(self, tiny_image_float):
        """Real model output (sigmoid) must stay in [0, 1]."""
        model = build_model()
        result = predict(model, tiny_image_float)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ===========================================================================
# PARAMETRIZED MATRIX — ISP W6 full expansion for postprocess_mask
#
# Characteristics:
#   C1 – image shape:   {(4,4),(8,8),(16,16),(32,32),(8,16),(16,8)}   6 blocks
#   C2 – threshold:     {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,               11 blocks
#                         0.6, 0.7, 0.8, 0.9, 1.0}
#   C3 – prob value:    {0.0, 0.25, 0.5, 0.75, 1.0}                   5 blocks
#
# Valid combinations: 6 × 11 × 5 = 330 tests
# BVA invalid threshold: 3 shapes × 4 invalid values = 12 tests
# Total new parametrized: 342
# ===========================================================================
import itertools as _it3

_PS_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32), (8, 16), (16, 8)]
# C2: threshold at 0 (include all), step through mid range, 1.0 (strict)
_PS_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# C3: prob_value at boundary (0), quarter marks, and boundary (1)
_PS_PROB_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
# BVA: outside [0,1]
_PS_INVALID_THRESHOLDS = [-0.1, -1.0, 1.1, 2.0]
_PS_SHAPES_SMALL = [(4, 4), (8, 8), (16, 16)]


class TestPostprocessParametrize:
    """ISP W6 C1 × C2 × C3 — 330 valid + 12 BVA invalid = 342 tests."""

    # Valid combinations: constant probability map (330 tests)
    # Each combination tests a specific (shape, threshold, prob_value) block.
    # Expected output: mask is all-True if prob_val >= threshold, else all-False.
    @pytest.mark.parametrize(
        "shape,threshold,prob_val",
        list(_it3.product(_PS_SHAPES, _PS_THRESHOLDS, _PS_PROB_VALUES)),
    )
    def test_constant_prob_correct_output(self, shape, threshold, prob_val):
        """ISP C1-C3: constant prob map → all-True or all-False per threshold."""
        prob = np.full(shape, prob_val, dtype=np.float32)
        mask = postprocess_mask(prob, threshold=threshold)
        expected = bool(prob_val >= threshold)
        assert mask.dtype == bool
        assert mask.shape == shape
        if expected:
            assert mask.all(), f"Expected all True for prob={prob_val}, thr={threshold}"
        else:
            assert not mask.any(), f"Expected all False for prob={prob_val}, thr={threshold}"

    # BVA: invalid threshold → ValueError (12 tests)
    @pytest.mark.parametrize(
        "shape,threshold",
        list(_it3.product(_PS_SHAPES_SMALL, _PS_INVALID_THRESHOLDS)),
    )
    def test_invalid_threshold_raises(self, shape, threshold):
        """BVA W6: threshold outside [0,1] → ValueError."""
        prob = np.full(shape, 0.5, dtype=np.float32)
        with pytest.raises(ValueError):
            postprocess_mask(prob, threshold=threshold)
