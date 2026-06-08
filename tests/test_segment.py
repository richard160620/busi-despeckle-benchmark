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


# ===========================================================================
# Mutation killers (W9 Syntax-Based / Mutation Testing) — see mutmut survivors
# for src/segment.py. Each test below targets one or more surviving mutants
# identified via `mutmut results` + `mutmut show <id>`.
#
# Documented EQUIVALENT mutants (no test can distinguish them — proven via
# direct computation, not tested here):
#   3, 6   — ReLU(inplace=True/False): output values are bit-identical
#   47, 54 — F.interpolate(size=e2.shape[-2:] vs [+2:]): identical slices for
#            any 4-D tensor (e2/e1 are always (N,C,H,W))
#   50, 57 — align_corners=True/False: the U-Net's stride-2 pool/up-conv pairs
#            guarantee up-conv output spatial size == skip-connection size for
#            any input padded to a multiple of 16, so F.interpolate is always
#            a same-size resize (a no-op regardless of corner alignment)
#   100    — unsqueeze(0).unsqueeze(0) vs unsqueeze(0).unsqueeze(1): both
#            insert a singleton axis adjacent to an existing size-1 leading
#            axis, producing bit-identical tensors (verified empirically)
# ===========================================================================

class TestSegmentMutationKillers:
    # -- _LightUNet architecture: kernel/stride/channel-width mutants
    #    (2,5,8,14,25,26,35,36) via pinned seeded forward-pass output --
    @pytest.mark.regression
    def test_real_model_predict_pinned_output_values(self):
        """Pin torch.manual_seed(0) U-Net forward-pass output for a 32x32
        (already-multiple-of-16, no padding) input. Any change to conv
        padding, base width, pool kernel, or up-conv kernel/stride shifts
        these values by >0.07 — far outside the pinned tolerance."""
        import torch
        torch.manual_seed(0)
        model = build_model()
        img = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
        out = predict(model, img)
        assert out.mean() == pytest.approx(0.472008228302, abs=1e-9)
        assert out[0, 0] == pytest.approx(0.451910674572, abs=1e-9)
        assert out[17, 23] == pytest.approx(0.476995676756, abs=1e-9)
        assert out[31, 31] == pytest.approx(0.705146908760, abs=1e-9)

    # -- ndim validation messages (64, 66, 67) --
    def test_ndim_too_low_message_anchored(self, mock_model):
        with pytest.raises(ValueError, match=r"^image must be 2-D or 3-D, got ndim=1$"):
            predict(mock_model, np.zeros(5))

    def test_4d_image_message_anchored(self, mock_model):
        """Anchored on the *correct* ndim>3 message for a 4-D input. A
        mutant that changes the boundary to ndim>4 (66) lets a 4-D image
        fall through to `h, w = img.shape`, raising a *different* ValueError
        ('too many values to unpack') — failing this match. Also kills the
        XX-wrapped message at this raise site (67)."""
        with pytest.raises(ValueError, match=r"^image must be 2-D or 3-D, got ndim=4$"):
            predict(mock_model, np.zeros((1, 32, 32, 1)))

    # -- 3-D channel-mean axis (71, 72) --
    def test_3d_image_channel_mean_uses_last_axis(self, mock_model):
        """HxWxC must be averaged over the channel axis (-1). Averaging over
        axis +1 or -2 instead collapses W (not C), producing shape (H, C)
        — observable here because H != C."""
        img3d = np.random.default_rng(7).random((20, 24, 3)).astype(np.float32)
        result = predict(mock_model, img3d)
        assert result.shape == (20, 24)

    # -- normalisation-to-[0,1] branch (74,75,76,77,78) --
    def test_image_at_max_exactly_one_is_not_rescaled(self):
        """Boundary: img.max() == 1.0 must NOT trigger the /255 rescale
        (kills the `>= 1.0` mutant 74, which would divide an already-unit
        image by 255)."""
        captured = {}

        class _SpyModel:
            def __call__(self, x):
                captured["tensor"] = x.clone()
                import torch
                return torch.full_like(x, 0.5)

        img = np.zeros((16, 16), dtype=np.float32)
        img[0, 0] = 1.0
        predict(_SpyModel(), img)
        assert float(captured["tensor"].max()) == pytest.approx(1.0, abs=1e-6)

    def test_image_with_max_above_one_gets_rescaled_by_255(self):
        """Pin the exact rescale arithmetic (img / 255.0) for an image whose
        max is in (1.0, 2.0]. Kills:
          75 (`> 2.0` boundary — would skip rescaling entirely),
          76 (`* 255.0`), 77 (`/ 256.0`), 78 (`= None` -> AttributeError)."""
        captured = {}

        class _SpyModel:
            def __call__(self, x):
                captured["tensor"] = x.clone()
                import torch
                return torch.full_like(x, 0.5)

        img = np.zeros((16, 16), dtype=np.float32)
        img[0, 0] = 1.5
        predict(_SpyModel(), img)
        assert float(captured["tensor"].max()) == pytest.approx(1.5 / 255.0, abs=1e-7)

    # -- pad_h/pad_w arithmetic + padding gate (80,81,83,85,87,88,90,92,94,95,96) --
    @pytest.mark.parametrize("h,w,expected_padded_shape", [
        (20, 27, (1, 1, 32, 32)),  # neither dim a multiple of 16 (general arithmetic)
        (32, 48, (1, 1, 32, 48)),  # both dims already multiples of 16 (r==0 edge case)
        (32, 27, (1, 1, 32, 32)),  # mixed: pad_h==0, pad_w>0 (kills `or` -> `and` gate)
    ])
    def test_predict_pads_to_correct_multiple_of_16(self, h, w, expected_padded_shape):
        """Pin the exact padded tensor shape fed to the model. Together these
        three (h, w) combinations distinguish every pad_h/pad_w arithmetic
        mutant (off-by-one constants, +/- swaps, %16 -> %17) and the
        `if pad_h or pad_w` -> `and` gate and reflect-pad offset mutants."""
        captured = {}

        class _SpyModel:
            def __call__(self, x):
                captured["shape"] = tuple(x.shape)
                import torch
                return torch.full_like(x, 0.5)

        img = np.random.default_rng(1).random((h, w)).astype(np.float32)
        predict(_SpyModel(), img)
        assert captured["shape"] == expected_padded_shape

    # -- postprocess_mask threshold-out-of-range message (109) --
    def test_threshold_out_of_range_message_anchored(self):
        with pytest.raises(
            ValueError, match=r"^threshold must be in \[0, 1\], got -0\.1$"
        ):
            postprocess_mask(np.zeros((4, 4)), threshold=-0.1)


# ===========================================================================
# REGRESSION — odd-sized inputs must not raise and must preserve shape (W14)
#
# Bug: POST /segment returned 500 with RuntimeError on skip-connection concat
# when H or W is not a multiple of 16.  Fixed by:
#   1. F.interpolate before each torch.cat in _LightUNet.forward()
#   2. Padding input to a multiple of 16 in predict(), cropped back after.
# ===========================================================================

@pytest.mark.regression
@pytest.mark.parametrize("h,w", [
    (307, 306),
    (101, 97),
    (33, 31),
])
def test_odd_sized_input_shape_preserved(h, w):
    """W14 Regression: predict() output shape == (H, W) for any odd-sized input.

    Pins the fix for the skip-connection size mismatch that caused 500 errors
    on inputs whose height or width is not a multiple of 16.
    """
    model = build_model()
    img = np.random.default_rng(42).random((h, w)).astype(np.float32)
    result = predict(model, img)
    assert result.shape == (h, w), (
        f"Expected output shape ({h}, {w}), got {result.shape}"
    )
