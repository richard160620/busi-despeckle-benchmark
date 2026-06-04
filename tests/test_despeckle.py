"""Tests for src/despeckle.py.

Testing methods applied:
  - Input Space Partitioning (W6): method x window_size x dtype
  - Boundary Value Analysis: window=1 (min valid), window=3 (typical),
    window=0 (invalid), window=-1 (invalid), even window (invalid)
  - Graph Coverage (W7): each dispatch edge (none/median/lee/frost/srad/nlm/unknown)
"""
import numpy as np
import pytest

from src.despeckle import despeckle


# ---------------------------------------------------------------------------
# ISP / Graph Coverage: method dispatch
# ---------------------------------------------------------------------------

class TestDespeckleDispatch:
    """Graph edges: [none, median, lee, frost, srad, nlm] dispatch paths."""

    @pytest.mark.parametrize("method", ["none", "median", "lee", "frost", "srad", "nlm"])
    def test_valid_methods_return_same_shape(self, tiny_image_float, method):
        """Edge: each dispatch branch -> output shape matches input."""
        out = despeckle(tiny_image_float, method=method, window_size=3)
        assert out.shape == tiny_image_float.shape

    @pytest.mark.parametrize("method", ["none", "median", "lee", "frost", "srad", "nlm"])
    def test_valid_methods_preserve_dtype(self, tiny_image_float, method):
        """Output dtype must match input dtype."""
        out = despeckle(tiny_image_float, method=method, window_size=3)
        assert out.dtype == tiny_image_float.dtype

    def test_none_method_returns_exact_copy(self, tiny_image_float):
        """Edge: none -> return copy (values unchanged)."""
        out = despeckle(tiny_image_float, method="none")
        np.testing.assert_array_equal(out, tiny_image_float)

    def test_unknown_method_raises_value_error(self, tiny_image_float):
        """Edge: unknown method -> ValueError."""
        with pytest.raises(ValueError, match="method"):
            despeckle(tiny_image_float, method="unknown_method")


# ---------------------------------------------------------------------------
# BVA: window / kernel size boundaries
# ---------------------------------------------------------------------------

class TestDespeckleWindowBVA:
    """BVA: window_size at boundaries — 1 (min valid), 0 (invalid), -1 (invalid),
    2 (even, invalid)."""

    def test_window_size_1_accepted(self, tiny_image_float):
        """BVA min-valid: window_size=1 should be accepted."""
        out = despeckle(tiny_image_float, method="median", window_size=1)
        assert out.shape == tiny_image_float.shape

    def test_window_size_3_accepted(self, tiny_image_float):
        """BVA typical: window_size=3."""
        out = despeckle(tiny_image_float, method="median", window_size=3)
        assert out.shape == tiny_image_float.shape

    def test_window_size_0_raises(self, tiny_image_float):
        """BVA boundary: 0 is not a positive integer."""
        with pytest.raises(ValueError):
            despeckle(tiny_image_float, method="median", window_size=0)

    def test_window_size_negative_raises(self, tiny_image_float):
        """BVA: negative window_size must raise ValueError."""
        with pytest.raises(ValueError):
            despeckle(tiny_image_float, method="median", window_size=-1)

    def test_window_size_even_raises(self, tiny_image_float):
        """BVA: even window_size is invalid (filters require odd)."""
        with pytest.raises(ValueError):
            despeckle(tiny_image_float, method="median", window_size=2)


# ---------------------------------------------------------------------------
# ISP: input dtype partitions
# ---------------------------------------------------------------------------

class TestDespeckleDtype:
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_various_dtypes_preserved(self, tiny_image, dtype):
        """ISP C3: output dtype matches input for uint8/float32/float64."""
        img = tiny_image.astype(dtype)
        if np.issubdtype(dtype, np.floating):
            img = img / 255.0
        out = despeckle(img, method="median", window_size=3)
        assert out.dtype == dtype

    def test_none_method_uint8(self, tiny_image):
        """ISP: none method with uint8 preserves exact values."""
        out = despeckle(tiny_image, method="none")
        np.testing.assert_array_equal(out, tiny_image)
