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


# ---------------------------------------------------------------------------
# Mutation killers (W10 Syntax-Based Testing)
# Each test is labelled with the mutant ID it is designed to kill.
# ---------------------------------------------------------------------------

class TestDespackleMutationKillers:
    """W10: targeted tests to kill surviving mutmut mutants."""

    def test_default_method_is_none(self, tiny_image_float):
        """Kill M105: default method='none' — calling without method must work."""
        out = despeckle(tiny_image_float)          # no method= arg
        np.testing.assert_array_equal(out, tiny_image_float)

    def test_window_zero_raises_positive_message(self, tiny_image_float):
        """Kill M97+M99: window=0 must raise 'positive', not fall through to 'odd'."""
        with pytest.raises(ValueError, match="positive"):
            despeckle(tiny_image_float, method="median", window_size=0)

    def test_even_window_raises_odd_message(self, tiny_image_float):
        """Kill M104: even window must say 'odd' in the error."""
        with pytest.raises(ValueError, match="odd"):
            despeckle(tiny_image_float, method="median", window_size=2)

    def test_unknown_method_error_contains_unsupported(self, tiny_image_float):
        """Kill M107+M108: error message for bad method must contain 'Unsupported'."""
        with pytest.raises(ValueError, match="Unsupported"):
            despeckle(tiny_image_float, method="bad_method")

    def test_default_window_size_is_valid_odd(self, tiny_image_float):
        """Kill M114: default window_size=3 (odd); if mutated to 4 (even) → ValueError."""
        out = despeckle(tiny_image_float, method="median")   # no window_size arg
        assert out.shape == tiny_image_float.shape

    def test_srad_n_iter_1_differs_from_10(self, tiny_image_float):
        """Kill M128: n_iter param must be forwarded; 1 iteration ≠ 10 iterations."""
        out1  = despeckle(tiny_image_float, method="srad", n_iter=1)
        out10 = despeckle(tiny_image_float, method="srad", n_iter=10)
        assert not np.allclose(out1, out10)


# ---------------------------------------------------------------------------
# Regression tests for filter output values — kill arithmetic mutants
# (W10 + W14: each pinned value was computed from the correct implementation)
#
# Fixed seed=99, 8×8 float32 image, window_size=3.
# Any arithmetic mutation in _median_filter, _lee_filter, _frost_filter,
# _srad_filter, or _nlm_filter will change these values.
# ---------------------------------------------------------------------------

class TestDespeckleFilterRegressions:
    """W10+W14: pinned output values kill arithmetic mutations in filter implementations."""

    _IMG = np.random.default_rng(99).random((8, 8)).astype(np.float32)

    @pytest.mark.regression
    def test_median_filter_pinned_mean(self):
        """Kill mutants in _median_filter: pin output mean/std on seed-99 image."""
        out = despeckle(self._IMG, method="median", window_size=3)
        assert out.mean() == pytest.approx(0.509988, abs=1e-4)
        assert out.std()  == pytest.approx(0.141972, abs=1e-4)

    @pytest.mark.regression
    def test_lee_filter_pinned_mean(self):
        """Kill mutants in _lee_filter: local_var, noise_var, k formula."""
        out = despeckle(self._IMG, method="lee", window_size=3)
        assert out.mean() == pytest.approx(0.505515, abs=1e-4)
        assert out[0, 0]  == pytest.approx(0.521568, abs=1e-4)

    @pytest.mark.regression
    def test_frost_filter_pinned_corner(self):
        """Kill mutants in _frost_filter: damping constant, weight formula."""
        out = despeckle(self._IMG, method="frost", window_size=3)
        assert out[0, 0]  == pytest.approx(0.556666, abs=2e-4)
        assert out[3, 3]  == pytest.approx(0.444108, abs=2e-4)

    @pytest.mark.regression
    def test_srad_filter_pinned_mean(self):
        """Kill mutants in _srad_filter: gradient, diffusion coefficient formula."""
        out = despeckle(self._IMG, method="srad", n_iter=10)
        assert out.mean() == pytest.approx(0.504512, abs=1e-4)
        assert out.std()  == pytest.approx(0.053290, abs=1e-4)

    @pytest.mark.regression
    def test_nlm_filter_pinned_mean(self):
        """Kill mutants in _nlm_filter: normalise/denormalise formula."""
        out = despeckle(self._IMG, method="nlm", window_size=3)
        assert out.mean() == pytest.approx(0.500486, abs=1e-3)
        assert out[3, 3]  == pytest.approx(0.580652, abs=1e-3)

    def test_median_float_output_stays_in_unit_interval(self):
        """Kill M145 (*255 return): if output is ×255 instead of ÷255, range >> 1."""
        out = despeckle(self._IMG, method="median", window_size=3)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_median_float_input_max_exactly_one(self):
        """Kill M139+M143 (<= vs <): image with max==1.0 must use float path."""
        img = np.zeros((8, 8), dtype=np.float32)
        img[0, 0] = 1.0   # max == 1.0 exactly
        out = despeckle(img, method="median", window_size=3)
        # float path: output in [0,1]; uint8 path would change values
        assert out.max() <= 1.0 + 1e-5
        assert out.dtype == np.float32

    def test_median_output_not_near_zero_for_bright_image(self):
        """Kill M135 (*255 → /255): bright float image (/255 → near-zero output)."""
        img = np.full((8, 8), 0.8, dtype=np.float32)  # all 0.8
        out = despeckle(img, method="median", window_size=3)
        # Correct: output ~0.8; M135 (/255): output ~0.003
        assert out.mean() > 0.5

    def test_median_output_differs_from_lee(self):
        """Sanity: different filters produce different outputs (catches method-swap mutants)."""
        med = despeckle(self._IMG, method="median", window_size=3)
        lee = despeckle(self._IMG, method="lee",    window_size=3)
        assert not np.allclose(med, lee)

    def test_lee_output_differs_from_srad(self):
        """Sanity: lee ≠ srad output."""
        lee  = despeckle(self._IMG, method="lee",  window_size=3)
        srad = despeckle(self._IMG, method="srad", n_iter=10)
        assert not np.allclose(lee, srad)


# ===========================================================================
# PARAMETRIZED MATRIX — ISP W6 full expansion
#
# Characteristics:
#   C1 – method:       {none, median, lee, frost, srad, nlm}           6 blocks
#   C2 – kernel_size:  {1, 3, 5, 7}   (valid odd positives)            4 blocks
#   C3 – image shape:  {(8,8)=small-sq, (16,16)=med-sq,                4 blocks
#                       (32,32)=large-sq, (8,16)=non-sq}
#   C4 – dtype:        {uint8, float32, float64}                        3 blocks
#
# Valid combinations: 6 × 4 × 4 × 3 = 288 tests
# Invalid kernel combinations (BVA): 5 methods × 6 invalid values = 30 tests
# Total new parametrized tests: 318
# ===========================================================================
import itertools as _it

_PM_METHODS = ["none", "median", "lee", "frost", "srad", "nlm"]
_PM_VALID_KERNELS = [1, 3, 5, 7]
# C3: small-square / medium-square / large-square / non-square
_PM_SHAPES = [(8, 8), (16, 16), (32, 32), (8, 16)]
_PM_DTYPES = [np.uint8, np.float32, np.float64]
# BVA: zero, negative, two negatives, even numbers
_PM_INVALID_KERNELS = [0, -1, -3, 2, 4, 6]
# none skips window validation; all others must raise on invalid kernel
_PM_WINDOWED_METHODS = ["median", "lee", "frost", "srad", "nlm"]


def _make_test_image(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    return rng.integers(0, 256, shape, dtype=dtype)


class TestDespeckleParametrize:
    """ISP W6: C1 × C2 × C3 × C4 full matrix — 288 valid + 30 BVA invalid."""

    # ------------------------------------------------------------------
    # Valid combinations (288 tests)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "method,kernel,shape,dtype",
        list(_it.product(_PM_METHODS, _PM_VALID_KERNELS, _PM_SHAPES, _PM_DTYPES)),
    )
    def test_output_shape_preserved(self, method, kernel, shape, dtype):
        """ISP C1-C4: valid input → output shape == input shape."""
        img = _make_test_image(shape, dtype)
        out = despeckle(img, method=method, window_size=kernel)
        assert out.shape == shape

    @pytest.mark.parametrize(
        "method,kernel,shape,dtype",
        list(_it.product(_PM_METHODS, _PM_VALID_KERNELS, _PM_SHAPES, _PM_DTYPES)),
    )
    def test_output_dtype_preserved(self, method, kernel, shape, dtype):
        """ISP C4: valid input → output dtype == input dtype."""
        img = _make_test_image(shape, dtype)
        out = despeckle(img, method=method, window_size=kernel)
        assert out.dtype == dtype

    # ------------------------------------------------------------------
    # BVA: invalid kernel sizes (30 tests)
    # C2 boundaries: 0 (below min), negative, even
    # none method is excluded — it returns before window validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "method,kernel",
        list(_it.product(_PM_WINDOWED_METHODS, _PM_INVALID_KERNELS)),
    )
    def test_invalid_kernel_raises_value_error(self, method, kernel):
        """BVA W6: kernel ∈ {0,-1,-3,2,4,6} → ValueError for windowed methods."""
        img = np.random.default_rng(1).random((8, 8)).astype(np.float32)
        with pytest.raises(ValueError):
            despeckle(img, method=method, window_size=kernel)


# ---------------------------------------------------------------------------
# Regression: _frost_filter must be vectorized, not a per-pixel Python loop
# (a naive double for-loop scales so badly on real BUSI-sized images that it
#  blocks the single-threaded Flask dev server for minutes — this is the
#  root cause of the user-reported "the web has not responded" symptom)
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestFrostFilterPerformance:
    def test_frost_runs_well_under_a_second_on_a_realistic_image(self):
        import time

        img = np.random.default_rng(2).random((256, 256)).astype(np.float64)
        t0 = time.time()
        despeckle(img, method="frost", window_size=5)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"frost took {elapsed:.2f}s on a 256x256 image — too slow for a web request"
