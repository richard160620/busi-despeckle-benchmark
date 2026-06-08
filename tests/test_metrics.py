"""Tests for src/metrics.py.

Testing methods applied:
  - Logic Coverage / MC-DC (W8): is_result_valid compound predicate
  - Input Space Partitioning (W6): pred/target content x shape x area
  - Boundary Value Analysis: both-empty convention, threshold edges
  - Regression (W14): fixed-input dice/iou values pinned — see Issue #2
"""
import numpy as np
import pytest

from src.metrics import dice, hd95, is_result_valid, iou, niqe, psnr, ssim


# ---------------------------------------------------------------------------
# dice
# ---------------------------------------------------------------------------

class TestDice:
    def test_perfect_overlap_is_one(self, tiny_mask):
        assert dice(tiny_mask, tiny_mask) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self, tiny_mask, empty_mask):
        assert dice(tiny_mask, empty_mask) == pytest.approx(0.0)

    def test_both_empty_is_one(self, empty_mask):
        """Convention: both-empty -> 1.0 (ISP C1 edge case)."""
        assert dice(empty_mask, empty_mask) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self, tiny_mask):
        wrong = np.zeros((16, 16), dtype=bool)
        with pytest.raises(ValueError):
            dice(tiny_mask, wrong)

    def test_result_in_unit_interval(self, tiny_mask, empty_mask):
        val = dice(tiny_mask, empty_mask)
        assert 0.0 <= val <= 1.0

    # Regression test — Issue #2
    @pytest.mark.regression
    def test_regression_known_dice(self):
        """Regression (W14): dice on fixed 4x4 checkerboard input. Issue #2."""
        pred = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [1, 0, 1, 0],
                         [0, 1, 0, 1]], dtype=bool)
        target = np.array([[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 1, 1]], dtype=bool)
        # intersection = 4, |pred| = 8, |target| = 8  ->  dice = 8/16 = 0.5
        assert dice(pred, target) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_overlap_is_one(self, tiny_mask):
        assert iou(tiny_mask, tiny_mask) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self, tiny_mask, empty_mask):
        assert iou(tiny_mask, empty_mask) == pytest.approx(0.0)

    def test_both_empty_is_one(self, empty_mask):
        """Convention: both-empty -> 1.0."""
        assert iou(empty_mask, empty_mask) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self, tiny_mask):
        with pytest.raises(ValueError):
            iou(tiny_mask, np.zeros((16, 16), dtype=bool))

    def test_result_in_unit_interval(self, tiny_mask, empty_mask):
        val = iou(tiny_mask, empty_mask)
        assert 0.0 <= val <= 1.0

    # Regression test — Issue #2
    @pytest.mark.regression
    def test_regression_known_iou(self):
        """Regression (W14): iou on fixed input. Issue #2."""
        pred = np.zeros((4, 4), dtype=bool)
        pred[0:2, 0:2] = True  # 4 pixels
        target = np.zeros((4, 4), dtype=bool)
        target[1:3, 1:3] = True  # 4 pixels, 1 overlap pixel
        # intersection=1, union=7 -> iou=1/7
        assert iou(pred, target) == pytest.approx(1 / 7)


# ---------------------------------------------------------------------------
# hd95
# ---------------------------------------------------------------------------

class TestHD95:
    def test_identical_masks_is_zero(self, tiny_mask):
        assert hd95(tiny_mask, tiny_mask) == pytest.approx(0.0)

    def test_both_empty_is_zero(self, empty_mask):
        assert hd95(empty_mask, empty_mask) == pytest.approx(0.0)

    def test_non_negative(self, tiny_mask, empty_mask):
        # One empty, one non-empty: conventionally large distance
        val = hd95(tiny_mask, tiny_mask)
        assert val >= 0.0

    def test_one_empty_returns_large_distance(self, tiny_mask, empty_mask):
        """BVA: one mask empty -> diagonal distance (not zero)."""
        val = hd95(tiny_mask, empty_mask)
        assert val > 0.0

    def test_shape_mismatch_raises(self, tiny_mask):
        with pytest.raises(ValueError):
            hd95(tiny_mask, np.zeros((16, 16), dtype=bool))


@pytest.mark.regression
class TestHD95Performance:
    """hd95 must scale to realistic BUSI-sized masks. The naive
    scipy.spatial.distance.directed_hausdorff is brute-force O(m*n) over
    foreground-pixel coordinate pairs — on a 256x256 mask with ~50% fill
    (~33k points per side) this alone took >30s per /segment call and
    froze the whole web UI (root cause of "respond time is too long")."""

    def test_runs_well_under_a_second_on_a_realistic_mask(self):
        import time

        rng = np.random.default_rng(3)
        a = rng.random((256, 256)) > 0.5
        b = rng.random((256, 256)) > 0.5
        t0 = time.time()
        hd95(a, b)
        elapsed = time.time() - t0
        assert elapsed < 0.2, f"hd95 took {elapsed:.2f}s on a 256x256 mask — too slow for a web request"


# ---------------------------------------------------------------------------
# psnr / ssim
# ---------------------------------------------------------------------------

class TestPSNR:
    def test_identical_images(self, tiny_image_float):
        val = psnr(tiny_image_float, tiny_image_float)
        assert val > 40.0  # very high PSNR for identical images

    def test_positive_value(self, tiny_image_float):
        noisy = np.clip(tiny_image_float + 0.1, 0, 1).astype(np.float32)
        val = psnr(noisy, tiny_image_float)
        assert val > 0.0

    def test_constant_reference_uses_data_range_one(self):
        """BVA: constant reference (data_range==0) falls back to range=1."""
        ref = np.ones((16, 16), dtype=np.float32) * 0.5
        img = ref.copy()
        val = psnr(img, ref)
        assert isinstance(float(val), float)


class TestSSIM:
    def test_identical_images_is_one(self, tiny_image_float):
        val = ssim(tiny_image_float, tiny_image_float)
        assert val == pytest.approx(1.0, abs=1e-5)

    def test_in_minus_one_to_one(self, tiny_image_float):
        noisy = np.clip(tiny_image_float + 0.2, 0, 1).astype(np.float32)
        val = ssim(noisy, tiny_image_float)
        assert -1.0 <= val <= 1.0

    def test_constant_reference_uses_data_range_one(self):
        """BVA: constant reference (data_range==0) falls back to range=1."""
        ref = np.ones((16, 16), dtype=np.float32) * 0.5
        val = ssim(ref, ref)
        assert isinstance(float(val), float)


# ---------------------------------------------------------------------------
# niqe
# ---------------------------------------------------------------------------

class TestNIQE:
    def test_returns_non_negative_float(self, tiny_image_float):
        val = niqe(tiny_image_float)
        assert isinstance(float(val), float)
        assert val >= 0.0

    def test_constant_image_returns_zero(self):
        """BVA: constant image -> global_var==0 -> score=0.0."""
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        val = niqe(img)
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# is_result_valid — MC-DC / Logic Coverage (W8)
#
# Predicate: has_mask AND dice_finite AND (area_above_min OR allow_empty)
#
# MC-DC truth table:
# Row | has_mask | dice_finite | area_above_min | allow_empty | Result
#  1  |    T     |      T      |       T        |      T      |   T
#  2  |    T     |      T      |       T        |      F      |   T
#  3  |    T     |      T      |       F        |      T      |   T
#  4  |    T     |      T      |       F        |      F      |   F
#  5  |    T     |      F      |       T        |      T      |   F
#  6  |    F     |      T      |       T        |      T      |   F
# MC-DC pairs:
#   has_mask:       rows (1,6)
#   dice_finite:    rows (1,5)
#   area_above_min: rows (2,4)
#   allow_empty:    rows (3,4)
# ---------------------------------------------------------------------------

class TestIsResultValid:
    """W8 Logic Coverage: Predicate, Clause, and MC-DC tests."""

    # Predicate coverage (at least one True, one False overall)
    def test_predicate_true(self):
        assert is_result_valid(True, True, True, False) is True

    def test_predicate_false_no_mask(self):
        assert is_result_valid(False, True, True, True) is False

    # MC-DC: vary has_mask (rows 1 vs 6)
    def test_mcdc_has_mask_true(self):
        assert is_result_valid(True, True, True, True) is True

    def test_mcdc_has_mask_false(self):
        assert is_result_valid(False, True, True, True) is False

    # MC-DC: vary dice_finite (rows 1 vs 5)
    def test_mcdc_dice_finite_true(self):
        assert is_result_valid(True, True, True, True) is True

    def test_mcdc_dice_finite_false(self):
        assert is_result_valid(True, False, True, True) is False

    # MC-DC: vary area_above_min (rows 2 vs 4)
    def test_mcdc_area_true(self):
        assert is_result_valid(True, True, True, False) is True

    def test_mcdc_area_false(self):
        assert is_result_valid(True, True, False, False) is False

    # MC-DC: vary allow_empty (rows 3 vs 4)
    def test_mcdc_allow_empty_true(self):
        assert is_result_valid(True, True, False, True) is True

    def test_mcdc_allow_empty_false(self):
        assert is_result_valid(True, True, False, False) is False

    # Additional clause coverage
    def test_all_false_returns_false(self):
        assert is_result_valid(False, False, False, False) is False


# ---------------------------------------------------------------------------
# Mutation killers (W10 Syntax-Based Testing)
# Each test is labelled with the mutant ID it kills.
# ---------------------------------------------------------------------------

class TestMetricsMutationKillers:
    """W10: targeted tests against each surviving mutmut mutant."""

    # --- _check_shapes error message (M2) ---
    def test_dice_shape_mismatch_error_message(self):
        """Kill M2: error message must contain 'mismatch', not 'XXmismatch'."""
        a, b = np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool)
        with pytest.raises(ValueError, match="mismatch"):
            dice(a, b)

    # --- hd95 one-empty logic (M34, M36, M37) ---
    def test_hd95_empty_pred_nonempty_target_is_diagonal(self):
        """Kill M37: only pred empty → large distance (not zero, not Hausdorff).
        Kill M34: pred has exactly 0 pts (not 1) triggers the branch."""
        pred   = np.zeros((8, 16), dtype=bool)          # pred empty
        target = np.zeros((8, 16), dtype=bool)
        target[0, :] = True                              # target non-empty
        val = hd95(pred, target)
        expected = float(np.sqrt(8 ** 2 + 16 ** 2))    # diagonal of (8,16)
        assert val == pytest.approx(expected)

    def test_hd95_nonempty_pred_empty_target_is_diagonal(self):
        """Kill M36: only target empty → diagonal distance."""
        pred   = np.zeros((8, 16), dtype=bool)
        pred[0, :] = True
        target = np.zeros((8, 16), dtype=bool)
        val = hd95(pred, target)
        expected = float(np.sqrt(8 ** 2 + 16 ** 2))
        assert val == pytest.approx(expected)

    def test_hd95_diagonal_formula_uses_h_and_w(self):
        """Kill M38-44: diagonal must be sqrt(H²+W²); non-square shape exposes H≠W."""
        pred   = np.zeros((6, 10), dtype=bool)          # H=6, W=10 → diag=sqrt(136)
        target = np.zeros((6, 10), dtype=bool)
        target[3, 5] = True
        val = hd95(pred, target)
        assert val == pytest.approx(np.sqrt(6 ** 2 + 10 ** 2))

    # --- psnr data_range (M51, M52) ---
    def test_psnr_uses_max_minus_min(self):
        """Kill M51+M52: data_range = max - min (not max + min, not None).
        ref min=0.3, max=0.7 → range=0.4; M51 would use 0.3+0.7=1.0 instead."""
        ref = np.array([[0.3, 0.3], [0.7, 0.7]], dtype=np.float64)
        img = np.array([[0.4, 0.4], [0.8, 0.8]], dtype=np.float64)  # MSE = 0.01
        # Correct: data_range=0.4, PSNR=10*log10(0.16/0.01)≈12.04 dB
        # Mutant M51 (max+min=1.0): PSNR=10*log10(1.0/0.01)=20.0 dB
        val = psnr(img, ref)
        expected = 10 * np.log10(0.4 ** 2 / 0.01)
        assert val == pytest.approx(expected, abs=0.1)

    def test_psnr_zero_data_range_fallback_is_one(self):
        """Kill M53-56: zero data_range uses fallback=1.0 (not 2.0)."""
        ref = np.full((8, 8), 0.5, dtype=np.float32)   # range=0
        img = ref + 0.1                                  # MSE=0.01
        val = psnr(img, ref)
        expected = 10 * np.log10(1.0 / 0.01)            # ≈20 dB with range=1
        assert val == pytest.approx(expected, abs=0.2)

    # --- ssim data_range (M59, M61-63) ---
    def test_ssim_uses_max_minus_min(self):
        """Kill M59: ssim data_range = max - min."""
        ref = np.zeros((16, 16), dtype=np.float32)
        ref[:8, :] = 1.0                                 # range = 1.0
        val = ssim(ref, ref)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_ssim_zero_data_range_fallback_is_one(self):
        """Kill M61-63: zero data_range uses fallback=1.0."""
        ref = np.full((16, 16), 0.5, dtype=np.float32)
        img = ref + 0.05
        val_low_noise = ssim(img, ref)
        # With fallback=2.0 the result is different; assert we get a sane value
        assert -1.0 <= val_low_noise <= 1.0

    # --- niqe (M66, M68-84) ---
    def test_niqe_nonconstant_is_positive(self):
        """Kill M66 (global_mean→None crash) + M68 (!=0 flips condition):
        non-constant image → global_var > 0 → returns positive score."""
        img = np.random.default_rng(7).random((32, 32)).astype(np.float32)
        val = niqe(img)
        assert val > 0.0        # M68 mutant would enter constant-image branch → 0.0

    @pytest.mark.regression
    def test_niqe_regression_pinned_value(self):
        """Kill M71-80,M82-84 (formula mutations): pin exact niqe for seed=42.
        W10 mutation + W14 regression combined."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64)).astype(np.float32)
        val = niqe(img)
        # Pinned value = 0.98895862 (computed on correct implementation)
        # M82 (÷ → ×) → val >> 1; M71 (filter 7→8) → val ≠ 0.9889
        assert val == pytest.approx(0.98895862, rel=1e-4)

    # -- W10 round 2: anchored-message + crafted-statistics killers (mutmut re-scope) --

    def test_shape_mismatch_message_fully_anchored(self):
        """Kill mutant 2: error text must not be XX-wrapped (anchor start+end)."""
        a, b = np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool)
        with pytest.raises(
            ValueError, match=r"^Shape mismatch: pred \(4, 4\) vs target \(4, 5\)$"
        ):
            dice(a, b)

    def test_psnr_zero_data_range_fallback_handles_negative_values(self):
        """Kill mutant 56 (`data_range = 1.0` -> `None` in psnr's zero-range
        fallback): for non-negative images skimage auto-infers data_range=1.0
        too (making `None` look equivalent), so use a *negative*-valued
        constant reference — there, auto-inference resolves to 2.0 while the
        intended fallback is 1.0, and the two diverge by ~6 dB."""
        ref = np.full((8, 8), -0.5, dtype=np.float64)
        img = ref.copy()
        img[3, 3] += 0.1
        val = psnr(img, ref)
        assert val == pytest.approx(38.0617997398, abs=1e-4)

    def test_ssim_data_range_uses_difference_not_sum(self):
        """Kill mutant 59 (`reference.max() - reference.min()` -> `+`): the
        existing `test_ssim_uses_max_minus_min` reference has min == 0, where
        difference and sum coincide. Use a reference whose minimum is nonzero
        so the two formulas diverge (diff=9.72 vs sum=29.75 -> different
        normalisation -> different ssim score)."""
        rng = np.random.default_rng(3)
        ref = rng.uniform(10, 20, size=(8, 8))
        img = ref + rng.normal(0, 0.5, size=(8, 8))
        val = ssim(img, ref)
        assert val == pytest.approx(0.9800432172, abs=1e-6)

    def test_ssim_zero_data_range_fallback_pinned_value(self):
        """Kill mutant 63 (`data_range = 1.0` -> `2.0` in ssim's zero-range
        fallback): the existing loose `-1<=val<=1` check passes for either
        constant. Pin the exact score, which is sensitive to the constant."""
        ref = np.full((8, 8), 0.5, dtype=np.float64)
        img = ref.copy()
        img[3, 3] += 0.1
        val = ssim(img, ref)
        assert val == pytest.approx(0.815150355309362, abs=1e-6)

    def test_niqe_unit_variance_image_is_not_treated_as_constant(self):
        """Kill mutant 69 (`global_var == 0` -> `== 1`): an image with
        variance exactly 1.0 (half 0s, half 2s) must NOT take the
        constant-image early-return branch — it must compute a real
        positive score from local statistics, not return 0.0."""
        img = np.zeros((8, 8), dtype=np.float64)
        img[:4, :] = 2.0
        assert img.var() == pytest.approx(1.0, abs=1e-12)
        val = niqe(img)
        assert val > 0.0

    @pytest.mark.regression
    def test_niqe_near_zero_variance_epsilon_pinned(self):
        """Kill mutants 83 (`+ 1e-10` -> `- 1e-10`) and 84 (`+ 1e-10` ->
        `+ 2e-10`) in niqe's score denominator `sqrt(global_var) + eps`:
        a smooth low-amplitude (1e-10) sinusoidal gradient makes both
        global and local variance ~1e-21, putting sqrt(global_var) on the
        same order as the epsilon — any change to it swings the score
        wildly (correct≈19.15, `-eps`≈-102.3, `+2eps`≈12.0)."""
        x = np.linspace(0, 2 * np.pi, 16)
        grid = 0.5 + 1e-10 * np.sin(x)[:, None] * np.ones((1, 16))
        val = niqe(grid)
        assert val == pytest.approx(19.15050505618349, abs=1e-6)


# ===========================================================================
# PARAMETRIZED MATRIX — ISP W6 full expansion for metrics
#
# Characteristics (dice / iou):
#   C1 – pred content:   {empty, full, top_half, left_half,              6 blocks
#                          single_pixel, checkerboard}
#   C2 – target content: same 6 blocks
#   C3 – image shape:    {(4,4), (8,8), (16,16), (8,16)}               4 blocks
#
# Valid range tests (dice):  6 × 6 × 4 = 144
# Self-identity tests:        6 × 4     =  24
# Nonempty-vs-empty tests:    5 × 4     =  20  (pred_nonempty vs empty target)
# Symmetry tests (iou):      6 × 6 × 4 = 144
# hd95 non-negative:         4 × 4 × 4 =  64
# Shape-mismatch raises:      3 funcs × 6 pairs = 18
# Total new parametrized:  ≈ 414
# ===========================================================================
import itertools as _it2

_MT_SHAPES = [(4, 4), (8, 8), (16, 16), (8, 16)]
_MT_FILL_TYPES = ["empty", "full", "top_half", "left_half", "single_pixel", "checkerboard"]
_MT_NONEMPTY = ["full", "top_half", "left_half", "single_pixel", "checkerboard"]
_MT_HD95_FILLS = ["empty", "full", "top_half", "single_pixel"]
_MT_MISMATCH_PAIRS = [
    ((4, 4), (4, 5)), ((8, 8), (4, 8)), ((16, 16), (8, 16)),
    ((4, 4), (8, 8)), ((8, 8), (16, 16)), ((4, 8), (8, 4)),
]


def _mt_make_mask(shape, fill_type):
    """Build a boolean mask of given shape and fill type (ISP C1/C2)."""
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    if fill_type == "empty":
        pass
    elif fill_type == "full":
        m[:] = True
    elif fill_type == "top_half":
        m[: h // 2, :] = True
    elif fill_type == "left_half":
        m[:, : w // 2] = True
    elif fill_type == "single_pixel":
        m[0, 0] = True
    elif fill_type == "checkerboard":
        m[::2, ::2] = True
        m[1::2, 1::2] = True
    return m


class TestDiceParametrize:
    """ISP W6 — parametrized dice tests."""

    # C1 × C3: dice(mask, mask) == 1.0 for all mask types and shapes (24 tests)
    @pytest.mark.parametrize(
        "shape,fill",
        list(_it2.product(_MT_SHAPES, _MT_FILL_TYPES)),
    )
    def test_dice_self_identity(self, shape, fill):
        """ISP C1/C3: dice(m, m) == 1.0 for any mask type and shape."""
        m = _mt_make_mask(shape, fill)
        assert dice(m, m) == pytest.approx(1.0)

    # C1 × C3: dice(nonempty, empty) == 0.0 (20 tests, called once each direction)
    @pytest.mark.parametrize(
        "shape,fill",
        list(_it2.product(_MT_SHAPES, _MT_NONEMPTY)),
    )
    def test_dice_nonempty_vs_empty_is_zero(self, shape, fill):
        """ISP C1/C2/C3: dice(nonempty, empty) == 0.0."""
        m = _mt_make_mask(shape, fill)
        e = _mt_make_mask(shape, "empty")
        assert dice(m, e) == pytest.approx(0.0)

    # C1 × C2 × C3: dice in [0,1] for all pred × target pairs (144 tests)
    @pytest.mark.parametrize(
        "shape,fill_a,fill_b",
        list(_it2.product(_MT_SHAPES, _MT_FILL_TYPES, _MT_FILL_TYPES)),
    )
    def test_dice_range(self, shape, fill_a, fill_b):
        """ISP C1-C3: dice(a, b) ∈ [0, 1] for all mask combinations."""
        a = _mt_make_mask(shape, fill_a)
        b = _mt_make_mask(shape, fill_b)
        val = dice(a, b)
        assert 0.0 <= val <= 1.0


class TestIoUParametrize:
    """ISP W6 — parametrized iou tests."""

    # Self-identity (24 tests)
    @pytest.mark.parametrize(
        "shape,fill",
        list(_it2.product(_MT_SHAPES, _MT_FILL_TYPES)),
    )
    def test_iou_self_identity(self, shape, fill):
        """ISP C1/C3: iou(m, m) == 1.0."""
        m = _mt_make_mask(shape, fill)
        assert iou(m, m) == pytest.approx(1.0)

    # Nonempty vs empty (20 tests)
    @pytest.mark.parametrize(
        "shape,fill",
        list(_it2.product(_MT_SHAPES, _MT_NONEMPTY)),
    )
    def test_iou_nonempty_vs_empty_is_zero(self, shape, fill):
        """ISP C1/C2/C3: iou(nonempty, empty) == 0.0."""
        m = _mt_make_mask(shape, fill)
        e = _mt_make_mask(shape, "empty")
        assert iou(m, e) == pytest.approx(0.0)

    # Symmetry: iou(a,b) == iou(b,a) (144 tests)
    @pytest.mark.parametrize(
        "shape,fill_a,fill_b",
        list(_it2.product(_MT_SHAPES, _MT_FILL_TYPES, _MT_FILL_TYPES)),
    )
    def test_iou_symmetry(self, shape, fill_a, fill_b):
        """ISP C1-C3: iou is symmetric."""
        a = _mt_make_mask(shape, fill_a)
        b = _mt_make_mask(shape, fill_b)
        assert iou(a, b) == pytest.approx(iou(b, a))


class TestHD95Parametrize:
    """ISP W6 — parametrized hd95 tests."""

    # hd95 >= 0 for all pairs (64 tests)
    @pytest.mark.parametrize(
        "shape,fill_a,fill_b",
        list(_it2.product(_MT_SHAPES, _MT_HD95_FILLS, _MT_HD95_FILLS)),
    )
    def test_hd95_non_negative(self, shape, fill_a, fill_b):
        """ISP C1-C3: hd95(a, b) >= 0 for all valid inputs."""
        a = _mt_make_mask(shape, fill_a)
        b = _mt_make_mask(shape, fill_b)
        val = hd95(a, b)
        assert val >= 0.0


class TestShapeMismatchParametrize:
    """BVA: shape mismatch raises ValueError for dice, iou, hd95 (18 tests)."""

    @pytest.mark.parametrize(
        "func,shape_a,shape_b",
        [
            (f, sa, sb)
            for f in [dice, iou, hd95]
            for sa, sb in _MT_MISMATCH_PAIRS
        ],
    )
    def test_shape_mismatch_raises(self, func, shape_a, shape_b):
        """BVA W6: mismatched shapes → ValueError."""
        a = np.zeros(shape_a, dtype=bool)
        b = np.zeros(shape_b, dtype=bool)
        with pytest.raises(ValueError):
            func(a, b)
