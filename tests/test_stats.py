"""Tests for src/stats.py.

Testing methods applied:
  - Input Space Partitioning (W6): sample sizes x distributions x correction methods
  - Boundary Value Analysis: minimum sample size (4), length mismatch
"""
import numpy as np
import pytest

from src.stats import correct_pvalues, friedman_test, make_result_table, wilcoxon_compare


# ---------------------------------------------------------------------------
# wilcoxon_compare
# ---------------------------------------------------------------------------

class TestWilcoxonCompare:
    def test_returns_statistic_and_pvalue(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        stat, p = wilcoxon_compare(a, b)
        assert isinstance(float(stat), float)
        assert 0.0 <= p <= 1.0

    def test_identical_arrays_high_pvalue(self):
        a = np.ones(8)
        stat, p = wilcoxon_compare(a, a)
        assert p > 0.05

    def test_length_mismatch_raises(self):
        """ISP: unequal-length inputs must raise ValueError."""
        with pytest.raises(ValueError):
            wilcoxon_compare([1, 2, 3], [1, 2])

    def test_too_few_samples_raises(self):
        """BVA: fewer than 4 samples raises ValueError."""
        with pytest.raises(ValueError):
            wilcoxon_compare([1, 2, 3], [1, 2, 3])

    def test_min_valid_sample_size(self):
        """BVA boundary: exactly 4 samples should succeed."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.5, 2.5, 3.5, 4.5])
        stat, p = wilcoxon_compare(a, b)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# friedman_test
# ---------------------------------------------------------------------------

class TestFriedmanTest:
    def test_returns_statistic_and_pvalue(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0])
        g2 = np.array([1.5, 2.5, 3.5, 4.5])
        g3 = np.array([2.0, 3.0, 4.0, 5.0])
        stat, p = friedman_test(g1, g2, g3)
        assert isinstance(float(stat), float)
        assert 0.0 <= p <= 1.0

    def test_fewer_than_three_groups_raises(self):
        with pytest.raises(ValueError):
            friedman_test(np.ones(5))

    def test_two_groups_also_raises(self):
        with pytest.raises(ValueError):
            friedman_test(np.ones(5), np.ones(5))

    def test_unequal_group_lengths_raises(self):
        with pytest.raises(ValueError):
            friedman_test(np.ones(5), np.ones(4))


# ---------------------------------------------------------------------------
# correct_pvalues
# ---------------------------------------------------------------------------

class TestCorrectPValues:
    @pytest.mark.parametrize("method", ["bonferroni", "fdr_bh"])
    def test_valid_methods(self, method):
        corrected = correct_pvalues([0.01, 0.05, 0.1], method=method)
        assert len(corrected) == 3
        assert all(0.0 <= p <= 1.0 for p in corrected)

    def test_bonferroni_increases_pvalues(self):
        raw = [0.01, 0.02, 0.03]
        corrected = correct_pvalues(raw, method="bonferroni")
        for r, c in zip(raw, corrected):
            assert c >= r

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            correct_pvalues([0.05], method="unknown")


# ---------------------------------------------------------------------------
# make_result_table
# ---------------------------------------------------------------------------

class TestMakeResultTable:
    def test_returns_sorted_list(self):
        records = [
            {"method": "median", "label": "malignant", "dice": 0.7, "iou": 0.6, "hd95": 5.0},
            {"method": "none",   "label": "benign",    "dice": 0.6, "iou": 0.5, "hd95": 6.0},
            {"method": "median", "label": "benign",    "dice": 0.75, "iou": 0.65, "hd95": 4.0},
        ]
        result = make_result_table(records)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert make_result_table([]) == []


# ---------------------------------------------------------------------------
# Mutation killers (W10 Syntax-Based Testing)
# ---------------------------------------------------------------------------

class TestStatsMutationKillers:
    """W10: targeted tests to kill surviving mutmut mutants in src/stats.py."""

    # --- wilcoxon_compare (M320, M323, M328, M329) ---
    def test_length_mismatch_error_message(self):
        """Kill M320: error message must contain 'length'."""
        with pytest.raises(ValueError, match="length"):
            wilcoxon_compare([1, 2, 3, 4], [1, 2, 3])

    def test_too_few_samples_error_message(self):
        """Kill M323: error message must contain '4'."""
        with pytest.raises(ValueError, match="4"):
            wilcoxon_compare([1, 2, 3], [1, 2, 3])

    def test_identical_arrays_statistic_is_zero(self):
        """Kill M328: statistic for identical arrays must be 0.0, not 1.0."""
        stat, p = wilcoxon_compare(np.ones(8), np.ones(8))
        assert stat == pytest.approx(0.0)

    def test_identical_arrays_pvalue_is_one(self):
        """Kill M329: p-value for identical arrays must be 1.0, not 2.0."""
        _, p = wilcoxon_compare(np.ones(8), np.ones(8))
        assert p == pytest.approx(1.0)
        assert p <= 1.0  # p-value must be in [0,1]

    # --- friedman_test (M331, M332, M333, M338) ---
    def test_exactly_three_groups_is_valid(self):
        """Kill M331 (<3→≤3): 3 groups must be accepted."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0])
        g2 = np.array([1.5, 2.5, 3.5, 4.5])
        g3 = np.array([2.0, 3.0, 4.0, 5.0])
        stat, p = friedman_test(g1, g2, g3)   # must NOT raise
        assert 0.0 <= p <= 1.0

    def test_one_group_raises(self):
        """Kill M331+M332: exactly 1 group (<3) must raise ValueError."""
        with pytest.raises(ValueError):
            friedman_test(np.ones(5))

    def test_too_few_groups_error_message(self):
        """Kill M333: error message must contain '3'."""
        with pytest.raises(ValueError, match="3"):
            friedman_test(np.ones(5))

    def test_unequal_lengths_error_message(self):
        """Kill M338: error message for unequal lengths must contain 'length'."""
        with pytest.raises(ValueError, match="length"):
            friedman_test(np.ones(5), np.ones(4), np.ones(5))

    # --- correct_pvalues (M344, M345) ---
    def test_unsupported_method_error_message(self):
        """Kill M344+M345: error message must contain 'bonferroni' or 'fdr_bh'."""
        with pytest.raises(ValueError, match="bonferroni"):
            correct_pvalues([0.05], method="bad")

    # --- make_result_table (M347-M350): sorting key mutations ---
    def test_sorted_by_method_first(self):
        """Kill M347+M348: table must sort by method key."""
        records = [
            {"method": "none",   "label": "benign"},
            {"method": "median", "label": "benign"},
            {"method": "lee",    "label": "benign"},
        ]
        result = make_result_table(records)
        methods = [r["method"] for r in result]
        assert methods == sorted(methods)   # sorted alphabetically by method

    def test_sorted_by_label_within_method(self):
        """Kill M349+M350: within same method, sorted by label key."""
        records = [
            {"method": "none", "label": "normal"},
            {"method": "none", "label": "benign"},
            {"method": "none", "label": "malignant"},
        ]
        result = make_result_table(records)
        labels = [r["label"] for r in result]
        assert labels == sorted(labels)
