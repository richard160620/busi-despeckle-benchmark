"""Statistical tests and result-table utilities.

ISP Characteristics (W6):
  C1 - sample sizes: {equal-length, mismatched, too-few (<4)}
  C2 - data distribution: {identical, slightly different, very different}
  C3 - correction method: {bonferroni, fdr_bh, none}
"""
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests


def wilcoxon_compare(a, b):
    """Paired Wilcoxon signed-rank test.

    Args:
        a, b: array-like of the same length (>= 4 samples recommended).

    Returns:
        (statistic, p_value) tuple.

    Raises:
        ValueError: if len(a) != len(b) or len(a) < 4.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"Arrays must have equal length, got {len(a)} and {len(b)}"
        )
    if len(a) < 4:
        raise ValueError(
            f"At least 4 samples required, got {len(a)}"
        )

    diff = a - b
    if np.all(diff == 0):
        # All differences zero: no evidence of difference, p = 1.0
        return 0.0, 1.0

    result = wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)


def friedman_test(*groups):
    """Friedman test for k related samples.

    Args:
        *groups: three or more equal-length array-likes (scipy requires k >= 3).

    Returns:
        (statistic, p_value) tuple.

    Raises:
        ValueError: if fewer than 3 groups or unequal lengths.
    """
    if len(groups) < 3:
        raise ValueError(
            f"At least 3 groups required, got {len(groups)}"
        )
    groups = [np.asarray(g, dtype=float) for g in groups]
    lengths = [len(g) for g in groups]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All groups must have equal length, got lengths {lengths}"
        )

    result = friedmanchisquare(*groups)
    return float(result.statistic), float(result.pvalue)


def correct_pvalues(pvalues, method="bonferroni"):
    """Multiple-comparison p-value correction.

    Args:
        pvalues: list of raw p-values.
        method: 'bonferroni' or 'fdr_bh'.

    Returns:
        list of corrected p-values.

    Raises:
        ValueError: unsupported method.
    """
    if method not in ("bonferroni", "fdr_bh"):
        raise ValueError(
            f"Unsupported correction method '{method}'. "
            f"Choose 'bonferroni' or 'fdr_bh'."
        )
    _, corrected, _, _ = multipletests(pvalues, method=method)
    return list(corrected)


def make_result_table(records):
    """Convert a list of result dicts to a CSV-serialisable list of dicts.

    Args:
        records: list of dicts with keys method, label, dice, iou, hd95.

    Returns:
        Same list sorted by method then label.
    """
    return sorted(records, key=lambda r: (r.get("method", ""), r.get("label", "")))
