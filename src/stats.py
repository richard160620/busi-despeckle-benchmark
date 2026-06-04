"""Statistical tests and result-table utilities.

ISP Characteristics (W6):
  C1 - sample sizes: {equal-length, mismatched, too-few (<4)}
  C2 - data distribution: {identical, slightly different, very different}
  C3 - correction method: {bonferroni, fdr_bh, none}
"""


def wilcoxon_compare(a, b):
    """Paired Wilcoxon signed-rank test.

    Args:
        a, b: array-like of the same length (>= 4 samples recommended).

    Returns:
        (statistic, p_value) tuple.

    Raises:
        ValueError: if len(a) != len(b) or len(a) < 4.
    """
    raise NotImplementedError


def friedman_test(*groups):
    """Friedman test for k related samples.

    Args:
        *groups: two or more equal-length array-likes.

    Returns:
        (statistic, p_value) tuple.

    Raises:
        ValueError: if fewer than 2 groups or unequal lengths.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def make_result_table(records):
    """Convert a list of result dicts to a CSV-serialisable list of dicts.

    Args:
        records: list of dicts with keys method, label, dice, iou, hd95.

    Returns:
        Same list sorted by method then label.
    """
    raise NotImplementedError
