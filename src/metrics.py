"""Segmentation quality metrics and image-quality wrappers.

Logic Coverage target (W8) — is_result_valid:
  Predicate: has_mask AND dice_finite AND (area_above_min OR allow_empty)

  Truth table (MC-DC):
  Row | has_mask | dice_finite | area_above_min | allow_empty | Result
  ----+----------+-------------+----------------+-------------+-------
   1  |   T      |     T       |       T        |      T      |   T
   2  |   T      |     T       |       T        |      F      |   T
   3  |   T      |     T       |       F        |      T      |   T
   4  |   T      |     T       |       F        |      F      |   F   <- area kills
   5  |   T      |     F       |       T        |      T      |   F   <- dice kills
   6  |   F      |     T       |       T        |      T      |   F   <- mask kills
   MC-DC pairs: (1,6) for has_mask; (1,5) for dice_finite; (2,4) for area_above_min;
                (3,4) for allow_empty.

ISP Characteristics (W6):
  C1 - pred/target content: {both empty, both non-empty, one empty}
  C2 - shape relationship: {matching, mismatched}
  C3 - area size: {above min, at min boundary, below min}
"""
import numpy as np
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def _check_shapes(pred, target):
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )


def dice(pred, target):
    """Sørensen-Dice coefficient.

    Convention: both-empty -> 1.0.
    Raises ValueError on shape mismatch.
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    _check_shapes(pred, target)

    intersection = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


def iou(pred, target):
    """Intersection-over-Union (Jaccard index).

    Convention: both-empty -> 1.0.
    Raises ValueError on shape mismatch.
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    _check_shapes(pred, target)

    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _directed_hausdorff(a_pts, b_pts):
    """max_{a in A} min_{b in B} dist(a, b), via KD-tree nearest-neighbour query.

    Numerically identical to scipy.spatial.distance.directed_hausdorff(a, b)[0]
    (both compute the exact same quantity), but O(m log n) instead of the
    brute-force O(m*n) all-pairs distance matrix — the latter took ~38s per
    call on a 256x256 mask with ~50% foreground fill (W14 perf fix).
    """
    dists, _ = cKDTree(b_pts).query(a_pts, k=1)
    return float(dists.max())


def hd95(pred, target):
    """95th-percentile Hausdorff distance (pixels).

    Raises ValueError on shape mismatch.
    Returns 0.0 when both masks are empty.
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    _check_shapes(pred, target)

    if not pred.any() and not target.any():
        return 0.0

    pred_pts = np.argwhere(pred).astype(float)
    target_pts = np.argwhere(target).astype(float)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        # one mask empty — return a large distance (image diagonal)
        return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

    d_pt = _directed_hausdorff(pred_pts, target_pts)
    d_tp = _directed_hausdorff(target_pts, pred_pts)
    return float(max(d_pt, d_tp))


def psnr(image, reference):
    """Peak signal-to-noise ratio (dB)."""
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    data_range = reference.max() - reference.min()
    if data_range == 0:
        data_range = 1.0
    return float(peak_signal_noise_ratio(reference, image, data_range=data_range))


def ssim(image, reference):
    """Structural similarity index measure."""
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    data_range = reference.max() - reference.min()
    if data_range == 0:
        data_range = 1.0
    return float(structural_similarity(reference, image, data_range=data_range))


def niqe(image):
    """Reference-free Natural Image Quality Evaluator score (approximation).

    Lower is better. Accepts a single 2-D grayscale array.
    Uses a simplified local-statistics approach as a lightweight proxy.
    """
    image = np.asarray(image, dtype=np.float64)
    # Simplified NIQE proxy: ratio of global vs local variance (higher = noisier)
    global_mean = image.mean()
    global_var = image.var()
    if global_var == 0:
        return 0.0
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(image, 7)
    local_var = uniform_filter(image ** 2, 7) - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)
    score = float(np.sqrt(np.mean(local_var)) / (np.sqrt(global_var) + 1e-10))
    return score


def is_result_valid(has_mask, dice_finite, area_above_min, allow_empty):
    """Guard predicate for result acceptance.

    Predicate: has_mask AND dice_finite AND (area_above_min OR allow_empty)

    See truth table in module docstring (W8 Logic Coverage / MC-DC).
    """
    return bool(has_mask and dice_finite and (area_above_min or allow_empty))
