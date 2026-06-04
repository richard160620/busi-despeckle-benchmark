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


def dice(pred, target):
    """Sørensen-Dice coefficient.

    Convention: both-empty -> 1.0.
    Raises ValueError on shape mismatch.
    """
    raise NotImplementedError


def iou(pred, target):
    """Intersection-over-Union (Jaccard index).

    Convention: both-empty -> 1.0.
    Raises ValueError on shape mismatch.
    """
    raise NotImplementedError


def hd95(pred, target):
    """95th-percentile Hausdorff distance (pixels).

    Raises ValueError on shape mismatch.
    Returns 0.0 when both masks are empty.
    """
    raise NotImplementedError


def psnr(image, reference):
    """Peak signal-to-noise ratio (dB)."""
    raise NotImplementedError


def ssim(image, reference):
    """Structural similarity index measure."""
    raise NotImplementedError


def niqe(image):
    """Reference-free Natural Image Quality Evaluator score.

    Lower is better.  Accepts a single 2-D grayscale array.
    """
    raise NotImplementedError


def is_result_valid(has_mask, dice_finite, area_above_min, allow_empty):
    """Guard predicate for result acceptance.

    Predicate: has_mask AND dice_finite AND (area_above_min OR allow_empty)

    See truth table in module docstring (W8 Logic Coverage / MC-DC).
    """
    raise NotImplementedError
