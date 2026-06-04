"""Speckle-reduction filters for ultrasound images.

ISP Characteristics (W6):
  C1 - method: {none, median, lee, frost, srad, nlm}
  C2 - kernel/window size: {1 (min-valid), 3 (typical), even (invalid), 0 (invalid), negative (invalid)}
  C3 - image dtype: {uint8, float32, float64}

Graph Coverage dispatch edges (W7):
[START] -> validate_params -> [invalid? -> ValueError]
-> dispatch: [none] -> return copy
              [median] -> _median_filter
              [lee]    -> _lee_filter
              [frost]  -> _frost_filter
              [srad]   -> _srad_filter
              [nlm]    -> _nlm_filter
              [other]  -> ValueError
-> validate_output -> [END]
"""


def despeckle(image, method="none", **params):
    """Apply speckle-reduction filter to a 2-D grayscale image.

    Args:
        image: 2-D numpy array (any float or uint dtype).
        method: one of 'none', 'median', 'lee', 'frost', 'srad', 'nlm'.
        **params: filter-specific parameters (window_size, n_iter, …).

    Returns:
        Filtered image with same shape and dtype as input.

    Raises:
        ValueError: unsupported method or invalid kernel/window size.
    """
    raise NotImplementedError
