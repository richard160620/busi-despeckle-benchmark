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
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import median as skimage_median
from skimage.morphology import square
from skimage.restoration import denoise_nl_means

_SUPPORTED_METHODS = frozenset(["none", "median", "lee", "frost", "srad", "nlm"])


def _validate_window(window_size):
    """Raise ValueError if window_size is not a positive odd integer."""
    if not isinstance(window_size, (int, np.integer)):
        raise ValueError(f"window_size must be an integer, got {type(window_size)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")


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

    Graph dispatch edges (W7):
        [none] -> _none
        [median] -> _median_filter
        [lee]    -> _lee_filter
        [frost]  -> _frost_filter
        [srad]   -> _srad_filter
        [nlm]    -> _nlm_filter
        [other]  -> ValueError
    """
    # Edge: validate method
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported despeckle method '{method}'. "
            f"Choose from {sorted(_SUPPORTED_METHODS)}"
        )

    original_dtype = image.dtype
    img_float = image.astype(np.float64)

    # Edge: none
    if method == "none":
        return image.copy()

    window_size = params.get("window_size", 3)
    _validate_window(window_size)

    # Edge: median
    if method == "median":
        result = _median_filter(img_float, window_size)
    # Edge: lee
    elif method == "lee":
        result = _lee_filter(img_float, window_size)
    # Edge: frost
    elif method == "frost":
        result = _frost_filter(img_float, window_size)
    # Edge: srad
    elif method == "srad":
        n_iter = params.get("n_iter", 10)
        result = _srad_filter(img_float, n_iter)
    # Edge: nlm
    elif method == "nlm":
        result = _nlm_filter(img_float, window_size)

    return result.astype(original_dtype)


# ---------------------------------------------------------------------------
# Filter implementations
# ---------------------------------------------------------------------------

def _median_filter(image, window_size):
    """Median filter via skimage."""
    uint_img = (image * 255).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    filtered = skimage_median(uint_img, square(window_size))
    if image.max() <= 1.0:
        return filtered.astype(np.float64) / 255.0
    return filtered.astype(np.float64)


def _lee_filter(image, window_size):
    """Lee speckle filter (local statistics-based)."""
    img = image.astype(np.float64)
    local_mean = uniform_filter(img, window_size)
    local_sq_mean = uniform_filter(img ** 2, window_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)

    noise_var = np.mean(local_var)
    # Avoid division by zero
    denom = local_var + noise_var
    denom = np.where(denom == 0, 1.0, denom)
    k = local_var / denom
    return local_mean + k * (img - local_mean)


def _frost_filter(image, window_size):
    """Frost speckle filter (exponential damping kernel)."""
    img = image.astype(np.float64)
    half = window_size // 2
    result = np.zeros_like(img)
    padded = np.pad(img, half, mode="reflect")

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            patch = padded[r: r + window_size, c: c + window_size]
            m = patch.mean()
            v = patch.var()
            # damping constant
            k = 1.0 if m == 0 else v / (m ** 2 + 1e-10)
            ys, xs = np.mgrid[-half: half + 1, -half: half + 1]
            weights = np.exp(-k * np.sqrt(ys ** 2 + xs ** 2))
            result[r, c] = np.sum(weights * patch) / (np.sum(weights) + 1e-10)

    return result


def _srad_filter(image, n_iter=10, delta_t=0.1):
    """Speckle Reducing Anisotropic Diffusion (SRAD), simplified."""
    img = image.astype(np.float64)
    for _ in range(n_iter):
        grad_n = np.roll(img, -1, axis=0) - img
        grad_s = np.roll(img, 1, axis=0) - img
        grad_e = np.roll(img, -1, axis=1) - img
        grad_w = np.roll(img, 1, axis=1) - img

        q = (0.5 * (grad_n ** 2 + grad_s ** 2 + grad_e ** 2 + grad_w ** 2)
             / (img ** 2 + 1e-10))
        c = 1.0 / (1.0 + q)

        img = img + delta_t * (
            c * grad_n + np.roll(c, 1, axis=0) * grad_s
            + c * grad_e + np.roll(c, 1, axis=1) * grad_w
        )
    return img


def _nlm_filter(image, window_size):
    """Non-local means denoising via skimage."""
    img = image.astype(np.float64)
    # normalise to [0,1] for skimage
    min_val, max_val = img.min(), img.max()
    rng = max_val - min_val if max_val != min_val else 1.0
    img_norm = (img - min_val) / rng
    filtered_norm = denoise_nl_means(img_norm, patch_size=window_size,
                                     patch_distance=window_size, h=0.1,
                                     fast_mode=True)
    return (filtered_norm * rng + min_val)
