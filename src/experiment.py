"""Experiment pipeline: data -> despeckle -> segment -> metrics.

ISP Characteristics (W6):
  Experiment matrix characteristics:
    C1 - despeckle method: {none, median, lee, frost, srad, nlm}
    C2 - lesion type:       {benign, malignant, normal}
    C3 - noise level:       {low, medium, high}  (simulated via param sweep)

Graph Coverage edges for run() (W7):
[START] -> validate_config -> [invalid? -> ValueError]
-> load_data -> [empty? -> ValueError]
-> for each (image, mask, label):
     -> despeckle_step -> segment_step -> metrics_step
     -> [step_error? -> propagate with context]
-> aggregate_results -> [END]
"""
import numpy as np

from src.despeckle import despeckle
from src.metrics import dice, hd95, iou
from src.segment import postprocess_mask, predict


_VALID_METHODS = frozenset(["none", "median", "lee", "frost", "srad", "nlm"])


def _validate_config(config):
    """Edge: validate_config -> ValueError on bad input."""
    methods = config.get("methods", [])
    if not methods:
        raise ValueError("config['methods'] must be a non-empty list")
    for m in methods:
        if m not in _VALID_METHODS:
            raise ValueError(f"Unknown despeckle method '{m}'")
    subset = config.get("data_subset", [])
    if not subset:
        raise ValueError("config['data_subset'] must be a non-empty list")


def _load_or_generate(config):
    """Edge: load_data; returns list of (image, mask, label)."""
    data_root = config.get("data_root")
    if data_root is not None:
        from src.data import load_busi
        records = load_busi(data_root)
        subset = set(config.get("data_subset", []))
        return [(img, mask, lbl) for img, mask, lbl in records if lbl in subset]

    # Synthetic data for unit tests (no GPU, no BUSI needed)
    rng = np.random.default_rng(0)
    n = config.get("synthetic_n", 2)
    h, w = config.get("synthetic_size", (32, 32))
    subset = config.get("data_subset", ["benign"])
    records = []
    for label in subset:
        for _ in range(n):
            image = rng.integers(0, 256, (h, w), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
            records.append((image, mask.astype(np.uint8) * 255, label))
    return records


def run(config):
    """Execute the full experiment pipeline.

    Args:
        config: dict with keys:
            - methods: list of despeckle method names
            - data_root: path to BUSI dataset root (or None for synthetic)
            - data_subset: list of labels to include, e.g. ['benign', 'malignant']
            - model: segmentation model or mock callable
            - despeckle_params: dict mapping method name -> param dict
            - min_lesion_area: int, minimum valid mask area in pixels
            - synthetic_n: int (default 2), images per label when data_root is None
            - synthetic_size: (H, W) tuple (default (32,32)) for synthetic images

    Returns:
        List of result dicts (CSV-serialisable via stats.make_result_table).

    Raises:
        ValueError: invalid config or empty dataset.
        RuntimeError: pipeline step failure with context.

    Graph Coverage edges (W7):
        validate_config -> [invalid -> ValueError]
        load_data -> records
        for method in methods:
          for (image, mask, label) in records:
            despeckle_step -> segment_step -> metrics_step -> aggregate
    """
    # Edge: validate_config
    _validate_config(config)

    model = config["model"]
    despeckle_params = config.get("despeckle_params", {})
    min_lesion_area = config.get("min_lesion_area", 1)

    # Edge: load_data
    records = _load_or_generate(config)
    if not records:
        raise ValueError("No data loaded — check data_root and data_subset")

    results = []

    for method in config["methods"]:
        params = despeckle_params.get(method, {"window_size": 3})

        for image, mask_raw, label in records:
            try:
                # Edge: despeckle_step
                img_float = image.astype(np.float32) / 255.0
                denoised = despeckle(img_float, method=method, **params)

                # Edge: segment_step
                prob_map = predict(model, denoised)
                pred_mask = postprocess_mask(prob_map, threshold=0.5)

                # Ground-truth mask: binarise
                gt_mask = np.asarray(mask_raw) > 127

                # Edge: metrics_step
                d = dice(pred_mask, gt_mask)
                j = iou(pred_mask, gt_mask)
                h = hd95(pred_mask, gt_mask)

                results.append({
                    "method": method,
                    "label": label,
                    "dice": d,
                    "iou": j,
                    "hd95": h,
                })

            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline failed for method='{method}', label='{label}': {exc}"
                ) from exc

    # Edge: aggregate_results
    return results
