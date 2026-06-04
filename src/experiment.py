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

    Returns:
        List of result dicts (CSV-serialisable via stats.make_result_table).

    Raises:
        ValueError: invalid config or empty dataset.
        RuntimeError: pipeline step failure with context.
    """
    raise NotImplementedError
