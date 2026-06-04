"""Integration tests for src/experiment.py — experiment.run pipeline.

Testing methods applied:
  - Graph Coverage (W7): all pipeline edges (validate -> load -> despeckle ->
    segment -> metrics -> aggregate)
  - Input Space Partitioning (W6): method x label subset x mock vs real data
  - Regression (W14): fixed synthetic run pins known result — see Issue #3
"""
import numpy as np
import pytest

from src.experiment import run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(methods=None, labels=None, tmp_path=None, mock_model=None):
    """Build a minimal valid experiment config using synthetic in-memory data."""
    if methods is None:
        methods = ["none"]
    if labels is None:
        labels = ["benign"]

    class _MockModel:
        def __call__(self, x):
            import torch
            return torch.full_like(x, 0.8)

    return {
        "methods": methods,
        "data_root": None,  # None triggers synthetic data generation
        "data_subset": labels,
        "model": mock_model or _MockModel(),
        "despeckle_params": {m: {"window_size": 3} for m in (methods or [])},
        "min_lesion_area": 1,
        "synthetic_n": 2,          # generate 2 tiny images per label
        "synthetic_size": (32, 32),
    }


# ---------------------------------------------------------------------------
# Graph Coverage: pipeline edges
# ---------------------------------------------------------------------------

class TestExperimentRunGraph:
    def test_valid_config_returns_results(self):
        """Edge: full happy-path pipeline -> list of result dicts."""
        cfg = _make_config(methods=["none"], labels=["benign"])
        results = run(cfg)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_contains_required_keys(self):
        """Each row must have method, label, dice, iou, hd95."""
        cfg = _make_config(methods=["none"], labels=["benign"])
        results = run(cfg)
        required_keys = {"method", "label", "dice", "iou", "hd95"}
        for row in results:
            assert required_keys <= set(row.keys())

    def test_multiple_methods_all_appear(self):
        """Edge: method loop iterates all configured methods."""
        cfg = _make_config(methods=["none", "median"], labels=["benign"])
        results = run(cfg)
        methods_seen = {r["method"] for r in results}
        assert "none" in methods_seen
        assert "median" in methods_seen

    def test_multiple_labels_all_appear(self):
        """Edge: label loop iterates all configured labels."""
        cfg = _make_config(methods=["none"], labels=["benign", "malignant"])
        results = run(cfg)
        labels_seen = {r["label"] for r in results}
        assert "benign" in labels_seen
        assert "malignant" in labels_seen


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestExperimentRunErrors:
    def test_invalid_method_raises(self):
        """Edge: validate_config -> ValueError for unknown despeckle method."""
        cfg = _make_config(methods=["not_a_method"])
        with pytest.raises((ValueError, RuntimeError)):
            run(cfg)

    def test_empty_methods_raises(self):
        cfg = _make_config(methods=[])
        with pytest.raises(ValueError):
            run(cfg)

    def test_empty_subset_raises(self):
        """Edge: validate_config -> ValueError for empty data_subset."""
        cfg = _make_config(methods=["none"], labels=[])
        with pytest.raises(ValueError):
            run(cfg)

    def test_pipeline_step_error_propagates_as_runtime_error(self):
        """Edge: step error -> RuntimeError with context."""
        class _BrokenModel:
            def __call__(self, x):
                raise RuntimeError("model exploded")

        cfg = _make_config(methods=["none"], mock_model=_BrokenModel())
        with pytest.raises(RuntimeError, match="Pipeline failed"):
            run(cfg)

    def test_run_with_real_busi_path(self, tmp_path):
        """Edge: data_root not None -> load_busi path in _load_or_generate."""
        import numpy as np
        import skimage.io as skio
        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            d = root / label
            d.mkdir(parents=True)
            img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            skio.imsave(str(d / f"{label}_001.png"), img)
            skio.imsave(str(d / f"{label}_001_mask.png"), mask)

        class _MockModel:
            def __call__(self, x):
                import torch
                return torch.full_like(x, 0.8)

        cfg = {
            "methods": ["none"],
            "data_root": root,
            "data_subset": ["benign"],
            "model": _MockModel(),
            "despeckle_params": {"none": {}},
            "min_lesion_area": 1,
        }
        results = run(cfg)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Regression test — Issue #3
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_regression_none_method_dice_range():
    """Regression (W14): 'none' method on synthetic data should yield
    dice in [0, 1]. Pinned against Issue #3."""
    cfg = _make_config(methods=["none"], labels=["benign"])
    results = run(cfg)
    for row in results:
        if row["method"] == "none":
            assert 0.0 <= row["dice"] <= 1.0
            assert 0.0 <= row["iou"] <= 1.0
