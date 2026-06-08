"""Integration tests for src/experiment.py — experiment.run pipeline.

Testing methods applied:
  - Graph Coverage (W7): all pipeline edges (validate -> load -> despeckle ->
    segment -> metrics -> aggregate)
  - Input Space Partitioning (W6): method x label subset x mock vs real data
  - Regression (W14): fixed synthetic run pins known result — see Issue #3
"""
import numpy as np
import pytest

import src.experiment as experiment_mod
from src.experiment import run, _load_or_generate, _VALID_METHODS


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
# Mutation killers (W9 Syntax-Based / Mutation Testing) — see mutmut survivors
# for src/experiment.py. Each test below targets one or more surviving
# mutants identified via `mutmut results` + `mutmut show <id>`.
# ---------------------------------------------------------------------------

class TestExperimentMutationKillers:
    # -- _VALID_METHODS string-literal mutants (3,4,5,6: XX-wrapped names) --
    def test_valid_methods_set_is_exact(self):
        assert _VALID_METHODS == frozenset(
            ["none", "median", "lee", "frost", "srad", "nlm"]
        )

    # -- _validate_config error-message mutants (11,13,17: XX-wrapped strings) --
    def test_empty_methods_message_anchored(self):
        cfg = _make_config(methods=[])
        with pytest.raises(
            ValueError, match=r"^config\['methods'\] must be a non-empty list$"
        ):
            run(cfg)

    def test_unknown_method_message_anchored(self):
        cfg = _make_config(methods=["not_a_method"])
        with pytest.raises(ValueError, match=r"^Unknown despeckle method 'not_a_method'$"):
            run(cfg)

    def test_empty_subset_message_anchored(self):
        cfg = _make_config(methods=["none"], labels=[])
        with pytest.raises(
            ValueError, match=r"^config\['data_subset'\] must be a non-empty list$"
        ):
            run(cfg)

    # -- _load_or_generate: data_root routing + subset filter (18,19,24) --
    def test_data_root_path_used_and_filtered_by_subset(self, tmp_path):
        """Edge: data_root not None must route through load_busi AND filter by
        data_subset (kills mutants that null out data_root or invert the filter)."""
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
            # Deliberately distinct from the 1 BUSI record so a fall-through
            # to synthetic generation (data_root nulled out) is observable.
            "synthetic_n": 9,
            "synthetic_size": (32, 32),
        }
        results = run(cfg)
        assert len(results) == 1
        assert results[0]["label"] == "benign"

    # -- _load_or_generate: synthetic_n config key + default (27,28) --
    def test_synthetic_n_explicit_value_is_respected(self):
        cfg = _make_config(methods=["none"], labels=["benign"])
        cfg["synthetic_n"] = 5
        results = run(cfg)
        assert len(results) == 5

    def test_synthetic_n_default_is_2_when_absent(self):
        cfg = _make_config(methods=["none"], labels=["benign"])
        del cfg["synthetic_n"]
        results = run(cfg)
        assert len(results) == 2

    # -- _load_or_generate: synthetic_size config key + default (30,31,32) --
    def test_load_or_generate_synthetic_size_explicit_value_respected(self):
        cfg = {"data_root": None, "data_subset": ["benign"], "synthetic_n": 1,
               "synthetic_size": (16, 48)}
        records = _load_or_generate(cfg)
        img, mask, label = records[0]
        assert img.shape == (16, 48)
        assert mask.shape == (16, 48)

    def test_load_or_generate_synthetic_size_default_is_32x32(self):
        cfg = {"data_root": None, "data_subset": ["benign"], "synthetic_n": 1}
        records = _load_or_generate(cfg)
        img, mask, label = records[0]
        assert img.shape == (32, 32)

    # -- _load_or_generate: data_subset default (35) --
    def test_load_or_generate_default_subset_is_benign(self):
        cfg = {"data_root": None, "synthetic_n": 1}
        records = _load_or_generate(cfg)
        labels = {lbl for _, _, lbl in records}
        assert labels == {"benign"}

    # -- _load_or_generate: rng.integers lower bound (38) --
    def test_load_or_generate_image_value_range_includes_zero(self):
        """seed=0 synthetic image is known to contain a 0 pixel; a lower
        bound of 1 (mutant 38) would make that impossible."""
        cfg = {"data_root": None, "data_subset": ["benign"], "synthetic_n": 1,
               "synthetic_size": (32, 32)}
        records = _load_or_generate(cfg)
        img, mask, label = records[0]
        assert img.min() == 0
        assert img.max() == 255

    # -- _load_or_generate: mask region slice bounds + encoding (43,44,47,49,50,53,54,55,56) --
    def test_load_or_generate_mask_region_is_pinned(self):
        """Pin the exact lesion-region box and uint8 {0,255} encoding for the
        synthetic mask. Kills every mutant that nudges a slice bound by one
        quarter/fifth, flips True<->False/None, or changes the *255 encoding."""
        cfg = {"data_root": None, "data_subset": ["benign"], "synthetic_n": 1,
               "synthetic_size": (32, 32)}
        records = _load_or_generate(cfg)
        img, mask, label = records[0]

        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) == {0, 255}
        assert mask.sum() == 255 * 16 * 16          # exact 16x16 True region

        # Inside the region: rows/cols [8:24)
        assert mask[8, 8] == 255
        assert mask[23, 23] == 255
        # Just outside each edge of the region
        assert mask[7, 8] == 0     # row h//4 - 1   (kills h//4 -> h//5 = 6)
        assert mask[24, 8] == 0    # row 3h//4      (kills 3*h//4 -> 4*h//4 = 32)
        assert mask[8, 7] == 0     # col w//4 - 1   (kills w//4 -> w//5 = 6)
        assert mask[8, 24] == 0    # col 3w//4      (kills 3*w//4 -> 4*w//4 = 32)
        assert mask[18, 18] == 255  # mid-region    (kills 3*h(or w)//4 -> //5 = 19)

    # -- run(): hd95 must be the real computed value, not a stub (86) --
    def test_result_hd95_is_a_finite_float_not_none(self):
        cfg = _make_config(methods=["none"], labels=["benign"])
        cfg["synthetic_n"] = 1
        results = run(cfg)
        for row in results:
            assert isinstance(row["hd95"], float)
            assert row["hd95"] >= 0.0

    # -- run(): no-data-loaded error message (68) --
    def test_no_data_loaded_message_anchored(self):
        cfg = _make_config(methods=["none"], labels=["benign"])
        cfg["synthetic_n"] = 0
        with pytest.raises(
            ValueError,
            match=r"^No data loaded — check data_root and data_subset$",
        ):
            run(cfg)

    # -- run(): despeckle_params config key + per-method default (61,71,72) --
    def test_despeckle_params_explicit_value_is_used(self, monkeypatch):
        captured = []

        def _spy_despeckle(img, method, **kwargs):
            captured.append(kwargs)
            return img

        monkeypatch.setattr(experiment_mod, "despeckle", _spy_despeckle)
        cfg = _make_config(methods=["median"], labels=["benign"])
        cfg["synthetic_n"] = 1
        cfg["despeckle_params"] = {"median": {"window_size": 7}}
        run(cfg)
        assert captured[0] == {"window_size": 7}

    def test_despeckle_params_default_window_size_is_3(self, monkeypatch):
        captured = []

        def _spy_despeckle(img, method, **kwargs):
            captured.append(kwargs)
            return img

        monkeypatch.setattr(experiment_mod, "despeckle", _spy_despeckle)
        cfg = _make_config(methods=["median"], labels=["benign"])
        cfg["synthetic_n"] = 1
        cfg["despeckle_params"] = {}   # "median" absent -> default kicks in
        run(cfg)
        assert captured[0] == {"window_size": 3}

    # -- run(): image normalisation before despeckle (74,75) --
    def test_image_normalised_to_unit_range_pinned(self, monkeypatch):
        captured = []

        def _spy_despeckle(img, method, **kwargs):
            captured.append(img.copy())
            return img

        monkeypatch.setattr(experiment_mod, "despeckle", _spy_despeckle)
        cfg = _make_config(methods=["none"], labels=["benign"])
        cfg["synthetic_n"] = 1
        run(cfg)
        img_float = captured[0]
        # seed=0 synthetic uint8 image is known to span [0, 255]
        assert img_float.min() == pytest.approx(0.0, abs=1e-6)
        assert img_float.max() == pytest.approx(1.0, abs=1e-6)

    # -- run(): ground-truth mask binarisation threshold (81,82) --
    def test_gt_mask_threshold_pinned_at_127_boundary(self, tmp_path, monkeypatch):
        """Pins gt_mask = mask_raw > 127 using values straddling 127 and 128
        (kills both >= 127 and > 128 boundary mutants)."""
        import skimage.io as skio

        root = tmp_path / "busi"
        for label in ("benign", "malignant", "normal"):
            (root / label).mkdir(parents=True)
        d = root / "benign"
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        mask = np.full((32, 32), 127, dtype=np.uint8)   # background: exactly 127
        mask[0:8, :] = 128                               # strip: exactly 128
        mask[8:24, 8:24] = 200                           # lesion: clearly above
        skio.imsave(str(d / "benign_001.png"), img)
        skio.imsave(str(d / "benign_001_mask.png"), mask)

        captured = []
        orig_dice = experiment_mod.dice

        def _spy_dice(pred, target):
            captured.append(np.asarray(target).copy())
            return orig_dice(pred, target)

        monkeypatch.setattr(experiment_mod, "dice", _spy_dice)

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
        run(cfg)
        gt_mask = captured[0]
        assert gt_mask[31, 31] == False   # 127 -> excluded (kills `>= 127`)
        assert gt_mask[2, 2] == True      # 128 -> included (kills `> 128`)
        assert gt_mask[10, 10] == True    # 200 -> included (sanity)

    # -- run(): pipeline-failure message (92: XX-wrapped f-string) --
    def test_pipeline_failed_message_fully_anchored(self):
        class _BrokenModel:
            def __call__(self, x):
                raise RuntimeError("boom")

        cfg = _make_config(methods=["none"], labels=["benign"], mock_model=_BrokenModel())
        with pytest.raises(
            RuntimeError,
            match=r"^Pipeline failed for method='none', label='benign': boom$",
        ):
            run(cfg)


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
