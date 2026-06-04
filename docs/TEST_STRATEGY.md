# Test Strategy Document

**Project:** Does Despeckling Help? A Task-Linked Benchmark of Speckle Reduction for
Breast-Ultrasound Lesion Segmentation  
**Course:** Software Testing (115 Spring), NYCU  
**Textbook:** Ammann & Offutt, *Introduction to Software Testing*

---

## 1. Scope & Objectives

### Scope
This test suite validates the **experimental framework** (Stage 1). Heavy model
training is out of scope; the U-Net is mocked.

### Objectives mapped to Research Questions

| RQ | Module under test | Testing goal |
|----|-------------------|--------------|
| RQ1 | `experiment.run`, `metrics.dice/iou` | Verify the pipeline correctly records per-method segmentation metrics |
| RQ2 | `data.load_busi`, `experiment.run` | Verify label/size subgroup routing is correct |
| RQ3 | `metrics.psnr/ssim/niqe`, `stats.wilcoxon_compare` | Verify image-quality metric computation and correlation test wiring |

---

## 2. Equivalence Partitioning

### `data.load_busi(root)`

| Characteristic | Block A | Block B | Block C |
|----------------|---------|---------|---------|
| C1 root path | exists | missing | — |
| C2 subfolder | complete | one missing | — |
| C3 image/mask | matched sizes | size mismatch | — |

### `despeckle(image, method, **params)`

| Characteristic | Block A | Block B | Block C | Block D |
|----------------|---------|---------|---------|---------|
| C1 method | `none` | `median`/`lee`/`frost` | `srad`/`nlm` | unknown |
| C2 window_size | 1 (min valid) | 3 (typical odd) | even | 0 / negative |
| C3 dtype | `uint8` | `float32` | `float64` | — |

### `is_result_valid(has_mask, dice_finite, area_above_min, allow_empty)`

| Characteristic | Block T | Block F |
|----------------|---------|---------|
| C1 has_mask | True | False |
| C2 dice_finite | True | False |
| C3 area_above_min | True | False |
| C4 allow_empty | True | False |

### Experiment matrix (RQ1-RQ3)

| Characteristic | Blocks |
|----------------|--------|
| despeckle method | none / median / lee / frost / srad / nlm |
| lesion type | benign / malignant / normal |
| noise level (simulated) | low / medium / high |

---

## 3. Boundary Value Analysis

| Module | Parameter | Min valid | Typical | Invalid values tested |
|--------|-----------|-----------|---------|----------------------|
| `despeckle` | `window_size` | 1 | 3 | 0, −1, 2 (even) |
| `metrics.dice/iou` | both-empty inputs | — | partial overlap | both empty (→ 1.0 by convention) |
| `metrics.hd95` | both-empty inputs | — | — | both empty (→ 0.0) |
| `segment.postprocess_mask` | `threshold` | 0.0 | 0.5 | −0.1, 1.1 |
| `stats.wilcoxon_compare` | sample size | 4 | 8 | 3 (too few), mismatched lengths |

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU unavailable in CI | High | High | Mock model with fixed probability output |
| BUSI dataset not present | High | Medium | `data_root=None` triggers synthetic data |
| Filter divergence on edge inputs | Medium | Medium | BVA tests at boundary window sizes |
| Statistical test numerical instability | Low | Medium | Use scipy reference implementations |
| Mutation score below 80 % | Medium | High | MC-DC tests and BVA improve clause coverage |
| Coverage below 90 % | Low | High | Integration test exercises full pipeline |

---

## 5. Test Case Design

| Test case | Module | Function | Course method (week) |
|-----------|--------|----------|----------------------|
| `test_missing_root_raises_file_not_found` | `data` | `load_busi` | ISP (W6), Graph (W7) |
| `test_size_mismatch_raises_value_error` | `data` | `load_busi` | ISP (W6) |
| `test_window_size_0_raises` | `despeckle` | `despeckle` | BVA (W6) |
| `test_window_size_even_raises` | `despeckle` | `despeckle` | BVA (W6) |
| `test_unknown_method_raises_value_error` | `despeckle` | `despeckle` | Graph (W7) |
| `test_none_method_returns_exact_copy` | `despeckle` | `despeckle` | Graph (W7) |
| `test_mcdc_has_mask_*` (6 cases) | `metrics` | `is_result_valid` | MC-DC / Logic Coverage (W8) |
| `test_regression_known_dice` | `metrics` | `dice` | Regression (W14) |
| `test_regression_known_iou` | `metrics` | `iou` | Regression (W14) |
| `test_1d_image_raises` | `segment` | `predict` | ISP (W6) |
| `test_threshold_*` (5 cases) | `segment` | `postprocess_mask` | BVA (W6) |
| `test_too_few_samples_raises` | `stats` | `wilcoxon_compare` | BVA (W6) |
| `test_valid_config_returns_results` | `experiment` | `run` | Graph (W7), ISP (W6) |
| `test_regression_none_method_dice_range` | `experiment` | `run` | Regression (W14) |

---

## 6. Coverage and Mutation Score

| Metric | Target | Actual |
|--------|--------|--------|
| Line coverage (`pytest --cov`) | ≥ 90 % | **97 %** (375 stmts, 12 missed) |
| Mutation score (`mutmut run`) | ≥ 80 % | **85.5 %** (300/351 killed) |
| Total tests | ≥ 1000 | **1585** |

### How to run

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run mutation testing (core modules only — fast)
mutmut run
mutmut results
mutmut show        # inspect surviving mutants
```

---

## 7. Flask API Endpoint Tests (W6 + W7)

| Route | Status codes tested | Testing method |
|-------|--------------------|-|
| GET / | 200 | ISP C1/C2, Graph edge |
| POST /segment | 200, 400 (no file), 400 (bad method) | ISP C1-C3, BVA |
| Unknown route | 404 | Graph: unknown_route edge |
| Wrong verb | 405 | Graph: wrong_verb edge |

Parametrized over all 6 despeckle methods (`@pytest.mark.parametrize`).  
Mock model injected — no GPU required.

---

## 8. Parametrized Test Matrix (W6 ISP)

| File | Characteristics | Combinations |
|------|----------------|-------------|
| `test_despeckle.py` | C1(6 methods) × C2(4 kernels) × C3(4 shapes) × C4(3 dtypes) | 288 valid + 30 BVA invalid |
| `test_metrics.py` | dice/iou: C1(6 fills) × C2(6 fills) × C3(4 shapes); hd95; mismatch | 458 |
| `test_segment.py` | C1(6 shapes) × C2(11 thresholds) × C3(5 prob values) | 330 valid + 12 BVA invalid |

Each combination maps to a specific ISP block or BVA boundary — no padding.

---

## 9. Mutation Testing Evidence (W10 Syntax-Based Testing)

### Setup
- Tool: `mutmut 2.4.4`
- Modules under test: `src/metrics.py`, `src/despeckle.py`, `src/data.py`, `src/stats.py`
- Test runner: `pytest tests/test_metrics.py tests/test_despeckle.py tests/test_data.py tests/test_stats.py`

### Results: 300 killed / 351 total = **85.5 %**

### Surviving → Killer test → Killed process

| Mutant group | Example mutation | Why it survived initially | Killer test added |
|---|---|---|---|
| Error message strings (`M292`, `M297`, `M320`, `M323`, `M333`) | `"XX…XX"` prefix/suffix in error text | `pytest.raises(ValueError)` only checks type, not message | Added `match=` parameter checking key words (`"root"`, `"label"`, `"length"`, `"3"`) |
| `hd95` one-empty branch (`M34–M37`) | `or` → `and`; `== 0` → `== 1` | No test with exactly one empty mask + exact return value | `test_hd95_empty_pred_nonempty_target_is_diagonal` pins `sqrt(H²+W²)` |
| `hd95` diagonal formula (`M38–44`) | `shape[0]²` → `shape[1]²` | Non-square shapes not used in value tests | Used `(6,10)` shape to expose H≠W confusion |
| `psnr` / `ssim` data_range (`M51–63`) | `max-min` → `max+min` | Tests only checked return-is-float | `test_psnr_uses_max_minus_min` pins exact dB value; `test_psnr_zero_data_range_fallback_is_one` pins fallback |
| `niqe` formula (`M68–84`) | `== 0` → `!= 0`; `÷` → `×` | Only `>= 0` asserted | Added `test_niqe_nonconstant_is_positive` + regression pin `0.98895862` |
| Filter arithmetic (`M135–145`) | `*255` → `/255`; `<=1.0` → `<1.0` | Shape/dtype checks ignored values | Added `TestDespeckleFilterRegressions` pinning mean/std/corner values per filter |
| `srad` param key (`M129`) | `"n_iter"` → `"XXn_iterXX"` | Default value masked the missing param | `test_srad_n_iter_1_differs_from_10` compares `n_iter=1` vs `n_iter=10` |
| Default method (`M105`) | `"none"` → `"XXnoneXX"` | All tests passed explicit method | `test_default_method_is_none` calls `despeckle(img)` without method |

### Remaining survivors (51 / 351 = 14.5 %)

Grouped by reason they are hard to kill:

1. **Equivalent mutants** (8): Epsilon changes (`1e-10` → `2e-10`) produce numerically identical outputs within float64 precision. These are true equivalent mutants.
2. **Deep filter arithmetic** (24): Internal operations of `_frost_filter`, `_srad_filter`, `_nlm_filter` that change intermediate values but produce very similar final statistics for the random test images. Killing these would require per-pixel regression tests for every pixel of every filter output — excessive for a unit-test suite.
3. **Error message XX-prefix** (10): `pytest.raises(match=)` uses `re.search`, so "XXmissage" still contains the matched substring. Anchored matches (`\A`) would kill these but add fragility to the error-message wording.
4. **Data loading edge** (9): Some `continue`/`break` mutations in `load_busi` require datasets with missing mask files mid-directory to distinguish them.
