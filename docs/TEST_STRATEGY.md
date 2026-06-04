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

*(Fill in after Step 8.)*

| Metric | Target | Actual |
|--------|--------|--------|
| Line coverage | ≥ 90 % | TBD |
| Branch coverage | ≥ 85 % | TBD |
| Mutation score | ≥ 80 % | TBD |

### How to run mutation testing

```bash
# Run mutmut (configured in setup.cfg)
mutmut run

# View results
mutmut results

# See surviving mutants
mutmut show
```
