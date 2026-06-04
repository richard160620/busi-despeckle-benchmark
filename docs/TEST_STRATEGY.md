# Test Strategy Document

**Project:** Does Despeckling Help? A Task-Linked Benchmark of Speckle Reduction for
Breast-Ultrasound Lesion Segmentation  
**Course:** Software Testing (115 Spring), NYCU  
**Textbook:** Ammann & Offutt, *Introduction to Software Testing*

---

## 1. Scope & Objectives

### Scope
This test suite validates the **experimental framework** (Stage 1). Heavy model
training is out of scope; the U-Net is mocked in all CI-facing tests.

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
| C2 subfolder structure | all three label dirs present | one dir missing | — |
| C3 image/mask pair | matched spatial dimensions | size mismatch | — |

### `despeckle(image, method, **params)`

| Characteristic | Block A | Block B | Block C | Block D |
|----------------|---------|---------|---------|---------|
| C1 method | `none` | `median` / `lee` / `frost` | `srad` / `nlm` | unknown string |
| C2 window_size | 1 (min valid odd) | 3 (typical) | even (2, 4, 6) | 0 / negative |
| C3 dtype | `uint8` | `float32` | `float64` | — |

### `metrics.dice / iou (pred, target)`

| Characteristic | Block A | Block B | Block C | Block D |
|----------------|---------|---------|---------|---------|
| C1 pred content | empty | full | partial (various fills) | single pixel |
| C2 target content | empty | full | partial (various fills) | single pixel |
| C3 shape relationship | matching | mismatched | — | — |

### `segment.postprocess_mask(prob_map, threshold)`

| Characteristic | Block A | Block B | Block C | Block D |
|----------------|---------|---------|---------|---------|
| C1 image shape | (4,4) small-sq | (16,16) med-sq | (32,32) large-sq | (8,16) non-sq |
| C2 threshold | 0.0 (all positive) | 0.1 – 0.9 (mid-range) | 1.0 (strict) | < 0 or > 1 (invalid) |
| C3 prob value | 0.0 | 0.25 / 0.5 / 0.75 | 1.0 | — |

### `is_result_valid(has_mask, dice_finite, area_above_min, allow_empty)`

| Characteristic | Block T | Block F |
|----------------|---------|---------|
| C1 has_mask | True | False |
| C2 dice_finite | True | False |
| C3 area_above_min | True | False |
| C4 allow_empty | True | False |

### Flask API `POST /segment`

| Characteristic | Block A | Block B | Block C |
|----------------|---------|---------|---------|
| C1 image file | present and valid | absent | — |
| C2 method param | valid (`none`…`nlm`) | unknown string | empty string |
| C3 HTTP verb | POST | GET / PUT / DELETE | — |

### Experiment matrix (RQ1–RQ3)

| Characteristic | Blocks |
|----------------|--------|
| despeckle method | none / median / lee / frost / srad / nlm |
| lesion type | benign / malignant / normal |
| noise level (simulated) | low / medium / high |

---

## 3. Boundary Value Analysis

### Core function boundaries

| Module | Parameter | Min valid | Typical | Invalid values tested |
|--------|-----------|-----------|---------|----------------------|
| `despeckle` | `window_size` | 1 | 3, 5, 7 | 0, −1, −3, 2, 4, 6 |
| `metrics.dice/iou` | both-empty | — | partial overlap | both-empty → 1.0 (convention) |
| `metrics.hd95` | one-empty mask | — | both non-empty | one-empty → `sqrt(H²+W²)` diagonal |
| `metrics.hd95` | both-empty | — | — | both-empty → 0.0 |
| `segment.postprocess_mask` | `threshold` | 0.0 | 0.5 | −0.1, −1.0, 1.1, 2.0 |
| `stats.wilcoxon_compare` | sample size | 4 | 8 | 3 (too few), mismatched lengths |
| `stats.friedman_test` | group count | 3 | 5 | 1, 2 (too few) |

### Parametrized BVA — `postprocess_mask` (330 valid + 12 invalid = 342 tests)

Full Cartesian product tested in `TestPostprocessParametrize`:

| Axis | Values |
|------|--------|
| C1 shape | (4,4) (8,8) (16,16) (32,32) (8,16) (16,8) |
| C2 threshold | 0.0 0.1 0.2 0.3 0.4 **0.5** 0.6 0.7 0.8 0.9 1.0 |
| C3 prob value | **0.0** 0.25 **0.5** 0.75 **1.0** |
| Invalid threshold | −0.1 −1.0 1.1 2.0 (× 3 shapes = 12 tests) |

For each `(shape, threshold, prob_val)` the expected output is deterministic:
all-True if `prob_val >= threshold`, all-False otherwise.

### Parametrized BVA — `despeckle` window_size (30 invalid tests)

All five windowed methods (`median lee frost srad nlm`) × six invalid values
`{0, −1, −3, 2, 4, 6}` must raise `ValueError`. The `none` method is excluded
because it returns before window validation.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU unavailable in CI | High | High | Mock model (`torch.full_like(x, 0.8)`) injected via factory |
| BUSI dataset not present | High | Medium | `data_root=None` generates synthetic 32×32 arrays in-memory |
| Filter divergence on edge inputs | Medium | Medium | BVA at window 0/negative/even; regression pins output statistics |
| Statistical test numerical instability | Low | Medium | Delegate to scipy/statsmodels reference implementations |
| Mutation score below 80 % | Medium | High | MC-DC, BVA, and regression tests collectively drive score to 85.5 % |
| Coverage below 90 % | Low | High | Integration test + real-model path exercises full pipeline |

---

## 5. Test Case Design

### 5a. Original hand-written tests (Sections 2–4)

| Test case | File | Function tested | Course method (week) |
|-----------|------|-----------------|----------------------|
| `test_missing_root_raises_file_not_found` | `test_data` | `load_busi` | ISP (W6), Graph (W7) |
| `test_missing_label_dir_raises_file_not_found` | `test_data` | `load_busi` | ISP (W6), Graph (W7) |
| `test_size_mismatch_raises_value_error` | `test_data` | `load_busi` | ISP (W6) |
| `test_single_image_per_label` | `test_data` | `load_busi` | BVA (W6) |
| `test_valid_methods_return_same_shape[×6]` | `test_despeckle` | `despeckle` | ISP (W6), Graph (W7) |
| `test_none_method_returns_exact_copy` | `test_despeckle` | `despeckle` | Graph (W7) |
| `test_unknown_method_raises_value_error` | `test_despeckle` | `despeckle` | Graph (W7) |
| `test_window_size_0_raises` | `test_despeckle` | `_validate_window` | BVA (W6) |
| `test_window_size_negative_raises` | `test_despeckle` | `_validate_window` | BVA (W6) |
| `test_window_size_even_raises` | `test_despeckle` | `_validate_window` | BVA (W6) |
| `test_both_empty_is_one` | `test_metrics` | `dice`, `iou` | ISP (W6) |
| `test_perfect_overlap_is_one` | `test_metrics` | `dice`, `iou` | ISP (W6) |
| `test_shape_mismatch_raises` | `test_metrics` | `dice`, `iou`, `hd95` | ISP (W6) |
| `test_mcdc_has_mask_*` (11 cases) | `test_metrics` | `is_result_valid` | MC-DC / Logic Coverage (W8) |
| `test_2d_image_accepted` | `test_segment` | `predict` | ISP (W6) |
| `test_1d_image_raises` | `test_segment` | `predict` | ISP (W6) |
| `test_threshold_zero_all_true` | `test_segment` | `postprocess_mask` | BVA (W6) |
| `test_threshold_one_all_false` | `test_segment` | `postprocess_mask` | BVA (W6) |
| `test_threshold_negative_raises` | `test_segment` | `postprocess_mask` | BVA (W6) |
| `test_too_few_samples_raises` | `test_stats` | `wilcoxon_compare` | BVA (W6) |
| `test_length_mismatch_raises` | `test_stats` | `wilcoxon_compare` | ISP (W6) |
| `test_unsupported_method_raises` | `test_stats` | `correct_pvalues` | ISP (W6) |
| `test_valid_config_returns_results` | `test_experiment` | `run` | Graph (W7), ISP (W6) |
| `test_multiple_methods_all_appear` | `test_experiment` | `run` | Graph (W7) |
| `test_pipeline_step_error_propagates` | `test_experiment` | `run` | Graph (W7) |

### 5b. Flask API tests (28 tests)

| Test case | Status code | Course method |
|-----------|-------------|---------------|
| `test_get_returns_200` | 200 | ISP C1/C2, Graph edge GET / |
| `test_response_contains_all_methods` | 200 | ISP C3 (all 6 method options) |
| `test_all_despeckle_methods_return_200[×6]` | 200 | ISP C2, Graph dispatch edge |
| `test_missing_image_returns_400` | 400 | ISP C1 invalid, Graph edge validate_file |
| `test_unknown_method_returns_400` | 400 | ISP C2 invalid, Graph edge validate_method |
| `test_invalid_method_names_return_400[×5]` | 400 | BVA: empty, wrong case, trailing digit |
| `test_unknown_route_get_returns_404` | 404 | Graph edge unknown_route |
| `test_segment_get_returns_405` | 405 | Graph edge wrong_verb |
| `test_segment_put_returns_405` | 405 | Graph edge wrong_verb |

### 5c. Parametrized matrix tests (W6 ISP — 1118 tests)

| Class | File | Matrix | Count |
|-------|------|--------|-------|
| `TestDespeckleParametrize.test_output_shape_preserved` | `test_despeckle` | 6 methods × 4 kernels × 4 shapes × 3 dtypes | 288 |
| `TestDespeckleParametrize.test_output_dtype_preserved` | `test_despeckle` | same | 288 |
| `TestDespeckleParametrize.test_invalid_kernel_raises_value_error` | `test_despeckle` | 5 methods × 6 invalid kernels | 30 |
| `TestDiceParametrize.test_dice_self_identity` | `test_metrics` | 6 fills × 4 shapes | 24 |
| `TestDiceParametrize.test_dice_nonempty_vs_empty_is_zero` | `test_metrics` | 5 fills × 4 shapes | 20 |
| `TestDiceParametrize.test_dice_range` | `test_metrics` | 6 fills × 6 fills × 4 shapes | 144 |
| `TestIoUParametrize.test_iou_self_identity` | `test_metrics` | 6 fills × 4 shapes | 24 |
| `TestIoUParametrize.test_iou_nonempty_vs_empty_is_zero` | `test_metrics` | 5 fills × 4 shapes | 20 |
| `TestIoUParametrize.test_iou_symmetry` | `test_metrics` | 6 fills × 6 fills × 4 shapes | 144 |
| `TestHD95Parametrize.test_hd95_non_negative` | `test_metrics` | 4 fills × 4 fills × 4 shapes | 64 |
| `TestShapeMismatchParametrize.test_shape_mismatch_raises` | `test_metrics` | 3 functions × 6 shape-pairs | 18 |
| `TestPostprocessParametrize.test_constant_prob_correct_output` | `test_segment` | 6 shapes × 11 thresholds × 5 prob values | 330 |
| `TestPostprocessParametrize.test_invalid_threshold_raises` | `test_segment` | 3 shapes × 4 invalid thresholds | 12 |

### 5d. Regression tests (W14 — 9 tests)

| Test | File | What is pinned | Related issue |
|------|------|---------------|---------------|
| `TestDice.test_regression_known_dice` | `test_metrics` | `dice=0.5` on fixed 4×4 checkerboard (intersection=4/8) | Issue #2 |
| `TestIoU.test_regression_known_iou` | `test_metrics` | `iou=1/7` on fixed 4×4 2×2-square overlap | Issue #2 |
| `TestMetricsMutationKillers.test_niqe_regression_pinned_value` | `test_metrics` | `niqe=0.98895862` on seed-42 64×64 image | W10 + W14 |
| `TestDespeckleFilterRegressions.test_median_filter_pinned_mean` | `test_despeckle` | mean=0.509988, std=0.141972 on seed-99 8×8 | W10 + W14 |
| `TestDespeckleFilterRegressions.test_lee_filter_pinned_mean` | `test_despeckle` | mean=0.505515, [0,0]=0.521568 | W10 + W14 |
| `TestDespeckleFilterRegressions.test_frost_filter_pinned_corner` | `test_despeckle` | [0,0]=0.556666, [3,3]=0.444108 | W10 + W14 |
| `TestDespeckleFilterRegressions.test_srad_filter_pinned_mean` | `test_despeckle` | mean=0.504512, std=0.053290 | W10 + W14 |
| `TestDespeckleFilterRegressions.test_nlm_filter_pinned_mean` | `test_despeckle` | mean=0.500486, [3,3]=0.580652 | W10 + W14 |
| `test_regression_none_method_dice_range` | `test_experiment` | `dice` and `iou` in [0,1] on synthetic pipeline run | Issue #3 |

---

## 6. Coverage and Mutation Score

| Metric | Target | Actual |
|--------|--------|--------|
| Line coverage (`pytest --cov`) | ≥ 90 % | **97 %** (375 stmts, 12 missed) |
| Mutation score (`mutmut run`) | ≥ 80 % | **85.5 %** (300/351 killed) |
| Total tests | ≥ 1000 | **1585** |

### How to run

```bash
# All tests with coverage
pytest --cov=src --cov-report=term-missing

# Regression tests only
pytest -m regression

# Mutation testing (core modules, ~10 min)
mutmut run
mutmut results
mutmut show <id>      # inspect a specific surviving mutant
```

---

## 7. Graph Coverage (W7)

### 7a. `despeckle()` dispatch graph

```
[START]
  │
  ▼
validate_method ──── method not in _SUPPORTED_METHODS ──► ValueError
  │
  ▼
method == "none" ──► return image.copy() ──────────────► [END]
  │
  ▼
_validate_window ─── window_size ≤ 0 ──────────────────► ValueError
                  └── window_size even ─────────────────► ValueError
  │
  ├─ method == "median" ──► _median_filter(img, w)
  ├─ method == "lee"    ──► _lee_filter(img, w)
  ├─ method == "frost"  ──► _frost_filter(img, w)
  ├─ method == "srad"   ──► _srad_filter(img, n_iter)
  └─ method == "nlm"    ──► _nlm_filter(img, w)
                                  │
                                  ▼
                        result.astype(original_dtype) ──► [END]
```

**Edges covered by:** `TestDespeckleDispatch` (all 6 dispatch edges),
`TestDespeckleWindowBVA` (3 error edges), `TestDespackleMutationKillers`
(default-method edge), `TestDespeckleParametrize` (all 6 × 4 × 4 × 3 = 288
valid paths).

### 7b. `experiment.run()` pipeline graph

```
[START]
  │
  ▼
_validate_config ─── methods empty ────────────────────► ValueError
                 └── unknown method ───────────────────► ValueError
                 └── data_subset empty ────────────────► ValueError
  │
  ▼
_load_or_generate ── data_root is not None ──► load_busi() ──► filter by subset
                 └── data_root is None ──────► synthetic arrays
  │
  ▼
records empty? ──── True ──────────────────────────────► ValueError
  │ False
  ▼
for method in methods:
  for (image, mask, label) in records:
    ├─ despeckle_step ──────────────────────────────────► denoised
    ├─ segment_step ────────────────────────────────────► pred_mask
    ├─ metrics_step ────────────────────────────────────► dice, iou, hd95
    └─ any exception ──────────────────────────────────► RuntimeError (with context)
  │
  ▼
aggregate_results ──────────────────────────────────────► [END] return list
```

**Edges covered by:** `TestExperimentRunGraph` (happy-path, multi-method,
multi-label edges), `TestExperimentRunErrors` (all three ValueError edges,
RuntimeError propagation edge), `test_run_with_real_busi_path` (real
`load_busi` path).

### 7c. Flask `create_app()` route graph

```
[START]
  │
  ▼
route_dispatch
  ├─ GET  /          ──────────────────────────────────► html_response (200)
  │
  ├─ POST /segment
  │     │
  │     ├─ no image file ──────────────────────────────► JSON 400
  │     ├─ unknown method ─────────────────────────────► JSON 400
  │     └─ valid ──► despeckle ──► predict ──► metrics ► JSON 200
  │
  ├─ unknown route ────────────────────────────────────► JSON 404
  └─ wrong HTTP verb ──────────────────────────────────► JSON 405
```

**Edges covered by:** `TestIndexRoute` (GET / edge), `TestSegmentHappyPath`
(valid POST edge, all 6 dispatch sub-edges via parametrize),
`TestSegmentErrors` (no-file 400, bad-method 400), `TestErrorHandlers`
(404, 405 × 3 verbs).

### 7d. `data.load_busi()` graph

```
[START]
  │
  ▼
check_root ──── root does not exist ──────────────────► FileNotFoundError
  │
  ▼
for label in ("benign", "malignant", "normal"):
  ├─ label_dir missing ────────────────────────────────► FileNotFoundError
  └─ iterate image files:
       ├─ no matching mask ──────────────────────────► continue
       └─ size mismatch ────────────────────────────► ValueError
  │
  ▼
append (image, mask, label) ────────────────────────────► [END] return records
```

**Edges covered by:** `TestLoadBusiMissingRoot`, `TestLoadBusiMissingSubfolder`,
`TestLoadBusiSizeMismatch`, `TestLoadBusiHappyPath`, `TestDataMutationKillers`
(continue-vs-break edge via 3-images-per-label test).

---

## 8. Flask API Endpoint Tests (W6 + W7)

| Route | Status codes tested | Testing method |
|-------|--------------------|-|
| `GET /` | 200 | ISP C1/C2, Graph edge GET / |
| `POST /segment` | 200, 400 (no file), 400 (bad method) | ISP C1–C3, BVA |
| Unknown route | 404 | Graph: unknown_route edge |
| Wrong verb on `/segment` | 405 | Graph: wrong_verb edge (GET, PUT, DELETE) |

Parametrized over all 6 despeckle methods (`@pytest.mark.parametrize`).
Mock model injected via `create_app(model=...)` — no GPU required.

---

## 9. Parametrized Test Matrix (W6 ISP)

| File | Characteristics | Combinations |
|------|----------------|-------------|
| `test_despeckle.py` | C1 (6 methods) × C2 (4 kernels) × C3 (4 shapes) × C4 (3 dtypes) | 288 valid shape + 288 valid dtype + 30 BVA invalid = **606** |
| `test_metrics.py` | dice/iou self-identity, range, symmetry; hd95; shape mismatch | **458** |
| `test_segment.py` | C1 (6 shapes) × C2 (11 thresholds) × C3 (5 prob values) + BVA invalid | 330 + 12 = **342** |

Each combination maps to a specific ISP block or BVA boundary — no padding.

---

## 10. Mutation Testing Evidence (W10 Syntax-Based Testing)

### Setup
- Tool: `mutmut 2.4.4`
- Scope: `src/metrics.py`, `src/despeckle.py`, `src/data.py`, `src/stats.py`
- Runner: `pytest tests/test_metrics.py tests/test_despeckle.py tests/test_data.py tests/test_stats.py -x -q --no-cov --timeout=30`

### Results: 300 killed / 351 total = **85.5 %**

### Survived → Killer test → Killed

| Mutant group | Example mutation | Why it survived initially | Killer test added |
|---|---|---|---|
| Error message strings (`M292`, `M297`, `M320`, `M323`, `M333`) | `"XX…XX"` prefix/suffix in error text | `pytest.raises(ValueError)` only checks exception type | Added `match=` parameter checking key substrings (`"root"`, `"label"`, `"length"`, `"3"`) |
| `hd95` one-empty branch (`M34–M37`) | `or` → `and`; `== 0` → `== 1` | No test with exactly one empty mask + pinned return value | `test_hd95_empty_pred_nonempty_target_is_diagonal` pins `sqrt(H²+W²)` |
| `hd95` diagonal formula (`M38–44`) | `shape[0]²` → `shape[1]²` | Only square shapes used in value tests | Used non-square `(6, 10)` shape to expose H≠W confusion |
| `psnr` / `ssim` data_range (`M51–63`) | `max−min` → `max+min` | Tests only checked return-is-float | `test_psnr_uses_max_minus_min` pins exact dB with known MSE; `test_psnr_zero_data_range_fallback_is_one` pins fallback |
| `niqe` formula (`M68–84`) | `== 0` → `!= 0`; `÷` → `×` | Only `>= 0` asserted | `test_niqe_nonconstant_is_positive` + regression pin `0.98895862` |
| Filter arithmetic (`M135–145`) | `* 255` → `/ 255`; `<= 1.0` → `< 1.0` | Shape/dtype checks ignored pixel values | `TestDespeckleFilterRegressions`: mean/std/corner values pinned per filter |
| `srad` param key (`M129`) | `"n_iter"` → `"XXn_iterXX"` | Default value masked the missing param | `test_srad_n_iter_1_differs_from_10`: `n_iter=1` ≠ `n_iter=10` |
| Default method (`M105`) | `"none"` → `"XXnoneXX"` | All tests passed explicit `method=` | `test_default_method_is_none`: calls `despeckle(img)` with no method arg |

### Remaining survivors (51 / 351 = 14.5 %)

| Category | Count | Reason |
|----------|-------|--------|
| **Equivalent mutants** | 8 | Epsilon changes (`1e-10` → `2e-10`) produce numerically identical results within float64 precision — no observable difference exists |
| **Deep filter arithmetic** | 24 | Internal operations of `_frost_filter`, `_srad_filter`, `_nlm_filter` where mutated intermediate values converge to similar statistics for random inputs; per-pixel regression across all pixels would be brittle |
| **XX-prefix error messages** | 10 | `re.search` finds the matched substring anywhere in `"XXoriginal_textXX"` — anchored regex (`\A`) or full-message equality checks would kill these but couple tests to exact wording |
| **Data loading edge cases** | 9 | `continue`→`break` and shape-slice mutations that require a dataset with partially-missing masks mid-directory to distinguish |
