# Does Despeckling Help? A Task-Linked Benchmark of Speckle Reduction for Breast-Ultrasound Lesion Segmentation

**Solo project** — all code and commits are by a single contributor.  
Course: Software Testing (115 Spring), NYCU.

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Does despeckling improve U-Net segmentation (Dice/IoU) vs. no denoising? |
| RQ2 | Does the effect depend on lesion type (benign/malignant) or lesion size? |
| RQ3 | Can reference-free metrics (e.g. NIQE) predict when despeckling helps the task? |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run only regression tests
pytest -m regression

# Run the web demo
python3 main.py

# Mutation testing (target: >= 80%)
mutmut run
mutmut results
```

---

## Quality Gates

| Metric | Target | Current |
|--------|--------|---------|
| Line coverage (`pytest --cov`) | ≥ 90 % | **99 %** (419/421 stmts; only `app.run()` in `__main__` guard uncovered) |
| Mutation score (`mutmut run`) | ≥ 80 % | **95.2 %** (687/722 killed; 96.6 % excl. 11 documented equivalents) |
| Total tests | ≥ 1000 | **1683** |

---

## Project Layout

```
src/            Source modules (data, despeckle, segment, metrics, stats, experiment)
tests/          pytest test suite (one file per module + integration)
docs/           TEST_STRATEGY.md
.github/        CI workflow + issue/PR templates
```

---

## Stage 2 TODO (real experiments — future work)

- [ ] Train a full U-Net on the BUSI dataset (split: 70/15/15 train/val/test)
- [ ] Swap mock `segment.predict` for the trained model checkpoint
- [ ] Run full RQ1 experiment: compare Dice/IoU for all despeckle methods vs. none
- [ ] Run RQ2 subgroup analysis by lesion type (benign/malignant) and lesion area quartile
- [ ] Run RQ3 metric-correlation study: correlate NIQE/PSNR/SSIM delta with Dice delta
- [ ] Add a second ultrasound dataset (e.g. STU or UDIAT) for cross-device generalisation
- [ ] Produce final results tables and figures for the paper
- [ ] Archive trained model weights and experiment configs for reproducibility
