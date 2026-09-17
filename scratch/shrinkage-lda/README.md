# Shrinkage-LDA preview (scratch — not a shipped template)

A test of the plan discussed on 2026-09-17: a third linear classifier, Ledoit-Wolf
**shrinkage LDA**, implemented in the **dual form** so it scales to the user's routine
width (20,000+ proteins), run on the two local classification datasets on the **same
outer folds** as the shipped elastic net and linear SVM.

Nothing under `lib/`, `agents/`, `commands/`, `conventions/`, or `templates/` is touched.
The shipped templates are imported read-only. The real data lives in the git-ignored
`testdata/`; only this directory is tracked (on the `shrinkage-lda-preview` branch).

| File | What |
|---|---|
| `dual_lda.py` | `DualShrinkageLDA` — reproduces `sklearn LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` exactly, via Woodbury on the n×n Gram matrix; never forms the p×p covariance. |
| `test_dual_vs_sklearn.py` | Pins it to sklearn at machine precision (shrinkage intensity, coefficients, intercepts, decision scores, probabilities; binary and multiclass) and checks a 100×20,000 fit is fast and finite. |
| `run_preview.py` | The two-dataset test: paired per-fold comparison vs the templates (corrected resampled t + indicative Wilcoxon), seed noise, a 1000-permutation null, a top-20 stability read + overlap, a multiclass-native run, timings. |
| `out/<dataset>/summary.md` | Results (+ `paired_folds.csv`, `top20.csv`, `paired_stats.json`). |
| `RESULTS.md` | The read of the results and the recommendation. |

Run:

```
PYTHONPATH=lib:scratch/shrinkage-lda ./.venv/bin/python -m pytest scratch/shrinkage-lda -q
PYTHONPATH=lib:scratch/shrinkage-lda ./.venv/bin/python scratch/shrinkage-lda/run_preview.py --dataset magnet
PYTHONPATH=lib:scratch/shrinkage-lda ./.venv/bin/python scratch/shrinkage-lda/run_preview.py --dataset 5xfad
```
