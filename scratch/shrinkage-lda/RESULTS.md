# Shrinkage-LDA preview — results and read (2026-09-17)

Two datasets, same outer 5×5 folds as the shipped elastic net (`classify`) and linear SVM
(`classify_svm`) at seed 0 (fold identity asserted), LDA in the dual form with in-fold
standardization and **no inner CV** (Ledoit-Wolf shrinkage is analytic). Full numbers in
`out/<dataset>/summary.md`; per-fold AUCs in `out/<dataset>/paired_folds.csv`.

## Headline numbers

| | 5xFAD (n = 52, p = 8,828) | MagNet-EV-ADD (n = 40, p = 2,343) |
|---|---|---|
| elastic net, seed 0 | 0.905 ± 0.139 (90 s) | 0.870 (defaults) |
| linear SVM, seed 0 | 0.796 ± 0.174 (7 s) **TuningNoiseWarning: 11/25 folds sub-plateau** | 0.963 ± 0.067 (2 s) |
| **shrinkage LDA**, seed 0 | **0.848 ± 0.140** (0.1 s for 25 folds) | **0.950 ± 0.108** (0.03 s) |
| LDA vs EN, paired W/L/T | 5 / 13 / 7, Δ −0.057 | 10 / 0 / 15, Δ +0.080 |
| LDA vs SVM, paired W/L/T | 15 / 4 / 6, Δ +0.052 | 1 / 3 / 21, Δ −0.013 |
| LDA − SVM across seeds 0/1/2 | +0.052 / +0.062 / +0.049 (sign holds) | −0.013 / 0.000 / +0.020 (tie) |
| LDA label-shuffle null (1000 perms) | null 0.50 ± 0.10, max 0.81, **p = 0.001**, 90 s | null 0.49 ± 0.12, max 0.88, **p = 0.001**, 32 s |
| LDA top-20 fold-membership (mean; ≥0.8) | **0.83; 13 of 20** | 0.45; 2 of 20 |
| LDA top-20 overlap with EN / SVM top-20 | 17 / 14 (EN∩SVM 12) | 8 / 11 (EN∩SVM 9) |
| multiclass-native (macro OvR AUC) | Genotype 3-class: 0.725 ± 0.130 | Condition 4-class: **0.957 ± 0.027** |
| Ledoit-Wolf shrinkage (per class) | 0.47 / 0.24 | 0.25 / 0.50 |
| synthetic 100 × 20,000 fit | 0.04 s | 0.04 s |

The elastic-net and SVM numbers reproduce the documented previews exactly (0.905 / 0.796 on
5xFAD; 0.870 / 0.963 on MagNet), confirming the LDA ran on the same folds.

## What the plan test says

1. **The dual form works and is exact.** Eight tests pin it to sklearn at ~1e-10 on
   coefficients, intercepts, scores and probabilities, binary and multiclass, including the
   Ledoit-Wolf intensity computed from the Gram matrix. A 20,000-feature fit is 40 ms;
   sklearn could not finish a 6,186-feature fit in ten minutes. The compute objection is gone.

2. **Performance is competitive, never the clear loser, and it costs nothing.** On MagNet the
   LDA ties the SVM (never worse on more than 3 of 25 folds at any seed) and beats the
   elastic net on 10 folds with 0 losses. On 5xFAD it sits between the two: below the elastic
   net (5/13/7) and above the SVM on every seed (15/4/6, 12/3/10, 15/1/9, Δ ≈ +0.05 each time).
   **Caveat on the 5xFAD SVM comparison:** the SVM fired `TuningNoiseWarning` (11 of 25 folds
   tuned below the plateau), so by the rule shipped in 0.5.3 that comparison is not valid until
   the SVM's `c_grid` is narrowed and re-run — the documented 0.796 is itself depressed by
   tuning noise. The LDA has no such failure mode: there is nothing to tune.

3. **The corrected resampled t is very conservative at J = 25.** MagNet's 10/0/15 vs the
   elastic net gives a corrected-t p of 0.34 (Wilcoxon 0.005); 5xFAD's 15/4/6 vs the SVM gives
   0.34 / 0.25 / 0.10 across seeds. The inflation factor is √((1/25 + 8/32)/(1/25)) ≈ 2.7 on
   the SE. This is the known behaviour of the Nadeau–Bengio correction and is exactly why the
   0.5.3 rule makes **sign consistency across seeds** the decision and the p secondary: the
   seeds say LDA > SVM on 5xFAD (three of three) and LDA ≈ SVM on MagNet.

4. **The null is tractable for the first time.** 1000 permutations × 15 folds in 32–90 s,
   against the elastic net's ~15,000-fit null (≈3.5× its run) and the SVM's ≈9×. The LDA
   could ship with `run_null=True` as its **default** and still return in under two minutes
   at 20,000 features — the two-pass "results first, null second" protocol is unnecessary here.

5. **The feature reading differs by dataset, and that is informative.** On 5xFAD the LDA's
   top-20 is the canonical AD set (APP + the human transgene, APOE, midkine, TICN1/2,
   clusterin, GPC1, the C1q trio, SPON1) with 0.83 mean fold-membership and 17/20 overlap with
   the elastic net — the covariance-adjusted direction lands on the same biology and is
   *more* stable than either shipped model's read. On MagNet the top-20 is unstable
   (0.45 mean membership, 2 of 20 ≥ 0.8) and overlaps less — at n = 40 with strong
   co-variation the discriminant direction is spread thin; the *performance* is stable, the
   *weights* are not. A shipped template must surface this (membership frequency, as the SVM
   does) so a dense weight list is never read as a stable feature set.

6. **Multiclass works natively.** MagNet's four conditions (10 each) give macro OvR AUC 0.957
   and balanced accuracy 0.79 against chance 0.25 — the first multiclass predictive result
   the engine can produce. 5xFAD's three-level Genotype is weak (0.725), as expected: two of
   the three levels (C57BL/6j, WT) are both non-transgenic and differ only by strain.

## Recommendation

Build it as a shipped template, in the dual form, with these design choices carried
from the preview:

- **`run_null=True` by default** (it is cheap) — the only classifier template whose first
  pass includes the null. The two-pass protocol stays as written for the other three.
- **Multiclass-native from the start** (`outcome=` categorical, `binarize=` optional), since
  that is the gap nothing else fills; binary is the two-class special case of the same code.
- **Dense-weight stability read = top-k membership frequency** (the SVM precedent), reported
  beside the per-class Ledoit-Wolf intensities (a diagnostic: λ → 1 means the covariance
  carried nothing and the direction is a standardized mean difference).
- **Fold identity recorded** like the other three, so it joins the paired comparison, and the
  0.5.3 selection rule extends to three linear models — with the same "sign across seeds"
  decision and the same pre-stated reading (sparse minimal set / dense module weights /
  covariance-adjusted DE-like direction).
- Exactness test against sklearn at small p as the oracle (the one place the engine ships
  custom numerics rather than a wrapper — justified by an identity that is testable to
  machine precision), plus the 5xFAD / MagNet smokes pinning the numbers above.

Not changed by this preview: the elastic net stays the default first model (it won on 5xFAD
and its sparse reading is what most studies ask for first). The LDA is the tuning-free,
multiclass-capable, null-included alternative, offered on the same triggers as the SVM plus
"the design has more than two groups".
