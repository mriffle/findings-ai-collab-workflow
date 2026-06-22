# Convention — Statistics

*Spec source: doc 05.3. These rules are checked by the **stats-reviewer** agent against every analysis (statistical correctness is judgment-heavy and not cleanly hook-enforceable). The vetted `lib/` template scripts implement these defaults, so agents inherit them by seeding from a template rather than writing fresh statistics code.*

## Reporting

- **No bare p-values.** Every significance claim reports, together: an **effect size**, a **confidence interval** for that effect, and a **corrected** p-value. A p-value alone is not a result. (Mirrored in the finding `evidence` schema, conventions/findings.md §2.3.)
- **Always apply and name a multiple-testing correction.** Default **Benjamini–Hochberg (BH/FDR)** for omics-scale testing; name the method and the family it was applied over. Report q-values / adjusted p-values.
- **Report all tests run**, not only the significant ones. This feeds the exploration log (doc 03.6) and keeps the false-discovery burden visible — silently dropping the tests that didn't pan out is how multiplicity gets hidden.

## Test choice

- **Canonical over esoteric.** Prefer widely used, explainable tests; a reviewer and a reader should recognize the method.
- **Moderated models for differential abundance.** Prefer a **moderated linear model (limma / MSstats-style)** over naive per-feature t-tests — borrowing variance across features is both more powerful and more honest at omics scale. Nonparametric tests (Mann–Whitney) where distributional assumptions fail.

## Machine learning / prediction

- **No data leakage.** Every preprocessing step that *learns from data* — normalization, imputation, feature selection, scaling, batch correction — happens **inside** each cross-validation fold, fit on the training partition only, never on the full dataset first. Use scikit-learn `Pipeline` so the fit/transform boundary is structural, not manual.
- **Match cross-validation to the generalization target.** Define the unit of generalization (new samples? new patients? new batches?) and structure folds accordingly — **group/subject-wise folds** (`GroupKFold`) when samples cluster within a patient/subject, so the same subject never appears in both train and test.
- **Mandate a label-shuffle null** for any classifier: permute the labels and re-run the full pipeline; if performance does not collapse to chance, there is leakage. Report the null distribution alongside the real score.

## Power & honesty

- **Be honest about power.** Treat small-n results as **`exploratory`** by default (conventions/findings.md §5). Don't dress an underpowered observation as a confirmatory result.
- **Confounds are first-class.** A confounded comparison (variable of interest aliased with batch/run-order) limits what any statistic can claim; surface it (detected in Stage 1) and carry it into the finding's caveats.

## Enforcement

| Rule | Enforced by |
|---|---|
| No bare p; effect+CI; named correction present | **Stats-reviewer** |
| No leakage; CV matched to target; label-shuffle null | **Stats-reviewer** |
| Canonical tests; moderated models for DE | **Stats-reviewer** |
| All tests run are reported (→ exploration log) | **Stats-reviewer** + orchestrator behavior |
| Small-n / confounded → exploratory | **Stats-reviewer** + findings-manager (phase) |
