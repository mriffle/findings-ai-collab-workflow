# Convention — Statistics

*Spec source: doc 05.3. These rules are checked by the **stats-reviewer** agent against every analysis (statistical correctness is judgment-heavy and not cleanly hook-enforceable). The vetted `lib/` template scripts implement these defaults so agents inherit them by seeding from a template rather than writing fresh statistics code: the normalization (scale-tag guard) and batch-correction (log-scale / batch-label-only / confounding) templates ship and carry their guards. The **moderated differential-abundance, leakage-safe classifier, and regression templates are still phase-E work** — for those the stats-reviewer is the sole active enforcer until they ship.*

## Reporting

- **No bare p-values.** Every significance claim reports, together: an **effect size**, a **confidence interval** for that effect, and a **corrected** p-value. A p-value alone is not a result. (Mirrored in the finding `evidence` schema, conventions/findings.md §2.3.)
- **Always apply and name a multiple-testing correction.** Default **Benjamini–Hochberg (BH/FDR)** for omics-scale testing; name the method and the family it was applied over. Report q-values / adjusted p-values.
- **Report all tests run**, not only the significant ones. This feeds the exploration log (doc 03.6) and keeps the false-discovery burden visible — silently dropping the tests that didn't pan out is how multiplicity gets hidden.

## Descriptive characterization (the cohort)

Before any modelling, the metadata is characterized in full (Stage 1) — to produce figures and tables the scientist can publish *and* to expose the biases that constrain every downstream claim. Done thoroughly this means:

- **Every variable's distribution** — sample counts per categorical level; summaries/histograms for continuous variables (age, BMI, run order).
- **Pairwise structure that matters** — cross-tabulate the variable of interest against each covariate and against batch/run-order (sex × group, age × sex, group × batch). A publication-ready **cohort table ("Table 1")** breaks the design factors down by the primary grouping.
- **Imbalance and confounding, quantified** — group-size ratios for imbalance; **bias-corrected Cramér's V** (categorical↔categorical) or the appropriate association measure (continuous) for confounding between the contrast and each covariate/batch.

**Record the material gotchas as caveat findings** (`kind: caveat`, conventions/findings.md §2.6): a finding is warranted when an imbalance, skew, or confound would change how a downstream result is analyzed or interpreted. The thresholds are **judgment, not hard rules** — as rough orientation, treat a group-size ratio beyond ~3:1 (or any arm under ~10 samples), or a contrast↔covariate Cramér's V above ~0.3, as worth a careful look and likely a caveat. A recorded confound or imbalance has teeth downstream: a confounded covariate should **enter the model as a covariate** (or the comparison be stratified) rather than be silently ignored, and a class imbalance dictates **balanced metrics and stratified folds** in any classifier (see ML below). Carry the caveat into the affected finding's `caveats` via a `relates_to` edge.

## Test choice

- **Canonical over esoteric.** Prefer widely used, explainable tests; a reviewer and a reader should recognize the method.
- **Moderated models for differential abundance.** Prefer a **moderated linear model (limma / MSstats-style)** over naive per-feature t-tests — borrowing variance across features is both more powerful and more honest at omics scale. Nonparametric tests (Mann–Whitney) where distributional assumptions fail.

## Normalization & batch correction

- **Normalization is a recorded scientific choice, not a silent default.** Recommend **median** to the scientist (simple, interpretable, linear in → linear out) and confirm it; offer **MAD** (robust per-sample z-score) and **VSN** (variance-stabilizing) as alternatives. Record the chosen method in the finding's `provenance.params`. Respect scale — MAD and VSN already log-transform internally, so never log again; for data that arrives already-log, use the log-domain `median_center` rather than re-normalizing from linear. The `lib/` `normalize` template carries a `scale` tag (`linear`/`log2`/`log10`/`ln`/`glog2`/`zscore`/`ratio`) and refuses the double-log at runtime.
- **Batch correction gets the batch label only.** When you correct for batch (ComBat, via the `batch-correct-combat` template), pass it the **batch variable and nothing else** — never the biological covariate of interest. Under confounding, handing the method the covariate to "protect" lets it attribute the confounded variance to biology and launder a batch artifact into the very signal you then test. Batch-only correction is the conservative choice: it cannot manufacture signal.
- **Prefer modeling batch over removing it — for significance testing.** ComBat removes between-batch variance *and* shrinks within-feature variance; feeding corrected values into a naive per-feature test understates standard errors and **inflates significance** (Nygaard et al. 2016). So for differential testing, prefer including **batch as a covariate in the design matrix** (limma / MSstats-style) over pre-correcting. Reserve ComBat *output* for visualization/clustering (PCA, heatmaps) and the both-ways robustness check below — not as direct input to a per-feature significance test.
- **Report corrected AND uncorrected as a robustness check.** Keep the uncorrected data and run the key analysis both ways. *Signal survives batch-only correction* → evidence it is robust to batch; *signal disappears* → confounded with batch, not cleanly attributable to biology. Two caveats keep this honest: under **partial** confounding, real biology is *attenuated* too, so a weakened-but-present effect is expected for genuine signal and is not itself an artifact; and because of the variance deflation above, treat a *more*-significant corrected result with suspicion rather than as strengthened. Carry the conclusion into the finding's caveats.
- **Surface the confound and get sign-off before correcting.** Quantify batch↔covariate confounding before any correction (`assess_batch_confounding`, bias-corrected Cramér's V). It warns when a covariate of interest is **perfectly confounded** with batch (constant within every batch → correction deletes its effect) *or* **strongly** confounded (above a threshold → correction attenuates it proportionally). Get explicit scientist sign-off on a flagged confound; do not silently proceed.

## Machine learning / prediction

- **No data leakage.** Every preprocessing step that *learns from data* — normalization, imputation, feature selection, scaling, batch correction — happens **inside** each cross-validation fold, fit on the training partition only, never on the full dataset first. Use scikit-learn `Pipeline` so the fit/transform boundary is structural, not manual. (For predictive models this *overrides* the global-normalization default above: fit the normalizer/ComBat on the training fold only.)
- **Match cross-validation to the generalization target.** Define the unit of generalization (new samples? new patients? new batches?) and structure folds accordingly — **group/subject-wise folds** (`GroupKFold`) when samples cluster within a patient/subject, so the same subject never appears in both train and test.
- **Mandate a label-shuffle null** for any classifier: permute the labels and re-run the full pipeline; if performance does not collapse to chance, there is leakage. Report the null distribution alongside the real score.

## Power & honesty

- **Be honest about power.** Treat small-n results as **`exploratory`** by default (conventions/findings.md §5). Don't dress an underpowered observation as a confirmatory result.
- **Confounds are first-class.** A confounded comparison (variable of interest aliased with batch/run-order) limits what any statistic can claim; surface it (detected in Stage 1, quantified by `assess_batch_confounding` before any batch correction) and carry it into the finding's caveats.

## Enforcement

| Rule | Enforced by |
|---|---|
| No bare p; effect+CI; named correction present | **Stats-reviewer** |
| No leakage; CV matched to target; label-shuffle null | **Stats-reviewer** |
| Canonical tests; moderated models for DE | **Stats-reviewer** |
| Normalization method recorded; scale respected (no double-log) | **Stats-reviewer** + the `normalize` template's runtime scale guard (holds only if the project copy keeps it) |
| Batch correction is batch-label-only; corrected **and** uncorrected both reported; prefer batch-as-covariate for testing | **Stats-reviewer** |
| Perfect/strong batch↔covariate confounding surfaced + signed off before correcting | **Stats-reviewer** + orchestrator behavior + human sign-off |
| All tests run are reported (→ exploration log) | **Stats-reviewer** + orchestrator behavior |
| Small-n / confounded → exploratory | **Stats-reviewer** + findings-manager (phase) |
| Cohort characterized; material imbalance/confound recorded as a caveat finding; confounded contrast modelled (covariate/stratify) not ignored | **Stats-reviewer** + findings-manager + human sign-off (Stage 1/3) |
