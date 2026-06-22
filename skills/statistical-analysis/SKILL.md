---
name: statistical-analysis
description: >-
  How to run a statistically sound analysis in a Findings Workflow project:
  prefer the vetted lib/ functions, and apply the leakage-safe, multiplicity-honest
  patterns the conventions require. Use when performing differential abundance,
  group comparisons, classification, regression, or dimensionality reduction.
---

# Statistical analysis procedure

Authoritative rules: `conventions/statistics.md`. This skill is the *how*; that doc is the *what*.

## Always prefer `lib/`

The vetted library (`${CLAUDE_PLUGIN_ROOT}/lib/`, version-recorded) implements these analyses with the assumptions and missingness handling already right. **Call it rather than generating fresh statistics code.** Record the `lib/` version in the finding's provenance. If `lib/` lacks what you need, write the analysis to `conventions/coding.md` standards and flag that a `lib/` addition may be warranted (a wrong default in `lib/` is wrong everywhere, so additions are reviewed).

## Reporting (every significance claim)

Report, together — never a bare p-value:

- **effect size** (e.g. log2 fold change, mean difference, AUC, r);
- a **confidence interval** for that effect;
- a **corrected** p-value with the correction **named** (default **Benjamini–Hochberg / FDR**, applied over the correct family).

Report **all tests run**, not only significant ones — append the discarded threads to `findings/exploration-log.md`.

## Differential abundance

Prefer a **moderated linear model** (limma / MSstats-style) over naive per-feature t-tests — variance shrinkage across features is more powerful and more honest at omics scale. Respect the data's transformation state (log vs linear) and missing-value semantics from `state/DATA_DESCRIPTION.md`.

## Prediction / classification — leakage-safe by construction

- Build a scikit-learn **`Pipeline`** so every learned preprocessing step (scaling, imputation, feature selection, normalization, batch correction) is **fit inside the training fold only**. Never fit on the full dataset before splitting.
- **Match CV to the generalization target.** Define the unit (new samples? patients? batches?) and use **`GroupKFold`** / grouped splits when samples cluster within a subject, so a subject never spans train and test.
- **Always run a label-shuffle null:** permute labels, re-run the full pipeline, confirm performance collapses to chance. Report the null distribution beside the real score. If it doesn't collapse, there is leakage — find it before reporting anything.
- Set and record `random_state` everywhere.

## Power & honesty

Treat small-n results as `exploratory`. State confounds (from Stage 1) and limit claims accordingly. Underpowered ≠ negative.

## Handoff

Produce a parameterized script + results, then route the analysis to the **stats-reviewer**. A finding built on the analysis is promoted only after that review passes.
