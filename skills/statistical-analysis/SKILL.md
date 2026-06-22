---
name: statistical-analysis
description: >-
  How to run a statistically sound analysis in a Findings Workflow project:
  seed from the vetted lib/ templates, and apply the leakage-safe, multiplicity-honest
  patterns the conventions require. Use when performing differential abundance,
  group comparisons, classification, regression, or dimensionality reduction.
---

# Statistical analysis procedure

Authoritative rules: `conventions/statistics.md`. This skill is the *how*; that doc is the *what*.

## Always seed from a `lib/` template (or reuse the project's existing script)

The `lib/` templates (`${CLAUDE_PLUGIN_ROOT}/lib/`) implement these analyses with the assumptions and missingness handling already right. **Copy the relevant template into `scripts/scratch/` and adapt it, rather than generating fresh statistics code** — and if the project already has a script for this analysis, reuse/extend that one (one script per task). Import the project's shared modules rather than re-implementing. Record the template lineage in `provenance.seeded_from`. If no template fits, write the analysis to `conventions/coding.md` standards and flag that a new `lib/` template may be worth contributing (templates are reviewed before they ship). Adapt parameters and specifics — but keep the **dangerous structure** (e.g. preprocessing inside CV folds, the label-shuffle null) intact; the stats-reviewer checks this.

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
