---
name: stage2-data
description: "Stage 2 — Understand the data. Determine orientation, transformation/normalization state, and missing-value semantics of the data matrix; characterize missingness, contaminants, decoys, and duplicates; write state/DATA_DESCRIPTION.md."
---

# Stage 2 — Understand the data

**Precondition:** `state/workflow.json` shows `metadata_done: true`. If not, run `stage1-metadata` first and say so.

The goal is a *verified* understanding of the sample-by-feature matrix itself — its structure and its semantics — before any loader is trusted or any analysis is run.

## Do this

For the data matrix:

- **Orientation, shape, identifier formats** — determine (do not assume) whether rows are samples or features; get the shape; characterize the feature-id and sample-id schemes.
- **Examine feature names, sample names, and values.**
- **Reconcile control samples against the data matrix.** The experimental/control classification was settled in Stage 1 from the *metadata*; confirm it holds in the *data*. Scan the sample columns/rows for the control-naming conventions surfaced in Stage 1 (Pool/QC/Ref/Bridge/Std/Blank) and reconcile **both ways** — a control present in the data but absent from the metadata (or vice versa) is a discrepancy the loader's exact-pairing check will fail loud on in Stage 3; surface it now so the classification entering Stage 3 is complete and the experimental subset is unambiguous.
- **Transformation state** — log vs linear (don't guess; check distributions and ranges). This is the `scale` the loader will record on its `Dataset` (`linear`/`log2`/`glog2`/`zscore`); the downstream normalization and batch-correction templates read it to refuse scale-incorrect steps (e.g. double-logging, ComBat on linear data), so getting it right here is load-bearing.
- **Normalization state** — whether and how the data has *already* been normalized (distinct from the normalization *you* will apply for analysis — see the decisions below).
- **Missing-value encoding** — how missingness is represented, and whether tokens are interchangeable. **They are not:** `0`, `NA`, `NaN`, empty string, and tool tokens like `"Filtered"` mean different things. Conflating a true zero with missing-not-at-random changes every downstream statistic. Identify and decide deliberately.
- **Characterize missingness structure, contaminants, decoys, and duplicates.**
- **Run tests against your current understanding and surface problems.** Assumptions are hypotheses (doc 05).

### Preprocessing decisions to confirm with the scientist

These are *analysis* choices (not facts about the raw file), but they depend on what you find above, so surface them now and record the decision in `state/DATA_DESCRIPTION.md`. They are applied later (QC in Stage 3, analysis in Stage 4), seeding from the `lib/` `normalize` and `batch-correct-combat` templates:

- **Normalization method.** Recommend **median** (simple, interpretable, linear in → linear out) and confirm with the scientist; offer **MAD** (robust per-sample z-score) and **VSN** (variance-stabilizing) as alternatives. It is a recorded scientific choice, not a silent default. Respect scale — MAD and VSN already log internally, so don't log again (`conventions/statistics.md`).
- **Batch axis and confounding.** If the design has a batch axis (e.g. cohort/run/plate, identified in Stage 1), name it and surface any **batch↔biology confounding** (carried from Stage 1; quantified later by `assess_batch_confounding`). A variable of interest aliased with batch limits what any statistic can claim. Batch correction, when done, uses the **batch label only** — never the covariate of interest (it would launder a confounded artifact into the signal under test).
- **Missing-value handling.** If the matrix has missing values (NaN, distinct from a real `0` — settled in the encoding step above), the normalization and batch-correction templates require **complete data** and will fail loud on NaN. So the handling is a scientist decision to make here, not in the templates: present the options — **leave as `0`** (only if missing genuinely means not-detected/zero), **impute** (per-sample mean/median, or **KNN**), or **drop** features/samples above a missingness threshold — with their trade-offs, and record the choice. The workflow does not bake in an imputation method; it is study-specific and lives in the project copy. **Make this evidence-based, not blind:** the choice is *surfaced* here, but the evidence only exists once the matrix is loaded, so confirm/revisit it against the **Stage-3 `missingness` diagnostic** (`lib/figures/missingness`, in the QC report) — its **MNAR panel** (detection rate vs abundance) shows whether missingness is left-censored (low-abundance features dropping out ⇒ a left-censored imputation like MinProb/QRILC, *not* mean/median) or closer to random, and its **completeness curve** sets any drop threshold.

### Watch for the domain fidelity traps (doc 05.4)

- **Spreadsheet identifier corruption** — gene symbols turned into dates (SEPT/MARCH/DEC), accessions in scientific notation, stripped leading zeros. Assume it happened until proven otherwise.
- **Scale confusion** — linear vs log mistaken for each other.
- **Contaminants/decoys** — `CON__`/`REV__` rows included or excluded by **explicit decision**, never by accident.
- **Protein groups/ambiguity** — semicolon-delimited members handled by explicit policy.
- **Mechanical parsing hazards** — locale/decimal separators, multi-row/merged headers, embedded metadata, duplicated/inconsistently named replicate columns.

## Output — `state/DATA_DESCRIPTION.md`

A verified description containing: orientation and shape; feature and sample identifier schemes; transformation/normalization state (incl. the `scale` the loader will record); missing-value semantics; contaminant/decoy handling decisions; the **preprocessing decisions confirmed above** (normalization method; batch axis + any confound; missing-value handling); known data-quality issues; and the **data-version stamp**. Same regeneration and stamping discipline as `METADATA.md`.

## Then

- Update `state/workflow.json`: `data_done: true`, `current_stage: 3`, bump `updated`.
- Next: **Stage 3 — Loaders, pairing, QC** (`stage3-loaders`) — the integrity gate. **No analysis may begin until that gate passes.**
