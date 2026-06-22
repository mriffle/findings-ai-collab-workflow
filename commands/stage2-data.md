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
- **Transformation state** — log vs linear (don't guess; check distributions and ranges).
- **Normalization state** — whether and how the data was normalized.
- **Missing-value encoding** — how missingness is represented, and whether tokens are interchangeable. **They are not:** `0`, `NA`, `NaN`, empty string, and tool tokens like `"Filtered"` mean different things. Conflating a true zero with missing-not-at-random changes every downstream statistic. Identify and decide deliberately.
- **Characterize missingness structure, contaminants, decoys, and duplicates.**
- **Run tests against your current understanding and surface problems.** Assumptions are hypotheses (doc 05).

### Watch for the domain fidelity traps (doc 05.4)

- **Spreadsheet identifier corruption** — gene symbols turned into dates (SEPT/MARCH/DEC), accessions in scientific notation, stripped leading zeros. Assume it happened until proven otherwise.
- **Scale confusion** — linear vs log mistaken for each other.
- **Contaminants/decoys** — `CON__`/`REV__` rows included or excluded by **explicit decision**, never by accident.
- **Protein groups/ambiguity** — semicolon-delimited members handled by explicit policy.
- **Mechanical parsing hazards** — locale/decimal separators, multi-row/merged headers, embedded metadata, duplicated/inconsistently named replicate columns.

## Output — `state/DATA_DESCRIPTION.md`

A verified description containing: orientation and shape; feature and sample identifier schemes; transformation/normalization state; missing-value semantics; contaminant/decoy handling decisions; known data-quality issues; and the **data-version stamp**. Same regeneration and stamping discipline as `METADATA.md`.

## Then

- Update `state/workflow.json`: `data_done: true`, `current_stage: 3`, bump `updated`.
- Next: **Stage 3 — Loaders, pairing, QC** (`stage3-loaders`) — the integrity gate. **No analysis may begin until that gate passes.**
