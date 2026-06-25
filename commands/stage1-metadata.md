---
name: stage1-metadata
description: "Stage 1 — Understand the metadata. Infer and validate the meaning of every column, test the relationships that should hold, detect confounds, and write state/METADATA.md."
---

# Stage 1 — Understand the metadata

**Precondition:** `state/workflow.json` shows `science_done: true`. If not, run `stage0-science` first and say so.

The goal is a *verified* understanding of the experiment's metadata — generated from evidence, never hand-asserted.

## Do this

1. **Locate the metadata file.** Ask the scientist where it is (typically in `data/`).
2. **Examine** its structure, columns, and value domains.
3. **Infer the meaning of each column** from names, values, and domains.
4. **Validate that understanding with the scientist** — this is a **human checkpoint** (doc 02.8). Present your inferred column meanings and design interpretation; get confirmation or correction before writing the file.
5. **Check value validity** — types, ranges, allowed sets, uniqueness where expected. Fail loud on anything inconsistent.
6. **Treat your inferences as hypotheses and test them in code** (doc 05): infer relationships that *should* hold if your understanding is correct, then test them — including whether the variable of interest is independent of batch, run order, and the key covariates (confounding; quantified and, where material, recorded in step 8). Assumptions are hypotheses, not assertions.
7. **Characterize the cohort — thorough descriptive plots and tables.** Generate, per the visualization conventions (dual export, Okabe–Ito via the color registry): the **distribution of every metadata variable** (sample counts per categorical level; summaries/histograms for continuous variables like age); **pairwise cross-tabulations** of the variables that matter — the variable of interest against each covariate and against batch/run-order (e.g. sex × group, age × sex, group × batch); and a publication-ready **cohort summary table ("Table 1")** broken down by the primary grouping. These are deliverables in their own right (papers, talks) *and* the lens for the next step. See `conventions/statistics.md` (descriptive characterization) for what to compute.
8. **Hunt for imbalance, bias, and confounding — and record the material ones as findings.** The point of the characterization is not just figures: it is to surface **class imbalance** (a grouping dominated by one level, severely unequal arm sizes), **covariate skew** (age/sex/etc. distributed unevenly across the contrast), and **confounding** (the variable of interest aliased with batch, run order, sex, or another factor — quantify with bias-corrected Cramér's V). A confounded or imbalanced design changes what every downstream finding can claim. **For each material gotcha, dispatch the findings-manager to record a caveat finding** (`kind: caveat`, `status: candidate`, `phase: exploratory`, `integrity_signoff: false`) capturing the evidence — the crosstab, the arm sizes, the confounding statistic — and the interpretive risk. The threshold is judgment, not a rule (suggestive cutoffs in `conventions/statistics.md`): record what would change a downstream analysis or its interpretation, not every histogram. These candidates have their `integrity_signoff` set at the integrity gate (Stage 3), which certifies the pairing they rest on.
9. **Extend the color registry.** Now that the design is understood, add project-specific categorical dimensions (treatment arms, cell lines, timepoints) to `state/color_registry.json` with `scope: project`, so every figure colors them consistently from the start.

## Output — `state/METADATA.md`

A verified, human- and agent-readable description containing:

- every column with its inferred meaning and validated type/domain;
- the experimental design it encodes;
- detected relationships, **class imbalances, covariate skews, and confounds** — each material one with a pointer to the caveat finding (`kind: caveat`) recorded for it;
- the **join key** to the data matrix (how samples in metadata map to columns/rows in the data);
- a **data-version stamp** (so the file cannot silently drift from the data it describes).

This is the canonical reference for *what the experiment is*. Regenerate it from verified understanding; never edit it into inconsistency with the data.

## Then

- Update `state/workflow.json`: `metadata_done: true`, `current_stage: 2`, bump `updated`.
- Next: **Stage 2 — Understand the data** (`stage2-data`).
